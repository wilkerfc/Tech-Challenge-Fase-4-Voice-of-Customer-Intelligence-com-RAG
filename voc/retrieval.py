"""Denso + BM25 PT, RRF, deduplicação e MMR; reranking somente opt-in.

``similarity`` continua sendo o cosseno denso. BM25 usa normalização leve,
não um stemmer linguístico completo. Nenhuma busca padrão baixa reranker.
"""
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
_STOP = set("a o as os um uma uns umas de da do das dos em no na nos nas para por com e ou que se ao aos pelo pela pelos pelas eu voce voces ele ela eles elas meu minha seus suas foi ser esta esse essa isso muito mais qual quais como sobre clientes cliente avaliacoes avaliacao relatos relatam dizem".split())


def normalize_query(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def plain(text):
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if not unicodedata.combining(c))


def lexical_tokens(text):
    """Normaliza plural/adjetivos comuns igualmente na consulta e corpus.

    Preserva não/nunca/sem. Não captura sinônimos: esse papel é do encoder.
    """
    output = []
    for word in re.findall(r"[a-z0-9]+", plain(text)):
        if word in _STOP or len(word) < 2:
            continue
        if len(word) > 4 and word.endswith("oes"):
            word = word[:-3] + "ao"
        elif len(word) > 4 and word.endswith("s") and not word.endswith(("is", "us")):
            word = word[:-1]
        if len(word) > 5:
            for suffix in ("mente", "ado", "ada", "ido", "ida"):
                if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                    word = word[:-len(suffix)]
                    break
        output.append(word)
    return output


def text_key(text):
    return " ".join(re.findall(r"[a-z0-9]+", plain(text)))


class BM25:
    """BM25 Okapi/IDF positivo com postings NumPy; sem novas dependências."""
    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        counts = [Counter(lexical_tokens(t)) for t in texts]
        self.lengths = np.asarray([sum(c.values()) for c in counts], dtype=np.float32)
        self.avg_length = max(float(self.lengths.mean()) if counts else 0, 1.0)
        postings = defaultdict(list)
        for index, count in enumerate(counts):
            for token, frequency in count.items():
                postings[token].append((index, frequency))
        self.postings = {token: (np.asarray([i for i, _ in pairs]), np.asarray([f for _, f in pairs], dtype=np.float32))
                         for token, pairs in postings.items()}

    def scores(self, question):
        scores = np.zeros(len(self.lengths), dtype=np.float32)
        for token in sorted(set(lexical_tokens(question))):
            if token not in self.postings:
                continue
            ids, tf = self.postings[token]
            idf = math.log(1 + (len(self.lengths) - len(ids) + 0.5) / (len(ids) + 0.5))
            norm = self.k1 * (1 - self.b + self.b * self.lengths[ids] / self.avg_length)
            scores[ids] += idf * tf * (self.k1 + 1) / (tf + norm)
        return scores


class Encoder:
    def __init__(self, model=MODEL, quantized=False, revision=None):
        from sentence_transformers import SentenceTransformer
        import torch
        import os
        torch.set_num_threads(max(1, int(os.environ.get("VOC_CPU_THREADS", "1"))))
        self.revision = revision or (MODEL_REVISION if model == MODEL else None)
        self.model = SentenceTransformer(model, revision=self.revision)
        self.quantized = quantized
        if quantized:
            self.model[0].auto_model = torch.ao.quantization.quantize_dynamic(
                self.model[0].auto_model, {torch.nn.Linear}, dtype=torch.qint8)

    def encode(self, texts):
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=len(texts) > 1).astype("float32")


class CrossEncoderReranker:
    def __init__(self, model=RERANKER_MODEL):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model, max_length=256)
        self.name = model

    def scores(self, question, texts):
        return np.asarray(self.model.predict([(question, t) for t in texts], show_progress_bar=False), dtype=float).reshape(-1)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _chunks(documents):
    chunks = []
    for doc in documents:
        words = doc["text"].split()
        for start in range(0, len(words), 70):
            chunks.append(dict(doc, text=" ".join(words[start:start + 90]), chunk=start // 70))
            if start + 90 >= len(words):
                break
    return chunks


def upgrade_index(directory):
    """Atesta arquivos atuais em schema 2 sem recalcular vetores.

    A migração não certifica procedência. Mudanças posteriores são detectadas.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    docs = json.loads((directory / "documents.json").read_text(encoding="utf-8"))
    vectors = np.load(directory / "vectors.npy", mmap_mode="r", allow_pickle=False)
    if vectors.ndim != 2 or vectors.shape != (len(docs), manifest["dimension"]) or not np.isfinite(vectors).all():
        raise ValueError("Índice inconsistente: quantidade/dimensão/valores dos vetores inválidos.")
    manifest.update(schema=2, chunks=len(docs), integrity={name: _sha256(directory / name)
                    for name in ("documents.json", "vectors.npy", "corpus.json") if (directory / name).exists()})
    _write_json(directory / "manifest.json", manifest)
    return manifest


def refresh_metadata(directory, documents, stats=None):
    """Atualiza corpus/metadados preservando texto, IDs, ordem e vetores.

    ``corpus.json`` guarda documentos inteiros, separado dos chunks do índice.
    Texto alterado exige reconstrução explícita para evitar vetores obsoletos.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest.get("integrity", {}).items():
        if not (directory / name).exists() or _sha256(directory / name) != digest:
            raise ValueError(f"Integridade inválida: {name}. Reconstrua o índice.")
    saved = json.loads((directory / "documents.json").read_text(encoding="utf-8"))
    updated = _chunks(documents)
    identity = lambda doc: (doc["id"], doc.get("chunk", 0), doc["text"])
    if [identity(d) for d in saved] != [identity(d) for d in updated]:
        raise ValueError("Texto/IDs/ordem mudaram: reconstrua o índice para não reutilizar vetores incorretos.")
    _write_json(directory / "documents.json", updated)
    _write_json(directory / "corpus.json", documents)
    if stats is not None:
        manifest["stats"] = stats
    _write_json(directory / "manifest.json", manifest)
    return upgrade_index(directory)


def build_index(documents, directory, encoder, model=MODEL, stats=None):
    if not documents:
        raise ValueError("Nenhuma avaliação textual válida para indexar.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    chunks = _chunks(documents)
    vectors = encoder.encode([d["text"] for d in chunks])
    np.save(directory / "vectors.npy", vectors)
    _write_json(directory / "documents.json", chunks)
    _write_json(directory / "corpus.json", documents)
    _write_json(directory / "manifest.json", {"model": model, "model_revision": getattr(encoder, "revision", None), "quantized": getattr(encoder, "quantized", False),
                "chunks": len(chunks), "dimension": vectors.shape[1], "stats": stats or {}, "schema": 1})
    upgrade_index(directory)


def rrf(rankings, k=60):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, index in enumerate(ranking, 1):
            scores[index] += 1.0 / (k + rank)
    return sorted(scores, key=lambda i: (-scores[i], i)), dict(scores)


class Retriever:
    def __init__(self, directory, encoder=None):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema", 1) >= 2:
            integrity = self.manifest.get("integrity", {})
            if not {"documents.json", "vectors.npy"} <= set(integrity):
                raise ValueError("Manifest schema 2 sem hashes obrigatórios.")
            for name, digest in integrity.items():
                if name not in {"documents.json", "vectors.npy", "corpus.json"} or not (directory / name).exists() or _sha256(directory / name) != digest:
                    raise ValueError(f"Integridade inválida: {name}. Reconstrua o índice.")
        self.docs = json.loads((directory / "documents.json").read_text(encoding="utf-8"))
        self.corpus = json.loads((directory / "corpus.json").read_text(encoding="utf-8")) if (directory / "corpus.json").exists() else self.docs
        self.vectors = np.load(directory / "vectors.npy", allow_pickle=False)
        if self.vectors.ndim != 2 or self.vectors.shape != (len(self.docs), self.manifest["dimension"]) or not np.isfinite(self.vectors).all():
            raise ValueError("Índice inconsistente: documentos e vetores divergem.")
        self.encoder = encoder or Encoder(self.manifest["model"], quantized=self.manifest.get("quantized", False), revision=self.manifest.get("model_revision"))
        self.bm25 = BM25([d["text"] for d in self.docs])
        self._keys = [text_key(d["text"]) for d in self.docs]
        self._rerankers = {}

    def search(self, question, k=6, min_similarity=0.30, score_min=1, score_max=5, category=None,
               date_from=None, date_to=None, late_only=False, uf=None, delivery_status=None, state=None,
               algorithm="hybrid", use_mmr=False, mmr_lambda=0.8, candidate_pool=100, reranker=None, rrf_k=60):
        """Pré-filtro, denso/BM25, RRF, dedup, reranking e MMR opcionais.

        Limiar SEMPRE de cosseno, não de probabilidade. Reranker é objeto com
        .scores(pergunta,textos) ou nome de modelo solicitado explicitamente.
        """
        from voc.query import matches_filters
        question = normalize_query(question)
        if not question or not 1 <= k <= 20:
            raise ValueError("Informe uma pergunta e k entre 1 e 20.")
        if algorithm not in {"dense", "hybrid"} or not 0 <= mmr_lambda <= 1 or candidate_pool < k or rrf_k < 1:
            raise ValueError("Configuração de recuperação inválida.")
        eligible = np.asarray([i for i, doc in enumerate(self.docs) if matches_filters(doc, score_min=score_min,
            score_max=score_max, category=category, date_from=date_from, date_to=date_to, late_only=late_only,
            uf=uf, state=state, delivery_status=delivery_status)], dtype=np.int64)
        if not len(eligible):
            return []
        query = np.asarray(self.encoder.encode([question])[0], dtype=np.float32)
        similarities = self.vectors[eligible] @ query
        similarity = {int(i): float(s) for i, s in zip(eligible, similarities)}
        dense = [int(eligible[j]) for j in np.argsort(-similarities, kind="stable")[:candidate_pool]]
        lexical = self.bm25.scores(question) if algorithm == "hybrid" else np.zeros(len(self.docs))
        bm25 = [int(eligible[j]) for j in np.argsort(-lexical[eligible], kind="stable")[:candidate_pool] if lexical[eligible[j]] > 0]
        fused, rrf_scores = rrf([dense, bm25] if algorithm == "hybrid" else [dense], rrf_k)
        dense_rank = {i: n for n, i in enumerate(dense, 1)}
        lexical_rank = {i: n for n, i in enumerate(bm25, 1)}
        seen_ids, seen_text, candidates = set(), set(), []
        for index in fused:
            if similarity[index] < min_similarity or self.docs[index]["id"] in seen_ids or self._keys[index] in seen_text:
                continue
            seen_ids.add(self.docs[index]["id"])
            seen_text.add(self._keys[index])
            candidates.append(index)
            if len(candidates) >= candidate_pool:
                break
        relevance = {i: (rrf_scores[i] * (rrf_k + 1) / 2 if algorithm == "hybrid" else similarity[i]) for i in candidates}
        rerank_scores, reranker_name = {}, None
        if reranker is not None and candidates:
            if isinstance(reranker, str):
                if reranker not in self._rerankers:
                    self._rerankers[reranker] = CrossEncoderReranker(reranker)
                reranker = self._rerankers[reranker]
            scores = np.asarray(reranker.scores(question, [self.docs[i]["text"] for i in candidates])).reshape(-1)
            if len(scores) != len(candidates) or not np.isfinite(scores).all():
                raise ValueError("Reranker retornou scores inválidos.")
            rerank_scores = dict(zip(candidates, map(float, scores)))
            candidates.sort(key=lambda i: (-rerank_scores[i], i))
            low, high = float(scores.min()), float(scores.max())
            relevance = {i: (rerank_scores[i] - low) / (high - low) if high > low else 1.0 for i in candidates}
            reranker_name = getattr(reranker, "name", type(reranker).__name__)
        selected = []
        while candidates and len(selected) < k:
            def utility(index):
                redundancy = max((float(self.vectors[index] @ self.vectors[other]) for other in selected), default=0)
                return mmr_lambda * relevance[index] - (1 - mmr_lambda) * redundancy
            best = max(candidates, key=utility) if use_mmr else candidates[0]
            candidates.remove(best)
            selected.append(best)
        from voc.query import matching_order_metadata
        return [dict(self.docs[i], similarity=round(similarity[i], 5), citation=f"E{n}",
                     bm25_score=round(float(lexical[i]), 5), rrf_score=round(rrf_scores.get(i, 0), 7),
                     dense_rank=dense_rank.get(i), lexical_rank=lexical_rank.get(i),
                     reranker_score=rerank_scores.get(i), reranker=reranker_name, retrieval_score=round(relevance[i], 7),
                     matching_orders=matching_order_metadata(self.docs[i], category=category,date_from=date_from,
                         date_to=date_to,late_only=late_only,uf=uf,state=state,delivery_status=delivery_status),
                     retrieval_method=algorithm + ("+reranker" if reranker_name else "") + ("+mmr" if use_mmr else ""))
                for n, i in enumerate(selected, 1)]
