"""Embeddings multilíngues, cosine similarity exata e diversidade por MMR."""
import json
import re
import unicodedata
from pathlib import Path
import numpy as np

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def normalize_query(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


class Encoder:
    def __init__(self, model=MODEL, quantized=False):
        from sentence_transformers import SentenceTransformer
        import torch
        import os
        torch.set_num_threads(max(1, int(os.environ.get("VOC_CPU_THREADS", "1"))))
        self.model = SentenceTransformer(model)
        self.quantized = quantized
        if quantized:
            self.model[0].auto_model = torch.ao.quantization.quantize_dynamic(
                self.model[0].auto_model, {torch.nn.Linear}, dtype=torch.qint8)

    def encode(self, texts):
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=len(texts) > 1).astype("float32")


def build_index(documents, directory, encoder, model=MODEL, stats=None):
    if not documents:
        raise ValueError("Nenhuma avaliação textual válida para indexar.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Fragmentos curtos evitam truncamento silencioso dos comentários longos.
    chunks = []
    for doc in documents:
        words = doc["text"].split()
        for start in range(0, len(words), 70):
            chunk = dict(doc, text=" ".join(words[start:start + 90]), chunk=start // 70)
            chunks.append(chunk)
            if start + 90 >= len(words):
                break
    vectors = encoder.encode([d["text"] for d in chunks])
    np.save(directory / "vectors.npy", vectors)
    (directory / "documents.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps({"model": model, "quantized": getattr(encoder, "quantized", False), "chunks": len(chunks), "dimension": vectors.shape[1], "stats": stats or {}, "schema": 1}, ensure_ascii=False, indent=2), encoding="utf-8")


class Retriever:
    def __init__(self, directory, encoder=None):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        self.docs = json.loads((directory / "documents.json").read_text(encoding="utf-8"))
        self.vectors = np.load(directory / "vectors.npy", allow_pickle=False)
        if len(self.docs) != len(self.vectors):
            raise ValueError("Índice inconsistente: documentos e vetores divergem.")
        self.encoder = encoder or Encoder(self.manifest["model"], quantized=self.manifest.get("quantized", False))

    def search(self, question, k=6, min_similarity=0.30, score_min=1, score_max=5, category=None, date_from=None, date_to=None, late_only=False):
        question = normalize_query(question)
        if not question or not 1 <= k <= 20:
            raise ValueError("Informe uma pergunta e k entre 1 e 20.")
        eligible = [i for i, d in enumerate(self.docs) if score_min <= d["score"] <= score_max
                    and (not category or category in d["categories"])
                    and (not date_from or (d["purchase_date"] and d["purchase_date"] >= date_from))
                    and (not date_to or (d["purchase_date"] and d["purchase_date"] <= date_to))
                    and (not late_only or d["late_delivery"] is True)]
        if not eligible:
            return []
        query = self.encoder.encode([question])[0]
        similarities = self.vectors[eligible] @ query
        ranking = np.argsort(-similarities, kind="stable")[:max(k * 8, 40)]
        candidates = [(eligible[int(i)], float(similarities[i])) for i in ranking if similarities[i] >= min_similarity]
        selected, ids = [], set()
        while candidates and len(selected) < k:
            def mmr(pair):
                redundancy = max((float(self.vectors[pair[0]] @ self.vectors[s[0]]) for s in selected), default=0)
                return 0.8 * pair[1] - 0.2 * redundancy
            best = max(candidates, key=mmr)
            candidates.remove(best)
            if self.docs[best[0]]["id"] in ids:
                continue
            selected.append(best)
            ids.add(self.docs[best[0]]["id"])
        return [dict(self.docs[i], similarity=round(sim, 5), citation=f"E{n}") for n, (i, sim) in enumerate(selected, 1)]
