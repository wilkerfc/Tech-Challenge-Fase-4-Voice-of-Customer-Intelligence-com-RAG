"""Benchmark pareado real: mesmo índice/perguntas/filtros, rótulos silver.

Não chama LLM. Reproduz ranking legado (denso+MMR .8, candidatos=max(8k,40))
e compara variantes novas. Regex são proxies publicados, não auditoria humana.
"""
import argparse
import csv
import json
import hashlib
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voc.query import matches_filters
from voc.retrieval import Retriever, RERANKER_MODEL, normalize_query, plain, text_key


def legacy_search(retriever, question, k=6, min_similarity=.30, **filters):
    """Algoritmo do commit a39930b, usando os mesmos filtros atuais da comparação."""
    eligible = [i for i, d in enumerate(retriever.docs) if matches_filters(d, **filters)]
    if not eligible:
        return []
    q = retriever.encoder.encode([normalize_query(question)])[0]
    similarities = retriever.vectors[eligible] @ q
    ranking = np.argsort(-similarities, kind="stable")[:max(k * 8, 40)]
    candidates = [(eligible[int(i)], float(similarities[i])) for i in ranking if similarities[i] >= min_similarity]
    selected, ids = [], set()
    while candidates and len(selected) < k:
        def mmr(pair):
            redundancy = max((float(retriever.vectors[pair[0]] @ retriever.vectors[s[0]]) for s in selected), default=0)
            return .8 * pair[1] - .2 * redundancy
        best = max(candidates, key=mmr)
        candidates.remove(best)
        if retriever.docs[best[0]]["id"] in ids:
            continue
        selected.append(best)
        ids.add(retriever.docs[best[0]]["id"])
    return [dict(retriever.docs[i], similarity=round(sim,5)) for i, sim in selected]


def main():
    import re
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="data/index")
    parser.add_argument("--questions", default="evaluation/retrieval-questions.json")
    parser.add_argument("--output", default="evaluation/retrieval-benchmark.json")
    parser.add_argument("--split", choices=["development", "holdout"], default="development")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--reranker", nargs="?", const=RERANKER_MODEL)
    args = parser.parse_args()
    cases = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    started = time.perf_counter()
    retriever = Retriever(args.index)
    init_ms = (time.perf_counter() - started) * 1000
    retriever.encoder.encode(["Aquecimento do encoder de avaliações de clientes"])
    variants = {"legacy_dense_mmr": None, "dense_dedup_mmr": {"algorithm":"dense","use_mmr":True},
                "hybrid_rrf": {"algorithm":"hybrid","use_mmr":False},
                "hybrid_rrf_mmr": {"algorithm":"hybrid","use_mmr":True}}
    if args.reranker:
        from voc.retrieval import CrossEncoderReranker
        variants["hybrid_reranker"] = {"algorithm":"hybrid","use_mmr":False,"reranker":CrossEncoderReranker(args.reranker)}
    records = []
    for case in cases:
        for name, options in variants.items():
            t0 = time.perf_counter()
            evidence = (legacy_search(retriever, case["question"], k=args.k, **case.get("filters", {})) if options is None
                        else retriever.search(case["question"], k=args.k, **case.get("filters", {}), **options))
            elapsed = (time.perf_counter() - t0) * 1000
            hits = [bool(re.search(case["relevance_regex"], plain(e["text"]))) for e in evidence]
            first = next((i for i, hit in enumerate(hits, 1) if hit), None)
            records.append({"id":case["id"],"question":case["question"],"filters":case.get("filters",{}),"variant":name,
                "silver_regex":case["relevance_regex"],"precision_at_k":sum(hits)/args.k,"mrr":1/first if first else 0,
                "returned":len(evidence),"unique_text_fraction":len({text_key(e["text"]) for e in evidence})/len(evidence) if evidence else 0,
                "latency_ms":round(elapsed,2),"evidence":[dict(id=e["id"],review_id=e["review_id"],order_id=e["order_id"],
                     text=e["text"],similarity=e["similarity"],silver_relevant=hit,bm25_score=e.get("bm25_score"),rrf_score=e.get("rrf_score"),
                     reranker_score=e.get("reranker_score")) for e,hit in zip(evidence,hits)]})
    summaries = []
    for name in variants:
        rows = [r for r in records if r["variant"] == name]
        summaries.append({"variant":name,"questions":len(rows),"precision_at_k":round(statistics.mean(r["precision_at_k"] for r in rows),4),
            "mrr":round(statistics.mean(r["mrr"] for r in rows),4),"unique_text_fraction":round(statistics.mean(r["unique_text_fraction"] for r in rows),4),
            "median_latency_ms":round(statistics.median(r["latency_ms"] for r in rows),2)})
    hashes = {}
    for filename in ("documents.json", "vectors.npy", "corpus.json"):
        if (Path(args.index)/filename).exists():
            with (Path(args.index)/filename).open("rb") as handle:
                hashes[filename] = hashlib.file_digest(handle,"sha256").hexdigest()
    output = {"created_utc":datetime.now(timezone.utc).isoformat(),"k":args.k,"split":args.split,
        "questions_file":Path(args.questions).name,"questions_sha256":hashlib.sha256(Path(args.questions).read_bytes()).hexdigest(),
        "index_manifest":retriever.manifest,
        "index_file_sha256":hashes,"configuration":{"min_similarity":0.30,"mmr_lambda":0.8,"rrf_k":60,
        "legacy_candidate_pool":max(args.k*8,40),"new_candidate_pool":100,"reranker_model":args.reranker},
        "index_documents":len(retriever.docs),"initialization_ms":round(init_ms,2),"summary":summaries,
        "limitations":[f"{len(cases)} perguntas no conjunto {args.split}; relevância silver por regex, sem rótulos humanos independentes.",
          "Regex podem favorecer BM25 e não reconhecem toda paráfrase ou negação: resultado não mede entailment.",
          "Mesmas perguntas/filtros/vetores nas variantes; baseline legado reproduz ranking anterior com filtros atuais.",
          "Precisão divide pelo k solicitado inclusive quando retorna menos; MRR usa primeira evidência que casa regex.",
          "Latência após aquecimento inclui encoder/busca/filtros/MMR, exclui carga inicial; uma passagem, sem intervalo estatístico.",
          "Nenhuma métrica permite comparar diretamente com B: perguntas, k, corpus, modelos e rótulos diferentes.",
          "Sem chamadas a LLM e sem auditoria humana nova. Reranker só avaliado se --reranker for fornecido.",
          "Padrão hybrid_rrf sem MMR escolhido nas 16 perguntas de desenvolvimento e congelado antes das 8 holdout; sem ajuste posterior ao holdout."],"records":records}
    target = Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
    with target.with_suffix(".csv").open("w",encoding="utf-8-sig",newline="") as handle:
        fields=["id","variant","precision_at_k","mrr","returned","unique_text_fraction","latency_ms"]
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore")
        writer.writeheader(); writer.writerows(records)
    markdown = [f"# Recuperação: benchmark pareado real ({args.split})", "", f"Corpus: {len(retriever.docs):,} chunks; k={args.k}; {len(cases)} perguntas. Mesmo MiniLM e vetores em todas as variantes.","",
        "| Variante | Precisão silver@k | MRR silver | Textos únicos | Latência mediana ms |", "|---|---:|---:|---:|---:|"]
    markdown += [f"| {r['variant']} | {r['precision_at_k']:.4f} | {r['mrr']:.4f} | {r['unique_text_fraction']:.4f} | {r['median_latency_ms']:.2f} |" for r in summaries]
    markdown += ["", "## Mesmas perguntas, resultados por caso", "", "A média não significa melhoria em cada pergunta. Valores são precisão silver@k.", "",
        "| ID | Pergunta | Legado denso+MMR | Híbrido RRF | Híbrido RRF+MMR |", "|---|---|---:|---:|---:|"]
    for case in cases:
        values = {r["variant"]:r["precision_at_k"] for r in records if r["id"]==case["id"]}
        markdown.append(f"| {case['id']} | {case['question']} | {values['legacy_dense_mmr']:.3f} | {values['hybrid_rrf']:.3f} | {values['hybrid_rrf_mmr']:.3f} |")
    markdown += ["", "Cosseno mínimo 0,30 (heurístico, sem calibração nova); MMR λ=0,8; RRF k=60; pool legado=max(8k,40), novo=100. Normalização BM25 PT leve, não stemmer linguístico completo. A escolha de variante após consultar este conjunto é exploratória; exige validação em perguntas novas."]
    markdown += ["", "## Limites", ""]+[f"- {item}" for item in output["limitations"]]
    markdown += ["", "IDs, textos, regex, filtros e tempos por pergunta estão em retrieval-benchmark.json e no CSV. Hashes do índice registram os dados realmente usados."]
    target.with_suffix(".md").write_text("\n".join(markdown)+"\n",encoding="utf-8")
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
