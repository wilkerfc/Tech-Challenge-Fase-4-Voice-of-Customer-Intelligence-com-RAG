"""Smoke evaluation real: proxy lexical, não substitui rótulos humanos."""
import argparse
import json
import sys
import unicodedata
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voc.retrieval import Retriever
from voc.pipeline import answer_question


def plain(text):
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if not unicodedata.combining(c))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/index")
    parser.add_argument("--output", default="evaluation/results.json")
    args = parser.parse_args()
    cases = json.loads(Path("evaluation/questions.json").read_text(encoding="utf-8"))
    retriever, results = Retriever(args.index), []
    for case in cases:
        result = answer_question(retriever, case["question"], score_min=case.get("score_min", 1), score_max=case.get("score_max", 5))
        terms = case.get("expected_terms", [])
        hits = sum(any(t in plain(e["text"]) for t in terms) for e in result["evidence"])
        results.append({"question": case["question"], "expected_refusal": case.get("expected_refusal", False), "insufficient": result["insufficient"],
                        "lexical_precision_proxy_at_k": hits / len(result["evidence"]) if terms and result["evidence"] else None, "result": result})
    output = {"notice": "Proxy lexical em perguntas escolhidas; não mede entailment, recall, nem desempenho de geração por LLM.",
              "manifest": retriever.manifest, "cases": results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{k: v for k, v in r.items() if k != "result"} for r in results], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
