import argparse
import json
from pathlib import Path
from voc.data import prepare
from voc.retrieval import Encoder, Retriever, build_index
from voc.pipeline import answer_question


def main():
    parser = argparse.ArgumentParser(description="Voice of Customer Olist")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("--raw", default="data/raw")
    ingest.add_argument("--index", default="data/index")
    ingest.add_argument("--limit", type=int, help="Amostra determinística; omita para toda a base")
    ingest.add_argument("--quantize", action="store_true", help="Quantização int8 das camadas lineares para CPU")
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--index", default="data/index")
    ask.add_argument("--mode", choices=["extractive", "llm"], default="extractive")
    ask.add_argument("--max-score", type=int, default=5)
    ask.add_argument("--min-score", type=int, default=1)
    ask.add_argument("--category")
    ask.add_argument("--state")
    ask.add_argument("--date-from")
    ask.add_argument("--date-to")
    ask.add_argument("--delivery-status", choices=["late", "on_time", "not_delivered", "unknown"])
    ask.add_argument("--algorithm", choices=["hybrid", "dense"], default="hybrid")
    ask.add_argument("--route", choices=["auto", "rag", "analytics"], default="auto")
    ask.add_argument("--reranker", help="Modelo cross-encoder opcional; baixa pesos na primeira execução")
    ask.add_argument("--output", help="Salvar resultado JSON neste arquivo")
    args = parser.parse_args()
    if args.command == "ingest":
        docs, stats = prepare(args.raw)
        if args.limit is not None:
            if args.limit < 1:
                parser.error("--limit deve ser positivo")
            docs = docs[:args.limit]
        stats["indexed_documents"] = len(docs)
        build_index(docs, args.index, Encoder(quantized=args.quantize), stats=stats)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    else:
        filters = {"algorithm": args.algorithm}
        if args.max_score != 5:
            filters["score_max"] = args.max_score
        if args.min_score != 1:
            filters["score_min"] = args.min_score
        for key in ["category", "state", "date_from", "date_to", "delivery_status", "reranker"]:
            value = getattr(args, key)
            if value is not None:
                filters[key] = value
        result = answer_question(Retriever(args.index), args.question, mode=args.mode, route=args.route, **filters)
        content = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(content, encoding="utf-8")
        print(content)


if __name__ == "__main__":
    main()
