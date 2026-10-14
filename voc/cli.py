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
        print(json.dumps(answer_question(Retriever(args.index), args.question, mode=args.mode, score_max=args.max_score), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
