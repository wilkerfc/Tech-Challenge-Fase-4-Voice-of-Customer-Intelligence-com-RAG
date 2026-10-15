"""Reattach complete Olist metadata without downloading models/re-embedding.

Run --dry-run first to compare every saved (id, chunk, text) with preparation.
The actual refresh verifies the saved integrity manifest and writes corpus.json,
documents.json and a new integrity manifest; vectors.npy must remain identical.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voc.data import prepare
from voc.retrieval import _chunks, refresh_metadata


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw", help="Directory with original Olist CSV files")
    parser.add_argument("--index", default="data/index")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    directory = Path(args.index)
    documents, stats = prepare(args.data)
    before = sha256(directory / "vectors.npy")
    saved = json.loads((directory / "documents.json").read_text(encoding="utf-8"))
    candidate = _chunks(documents)
    identity = lambda doc: (doc["id"], doc.get("chunk", 0), doc["text"])
    if [identity(doc) for doc in saved] != [identity(doc) for doc in candidate]:
        raise ValueError("IDs/textos/ordem diferem do índice: é necessário reconstruir os vetores.")
    summary = {"dry_run": args.dry_run, "documents": len(documents), "chunks": len(candidate),
               "unique_textual_reviews": stats["unique_textual_reviews"],
               "text_and_identity_unchanged": True, "vector_sha256_before": before,
               "metadata_schema": stats["metadata_schema"]}
    if not args.dry_run:
        manifest = refresh_metadata(directory, documents, stats=stats)
        after = sha256(directory / "vectors.npy")
        if after != before:
            raise RuntimeError("A atualização de metadados alterou os vetores inesperadamente.")
        summary.update(vector_sha256_after=after, vectors_unchanged=True,
                       integrity=manifest.get("integrity"), schema=manifest.get("schema"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
