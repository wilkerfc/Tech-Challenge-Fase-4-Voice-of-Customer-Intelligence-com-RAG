"""Baixa exclusivamente os CSVs necessários do endpoint oficial do Kaggle."""
import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce"
FILES = ["olist_order_reviews_dataset.csv", "olist_orders_dataset.csv", "olist_order_items_dataset.csv", "olist_products_dataset.csv"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(URL, headers={"User-Agent": "Olist-VOC-Tech-Challenge/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    checksums = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for filename in FILES:
            content = archive.read(filename)
            (output / filename).write_bytes(content)
            checksums[filename] = hashlib.sha256(content).hexdigest()
    (output / "provenance.json").write_text(json.dumps({"source": URL, "sha256": checksums}, indent=2), encoding="utf-8")
    print("CSV oficiais baixados para", output)


if __name__ == "__main__":
    main()
