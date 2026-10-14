"""Ingestão determinística da Olist, sem dependência de pandas."""
import csv
import hashlib
import html
import re
from collections import Counter, defaultdict
from pathlib import Path


def clean(text):
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", "[EMAIL]", text)
    text = re.sub(r"(?<!\d)(?:\+55\s*)?\(?\d{2}\)?[\s.-]*\d{4,5}[\s.-]*\d{4}(?!\d)", "[TELEFONE]", text)
    return re.sub(r"\s+", " ", text).strip()


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        yield from csv.DictReader(file)


def prepare(directory):
    directory = Path(directory)
    orders = {r["order_id"]: r for r in rows(directory / "olist_orders_dataset.csv")} if (directory / "olist_orders_dataset.csv").exists() else {}
    products = {r["product_id"]: r.get("product_category_name", "") for r in rows(directory / "olist_products_dataset.csv")} if (directory / "olist_products_dataset.csv").exists() else {}
    categories = defaultdict(set)
    if (directory / "olist_order_items_dataset.csv").exists():
        for item in rows(directory / "olist_order_items_dataset.csv"):
            category = products.get(item["product_id"], "")
            if category:
                categories[item["order_id"]].add(category)
    stats, documents, seen = Counter(), [], set()
    for row in rows(directory / "olist_order_reviews_dataset.csv"):
        stats["rows_read"] += 1
        text = clean(" ".join(filter(None, [row.get("review_comment_title"), row.get("review_comment_message")])))
        if not text:
            stats["empty_text"] += 1
            continue
        try:
            score = int(row["review_score"])
            if not 1 <= score <= 5:
                raise ValueError()
        except (ValueError, KeyError):
            stats["invalid_score"] += 1
            continue
        # Olist possui review_id repetido para pedidos diferentes: preservar ambos.
        key = (row["review_id"], row["order_id"], text)
        if key in seen:
            stats["duplicates_removed"] += 1
            continue
        seen.add(key)
        order = orders.get(row["order_id"], {})
        actual, estimated = order.get("order_delivered_customer_date", ""), order.get("order_estimated_delivery_date", "")
        late = actual[:10] > estimated[:10] if actual and estimated else None
        doc_id = hashlib.sha256("|".join(key).encode()).hexdigest()[:20]
        documents.append({"id": doc_id, "review_id": row["review_id"], "order_id": row["order_id"], "text": text, "score": score,
                          "review_date": row.get("review_creation_date", "")[:10], "purchase_date": order.get("order_purchase_timestamp", "")[:10],
                          "categories": sorted(categories[row["order_id"]]), "late_delivery": late})
    stats["documents"] = len(documents)
    stats["score_distribution"] = dict(Counter(str(d["score"]) for d in documents))
    return documents, dict(stats)
