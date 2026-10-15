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


def delivery_status(order):
    """Distinguish observed delivery, an explicit terminal failure and missing data.

    Missing delivery dates alone say nothing about the outcome. In particular,
    shipped/processing orders are not classified as 'not_delivered'.
    """
    actual = (order.get("order_delivered_customer_date") or "")[:10]
    estimated = (order.get("order_estimated_delivery_date") or "")[:10]
    if actual and estimated:
        return "late" if actual > estimated else "on_time"
    if not actual and order.get("order_status") in {"canceled", "unavailable"}:
        return "not_delivered"
    return "unknown"


def prepare(directory):
    directory = Path(directory)
    orders = {r["order_id"]: r for r in rows(directory / "olist_orders_dataset.csv")} if (directory / "olist_orders_dataset.csv").exists() else {}
    customers = {r["customer_id"]: r.get("customer_state", "").upper() for r in rows(directory / "olist_customers_dataset.csv")} if (directory / "olist_customers_dataset.csv").exists() else {}
    products = {r["product_id"]: r.get("product_category_name", "") for r in rows(directory / "olist_products_dataset.csv")} if (directory / "olist_products_dataset.csv").exists() else {}
    categories = defaultdict(set)
    if (directory / "olist_order_items_dataset.csv").exists():
        for item in rows(directory / "olist_order_items_dataset.csv"):
            category = products.get(item["product_id"], "")
            if category:
                categories[item["order_id"]].add(category)
    stats, documents, seen = Counter(), [], set()
    review_orders = defaultdict(set)
    order_metadata = {}
    for row in rows(directory / "olist_order_reviews_dataset.csv"):
        stats["rows_read"] += 1
        review_orders[row["review_id"]].add(row["order_id"])
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
        status = delivery_status(order)
        state = customers.get(order.get("customer_id"), "")
        metadata = {"order_id": row["order_id"],
                    "purchase_date": order.get("order_purchase_timestamp", "")[:10],
                    "categories": sorted(categories[row["order_id"]]), "state": state,
                    "delivery_status": status, "late_delivery": late,
                    "order_status": order.get("order_status", "unknown"),
                    "delivered_date": actual[:10], "estimated_delivery_date": estimated[:10]}
        order_metadata[row["order_id"]] = metadata
        doc_id = hashlib.sha256("|".join(key).encode()).hexdigest()[:20]
        documents.append({"id": doc_id, "review_id": row["review_id"], "order_id": row["order_id"], "text": text, "score": score,
                          "review_date": row.get("review_creation_date", "")[:10], "purchase_date": order.get("order_purchase_timestamp", "")[:10],
                          "categories": sorted(categories[row["order_id"]]), "late_delivery": late,
                          "state": state, "uf": state, "delivery_status": status})
    # Keep original IDs/texts and one row per (review, order, text): old vectors
    # remain aligned. The review's complete order linkage is explicit, so a
    # consumer never has to silently choose an order to count a unique review.
    for doc in documents:
        linked = sorted(review_orders[doc["review_id"]])
        for oid in linked:
            if oid not in order_metadata:
                order = orders.get(oid, {})
                status = delivery_status(order)
                order_metadata[oid] = {"order_id": oid,
                    "purchase_date": order.get("order_purchase_timestamp", "")[:10],
                    "categories": sorted(categories[oid]),
                    "state": customers.get(order.get("customer_id"), ""),
                    "delivery_status": status,
                    "late_delivery": True if status == "late" else False if status == "on_time" else None,
                    "order_status": order.get("order_status", "unknown"),
                    "delivered_date": order.get("order_delivered_customer_date", "")[:10],
                    "estimated_delivery_date": order.get("order_estimated_delivery_date", "")[:10]}
        doc["order_ids"] = linked
        doc["order_metadata"] = [dict(order_metadata[oid]) for oid in linked]
        doc["states"] = sorted({m["state"] for m in doc["order_metadata"] if m["state"]})
    stats["documents"] = len(documents)
    stats["score_distribution"] = dict(Counter(str(d["score"]) for d in documents))
    stats["unique_textual_reviews"] = len({d["review_id"] for d in documents})
    stats["reviews_with_multiple_orders"] = len({d["review_id"] for d in documents if len(d["order_ids"]) > 1})
    stats["delivery_status_distribution_documents"] = dict(Counter(d["delivery_status"] for d in documents))
    stats["metadata_schema"] = 2
    return documents, dict(stats)
