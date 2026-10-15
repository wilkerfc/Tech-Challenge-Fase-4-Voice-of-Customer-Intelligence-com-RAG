import csv
import hashlib

from voc.data import delivery_status, prepare


def write_csv(directory, name, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (directory / name).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def test_delivery_missing_date_is_unknown_not_a_failure_or_on_time():
    assert delivery_status({"order_status": "delivered"}) == "unknown"
    assert delivery_status({"order_status": "shipped"}) == "unknown"
    assert delivery_status({"order_delivered_customer_date": "2018-01-01"}) == "unknown"
    assert delivery_status({"order_status": "canceled"}) == "not_delivered"
    assert delivery_status({"order_status": "unavailable"}) == "not_delivered"
    assert delivery_status({"order_delivered_customer_date": "2018-01-02", "order_estimated_delivery_date": "2018-01-01"}) == "late"
    assert delivery_status({"order_delivered_customer_date": "2018-01-01 22:00", "order_estimated_delivery_date": "2018-01-01 00:00"}) == "on_time"


def test_prepare_preserves_original_identity_text_and_all_linked_orders(tmp_path):
    text = "Produto excelente"
    write_csv(tmp_path, "olist_order_reviews_dataset.csv", [
        {"review_id": "r", "order_id": "o1", "review_score": "5", "review_comment_title": "", "review_comment_message": text},
        {"review_id": "r", "order_id": "o2", "review_score": "5", "review_comment_title": "", "review_comment_message": text},
        {"review_id": "r", "order_id": "o1", "review_score": "5", "review_comment_title": "", "review_comment_message": text},
    ])
    write_csv(tmp_path, "olist_orders_dataset.csv", [
        {"order_id": "o1", "customer_id": "c1", "order_purchase_timestamp": "2018-01-01 12:00", "order_status": "delivered", "order_delivered_customer_date": "2018-01-05", "order_estimated_delivery_date": "2018-01-04"},
        {"order_id": "o2", "customer_id": "c2", "order_purchase_timestamp": "2017-02-01 12:00", "order_status": "shipped"},
    ])
    write_csv(tmp_path, "olist_customers_dataset.csv", [
        {"customer_id": "c1", "customer_state": "SP"}, {"customer_id": "c2", "customer_state": "RJ"},
    ])
    docs, stats = prepare(tmp_path)
    assert len(docs) == 2 and stats["duplicates_removed"] == 1
    assert docs[0]["id"] == hashlib.sha256("r|o1|Produto excelente".encode()).hexdigest()[:20]
    assert docs[0]["text"] == text
    assert all(doc["order_ids"] == ["o1", "o2"] for doc in docs)
    assert docs[0]["state"] == "SP" and docs[1]["state"] == "RJ"
    assert docs[0]["states"] == ["RJ", "SP"]
    assert docs[0]["delivery_status"] == "late" and docs[1]["delivery_status"] == "unknown"
    assert stats["unique_textual_reviews"] == 1 and stats["reviews_with_multiple_orders"] == 1
