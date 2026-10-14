import csv
import json
import numpy as np
import pytest
from voc.data import clean, prepare
from voc.retrieval import Retriever, build_index
from voc.pipeline import answer_question, validate_generation, build_context


class FakeEncoder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            text = text.lower()
            v = np.array([float("entrega" in text or "atras" in text), float("produto" in text), float("bolo" in text)], dtype="float32")
            vectors.append(v / max(float(np.linalg.norm(v)), 1))
        return np.array(vectors)


@pytest.fixture
def retriever(tmp_path):
    docs = [{"id": str(i), "review_id": str(i), "order_id": str(i), "text": text, "score": score, "purchase_date": date,
             "categories": [cat], "late_delivery": late} for i, (text, score, date, cat, late) in enumerate([
                 ("Entrega atrasada e sem aviso", 1, "2018-01-10", "moveis", True),
                 ("Entrega com atraso", 2, "2018-02-10", "moveis", True),
                 ("Produto excelente", 5, "2017-01-01", "beleza", False)])]
    build_index(docs, tmp_path, FakeEncoder(), model="fake")
    return Retriever(tmp_path, FakeEncoder())


def test_semantic_ranking_and_metadata(retriever):
    result = retriever.search("entrega", score_max=2, category="moveis", late_only=True, date_from="2018-02-01")
    assert [e["id"] for e in result] == ["1"]
    assert result[0]["citation"] == "E1"


def test_no_evidence_and_scope(retriever):
    assert answer_question(retriever, "receita de bolo")["insufficient"]
    assert answer_question(retriever, "entrega", category="inexistente")["insufficient"]
    assert answer_question(retriever, "produto")["insufficient"]  # apenas uma evidência


def test_traceability(retriever):
    result = answer_question(retriever, "entrega")
    assert not result["insufficient"]
    assert "[E1]" in result["answer"] and "[E2]" in result["answer"]
    assert len({e["id"] for e in result["evidence"]}) == 2


def test_unknown_and_missing_citations_rejected():
    evidence = [{"citation": "E1"}]
    with pytest.raises(ValueError):
        validate_generation({"answer": "Problema [E9]", "citations": ["E9"], "insufficient": False}, evidence)
    with pytest.raises(ValueError):
        validate_generation({"answer": "Afirmação sem fonte", "citations": [], "insufficient": False}, evidence)


def test_llm_failure_is_explicit(retriever, monkeypatch):
    import requests
    def fail(*args, **kwargs):
        raise requests.ConnectionError()
    monkeypatch.setattr("voc.pipeline.requests.post", fail)
    result = answer_question(retriever, "entrega", mode="llm")
    assert result["insufficient"] and result["error"] == "ConnectionError"
    assert len(result["evidence"]) == 2


def test_llm_contract(retriever, monkeypatch):
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"choices": [{"message": {"content": json.dumps({"answer": "Relatos de atraso [E1] [E2].", "citations": ["E1", "E2"], "insufficient": False})}}]}
    monkeypatch.setattr("voc.pipeline.requests.post", lambda *a, **k: Response())
    result = answer_question(retriever, "entrega", mode="llm")
    assert not result["insufficient"] and result["citations"] == ["E1", "E2"]


def test_clean_and_real_identity(tmp_path):
    assert clean("<p>Ótimo</p>   produto") == "Ótimo produto"
    assert "[EMAIL]" in clean("contato a@b.com")
    fields = ["review_id", "order_id", "review_score", "review_comment_title", "review_comment_message"]
    with (tmp_path / "olist_order_reviews_dataset.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([
            dict(zip(fields, ["r", "o1", "1", "", "entrega atrasada"])),
            dict(zip(fields, ["r", "o2", "1", "", "entrega atrasada"])),
            dict(zip(fields, ["r", "o1", "1", "", "entrega atrasada"])),
            dict(zip(fields, ["empty", "o3", "5", "", ""])),
            dict(zip(fields, ["bad", "o4", "9", "", "produto"]))])
    docs, stats = prepare(tmp_path)
    assert len(docs) == 2 and docs[0]["id"] != docs[1]["id"]
    assert stats["duplicates_removed"] == 1
    assert stats["empty_text"] == 1 and stats["invalid_score"] == 1


def test_chunk_diversity(tmp_path):
    doc = {"id": "long", "review_id": "long", "order_id": "long", "text": "entrega " * 180, "score": 1,
           "purchase_date": "", "categories": [], "late_delivery": None}
    build_index([doc], tmp_path, FakeEncoder(), model="fake")
    result = Retriever(tmp_path, FakeEncoder()).search("entrega")
    assert len(result) == 1


def test_context_obeys_budget(retriever):
    evidence = retriever.search("entrega")
    retained, context = build_context(evidence, max_chars=330)
    assert len(context) <= 330
    assert len(json.loads(context)) == len(retained)


@pytest.mark.parametrize("response,expected_failure", [("Há relatos de atraso [E1] [E2].", False), ("Há atrasos [E9].", True), ("Há atrasos sem citações.", True)])
def test_local_generation_references(retriever, monkeypatch, response, expected_failure):
    monkeypatch.setenv("LLM_BASE_URL", "local")
    monkeypatch.setattr("voc.pipeline.generate_local", lambda *a, **k: response)
    result = answer_question(retriever, "entrega", mode="llm")
    assert result["insufficient"] is expected_failure
