"""Contratos de ranking e integridade sem baixar/carregar modelos neurais."""
import json
import numpy as np
import pytest
from voc.retrieval import BM25, Retriever, build_index, lexical_tokens, refresh_metadata, rrf, upgrade_index


class FixedEncoder:
    def encode(self, texts):
        output = []
        for text in texts:
            value = np.asarray([1.0, float("fiscal" in text.lower())], dtype=np.float32)
            output.append(value / np.linalg.norm(value))
        return np.asarray(output)


def documents():
    return [{"id": str(i), "review_id": str(i), "order_id": str(i), "text": text, "score": 1,
             "purchase_date": "2018-01-01", "categories": ["moveis"], "late_delivery": True,
             "state": "SP", "delivery_status": "late"}
            for i, text in enumerate(["Produto sem nota fiscal", "Entrega lenta", "ENTREGA lenta!", "Produto quebrado"])]


@pytest.fixture
def indexed(tmp_path):
    build_index(documents(), tmp_path, FixedEncoder(), model="fake")
    return tmp_path


def test_bm25_rare_term_and_portuguese_negation():
    bm25 = BM25(["Produto veio sem nota fiscal", "Produto bom", "Entrega rápida"])
    assert bm25.scores("nota fiscal")[0] > bm25.scores("nota fiscal")[1]
    assert set("nao nunca sem".split()) <= set(lexical_tokens("não nunca sem"))
    assert lexical_tokens("entregas atrasadas") == lexical_tokens("entrega atrasada")


def test_rrf_rewards_consensus():
    ranking, scores = rrf([[0, 1], [2, 1]])
    assert ranking[0] == 1 and scores[1] > scores[0]


def test_hybrid_deduplicates_text_and_returns_independent_scores(indexed):
    result = Retriever(indexed, FixedEncoder()).search("nota fiscal", min_similarity=0, use_mmr=False)
    assert result[0]["id"] == "0"
    assert result[0]["similarity"] == pytest.approx(1, abs=1e-5)
    assert result[0]["bm25_score"] > 0 and result[0]["rrf_score"] > 0
    assert len(result) == 3
    assert [d["citation"] for d in result] == ["E1", "E2", "E3"]


def test_filters_apply_before_ranking(indexed):
    r = Retriever(indexed, FixedEncoder())
    assert len(r.search("produto", uf="SP", min_similarity=0)) == 3
    assert r.search("produto", uf="RJ") == []
    assert r.search("produto", score_min=4) == []


def test_reranker_is_optional_injected_and_can_change_order(indexed):
    class FakeReranker:
        name = "fake-no-download"
        def scores(self, question, texts):
            return [10 if "quebrado" in t else 0 for t in texts]
    result = Retriever(indexed, FixedEncoder()).search("produto", min_similarity=0, use_mmr=False,
                                                     reranker=FakeReranker())
    assert result[0]["id"] == "3" and result[0]["reranker_score"] == 10


@pytest.mark.parametrize("filename", ["documents.json", "vectors.npy", "corpus.json"])
def test_manifest_rejects_tampered_files(indexed, filename):
    path = indexed / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Integridade"):
        Retriever(indexed, FixedEncoder())


def test_refresh_metadata_preserves_vectors_and_rejects_changed_text(indexed):
    before = (indexed / "vectors.npy").read_bytes()
    docs = documents()
    docs[0]["state"] = "RJ"
    manifest = refresh_metadata(indexed, docs)
    assert manifest["schema"] == 2 and before == (indexed / "vectors.npy").read_bytes()
    assert Retriever(indexed, FixedEncoder()).docs[0]["state"] == "RJ"
    docs[0]["text"] = "Outro texto que precisa de outros vetores"
    with pytest.raises(ValueError, match="Texto/IDs/ordem"):
        refresh_metadata(indexed, docs)


def test_upgrade_legacy_registers_hashes_without_reembedding(indexed):
    path = indexed / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.pop("integrity")
    manifest["schema"] = 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert upgrade_index(indexed)["schema"] == 2
    assert len(Retriever(indexed, FixedEncoder()).docs) == 4


def test_invalid_search_settings(indexed):
    r = Retriever(indexed, FixedEncoder())
    for kwargs in [{"algorithm": "magic"}, {"mmr_lambda": 1.1}, {"k": 21}]:
        with pytest.raises(ValueError):
            r.search("produto", **kwargs)
