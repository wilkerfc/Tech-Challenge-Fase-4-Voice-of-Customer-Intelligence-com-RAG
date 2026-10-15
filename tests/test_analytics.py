from copy import deepcopy

import pytest

from voc.analytics import analyze, label_themes, validate_analytics


def document(index, text="Entrega atrasada", score=1, **extra):
    return {"id": str(index), "review_id": str(index), "order_id": f"o{index}",
            "text": text, "score": score, "purchase_date": "2018-01-01",
            "categories": ["beleza_saude"], "state": "SP", "late_delivery": True,
            "delivery_status": "late", **extra}


def test_counts_use_whole_corpus_deduplicate_review_ids_and_keep_text_duplicates():
    docs = [document(i) for i in range(40)] + [document("other", "Produto excelente", score=5)]
    docs += [dict(docs[0], id="another-order", order_id="new-order")]
    result = analyze(docs, "Qual a frequência de atrasos?", min_base=30)
    assert not result["insufficient"]
    assert result["analytics"]["denominator"] == 41
    row = result["analytics"]["rows"][0]
    assert row["count"] == 40 and row["percentage"] == 97.56
    assert "40 avaliações (97,56%" in result["answer"]
    assert result["analytics"]["multiple_order_reviews"] == 1


def test_score_never_substitutes_for_textual_cause():
    docs = [document(i, "Sem comentário específico", score=1) for i in range(30)]
    docs += [document("positive-note-complaint", "Produto quebrado", score=5)]
    result = analyze(docs, "Principais problemas")
    row = next(row for row in result["analytics"]["rows"] if row["theme"] == "danificado")
    assert row["count"] == 1 and result["analytics"]["denominator"] == 31
    assert "qualidade_ruim" not in {row["theme"] for row in result["analytics"]["rows"]}


def test_small_and_empty_groups_do_not_publish_percentages():
    result = analyze([document(1)], "Frequência de atrasos")
    assert result["insufficient"] and result["analytics"]["rows"][0]["percentage"] is None
    assert result["analytics"]["coverage_percentage"] is None
    assert "%" not in result["answer"]
    empty = analyze([document(1)], "Frequência de atrasos", {"state": "RJ"})
    assert empty["analytics"]["denominator"] == 0 and empty["insufficient"]


def test_overlap_and_coverage_are_declared_without_summing_percentages():
    result = analyze([document(i, "Produto quebrado e com defeito") for i in range(30)], "Principais problemas")
    assert result["analytics"]["count"] == 30
    assert sum(r["count"] for r in result["analytics"]["rows"]) == 60
    assert result["analytics"]["coverage_percentage"] == 100
    assert "percentuais não devem ser somados" in result["answer"]


@pytest.mark.parametrize("tamper", ["answer", "count", "percentage", "label", "example"])
def test_numeric_and_example_tampering_is_rejected(tamper):
    result = analyze([document(i) for i in range(30)], "Frequência de atrasos")
    changed = deepcopy(result)
    row = changed["analytics"]["rows"][0]
    if tamper == "answer":
        changed["answer"] = changed["answer"].replace("30 avaliações", "300 avaliações")
    elif tamper == "count":
        row["count"] = 300
    elif tamper == "percentage":
        row["percentage"] = 50
    elif tamper == "label":
        row["label"] = "Boa qualidade"
    else:
        changed["evidence"][0]["texts"] = ["Sem reclamações de qualquer tipo"]
    with pytest.raises(ValueError):
        validate_analytics(changed)


def test_negation_preserved_for_praise_defects_and_delays():
    assert "entrega_rapida" not in label_themes("Não entregou no prazo")
    assert "atraso" not in label_themes("Não houve atraso e sem demora")
    assert "defeito" not in label_themes("Produto sem defeito")
    assert "defeito" in label_themes("Produto não funciona")
    assert "recomenda" not in label_themes("Não recomendo")
    assert "nao_recebido" not in label_themes("Produto não veio na cor escolhida")


def test_multiorder_cannot_mix_state_and_category_across_orders():
    doc = document(1, order_metadata=[
        {"order_id": "one", "state": "SP", "categories": ["beleza_saude"], "purchase_date": "2017-01-01"},
        {"order_id": "two", "state": "RJ", "categories": ["automotivo"], "purchase_date": "2018-01-01"},
    ])
    result = analyze([doc], "Principais problemas", {"state": "SP", "category": "automotivo"}, min_base=1)
    assert result["analytics"]["denominator"] == 0


def test_chunk_population_is_rejected_and_unmatched_rules_abstain():
    with pytest.raises(ValueError, match="corpus.json"):
        analyze([document(1, chunk=0)], "Principais problemas")
    no_match = analyze([document(i, "Uma observação sem tema") for i in range(30)], "Frequência de atrasos")
    assert no_match["insufficient"] and no_match["status"] == "no_evidence"
    assert "não prova ausência" in no_match["answer"]


@pytest.mark.parametrize("question,text,theme", [
    ("Quais são as reclamações mais frequentes sobre atendimento?", "Péssimo atendimento e sem resposta", "atendimento_ruim"),
    ("Quais as principais reclamações sobre os produtos?", "Produto quebrado", "danificado"),
    ("Quais os principais temas sobre documentação fiscal?", "Não enviaram a nota fiscal", "nota_fiscal"),
    ("Quais as principais reclamações de pós-venda?", "Estou aguardando reembolso", "troca_reembolso"),
    ("Quais os principais elogios sobre preço e custo-benefício?", "Bom preço", "preco_bom"),
])
def test_explicit_subject_restricts_aggregate_rows(question, text, theme):
    docs = [document(i, "Entrega atrasada e não recebi o pedido") for i in range(30)]
    docs += [document("matching", text)]
    result = analyze(docs, question, min_base=1)
    assert not result["insufficient"]
    assert {row["theme"] for row in result["analytics"]["rows"]} == {theme}
    assert result["analytics"]["rows"][0]["count"] == 1
    assert result["analytics"]["denominator"] == 31


def test_two_explicit_subjects_form_union_without_unrelated_product_topics():
    docs = [document(1, "Entrega atrasada"), document(2, "Péssimo atendimento"), document(3, "Produto quebrado")]
    result = analyze(docs, "Principais reclamações sobre entrega e atendimento", min_base=1)
    assert {row["theme"] for row in result["analytics"]["rows"]} == {"atraso", "atendimento_ruim"}


@pytest.mark.parametrize("question", [
    "Quantas avaliações mencionam cashback?",
    "Quais as reclamações mais frequentes sobre fidelidade?",
    "Quantas avaliações citam gorjetas?",
    "Qual a frequência de entregas por drones?",
])
def test_unknown_analytic_focus_never_substitutes_general_ranking(question):
    result = analyze([document(i) for i in range(30)], question)
    assert result["insufficient"] and result["status"] == "no_evidence"
    assert result["reason"] == "unsupported_analytic_subject"
    assert result["analytics"]["rows"] == [] and result["analytics"]["count"] == 0
    assert result["evidence"] == [] and result["citations"] == []
    assert result["analytics"]["coverage_percentage"] is None
    assert "modo RAG" in result["answer"]


def test_generic_question_and_category_filter_still_allow_general_ranking():
    docs = [document(i, "Entrega rápida no prazo, produto excelente") for i in range(30)]
    result = analyze(docs, "Quais aspectos da experiência de compra são mais elogiados pelos clientes em beleza e saúde?",
                     {"category": "beleza_saude"})
    assert not result["insufficient"]
    assert {row["theme"] for row in result["analytics"]["rows"]} == {"entrega_rapida", "qualidade_boa"}


def test_explicit_category_value_is_a_filter_even_without_a_language_alias():
    docs = [document(1, categories=["seguros_e_servicos"])]
    result = analyze(docs, "Quais os temas frequentes em seguros_e_servicos?", {"category": "seguros_e_servicos"})
    assert result["status"] == "insufficient_evidence"
    assert "reason" not in result
    assert result["analytics"]["denominator"] == 1
    assert result["analytics"]["rows"][0]["theme"] == "atraso"
    assert result["analytics"]["rows"][0]["percentage"] is None
