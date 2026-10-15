import pytest

from voc.query import matches_filters, resolve_query


def test_natural_filters_combine_score_category_year_state_and_delivery():
    plan = resolve_query("Quais principais problemas em avaliações negativas de móveis em 2018, de SP, com pedidos entregues com atraso?")
    assert plan["route"] == "analytics"
    assert plan["filters"]["score_max"] == 2
    assert plan["filters"]["date_from"] == "2018-01-01"
    assert plan["filters"]["date_to"] == "2018-12-31"
    assert plan["filters"]["state"] == "SP"
    assert plan["filters"]["delivery_status"] == "late"
    assert "moveis_decoracao" in plan["filters"]["category"]
    assert plan["notes"]


def test_interface_defaults_do_not_erase_natural_language_constraints():
    plan = resolve_query("Padrões de clientes insatisfeitos de SP em 2018", {"score_min": 1, "score_max": 5, "late_only": False})
    assert plan["filters"]["score_max"] == 2
    assert plan["filters"]["state"] == "SP"


@pytest.mark.parametrize("manual", [
    {"score_min": 4}, {"state": "RJ"}, {"date_from": "2019-01-01"},
    {"category": "automotivo"},
])
def test_conflicting_manual_filters_fail_closed(manual):
    with pytest.raises(ValueError, match="conflit"):
        resolve_query("Principais problemas de nota baixa em beleza e saúde, de SP, em 2018", manual)


def test_same_order_must_satisfy_all_order_filters():
    document = {"score": 1, "order_metadata": [
        {"order_id": "one", "state": "SP", "purchase_date": "2017-01-01", "categories": ["beleza_saude"], "delivery_status": "late"},
        {"order_id": "two", "state": "RJ", "purchase_date": "2018-01-01", "categories": ["automotivo"], "delivery_status": "on_time"},
    ]}
    assert matches_filters(document, state="SP", category="beleza_saude", late_only=True)
    assert not matches_filters(document, state="SP", category="automotivo")
    assert not matches_filters(document, state="SP", date_from="2018-01-01")
    assert not matches_filters(document, state="RJ", late_only=True)


def test_unknown_delivery_is_neither_late_nor_on_time():
    document = {"score": 5, "categories": [], "late_delivery": None, "delivery_status": "unknown"}
    assert matches_filters(document, delivery_status="unknown", algorithm="hybrid", k=6)
    assert not matches_filters(document, delivery_status="on_time")
    assert not matches_filters(document, late_only=True)


def test_leap_month_dates_and_purchase_date_semantics():
    plan = resolve_query("Quantas reclamações em fevereiro de 2020?")
    assert plan["filters"]["date_from"] == "2020-02-01"
    assert plan["filters"]["date_to"] == "2020-02-29"
    assert resolve_query("Principais problemas antes de 2018")["filters"]["date_to"] == "2017-12-31"


def test_no_state_inferred_from_preposition_para():
    plan = resolve_query("O que relatam para entender os problemas de embalagem?")
    assert "state" not in plan["filters"]
    assert plan["route"] == "rag"


def test_specific_question_stays_rag_aggregate_goes_analytics():
    assert resolve_query("O que relatam sobre embalagens danificadas?")["route"] == "rag"
    assert resolve_query("O que os clientes mais elogiam em beleza e saúde?")["route"] == "analytics"
    assert resolve_query("Qual é a frequência de atrasos?")["route"] == "analytics"


def test_rating_conflicts_and_unrecognized_filter_are_explicit():
    document = {"scores": [1, 5], "score": None, "categories": []}
    assert matches_filters(document)
    assert not matches_filters(document, score_max=2)
    with pytest.raises(ValueError, match="desconhecido"):
        resolve_query("Entrega", {"socre_max": 2})
    with pytest.raises(ValueError, match="UF"):
        resolve_query("Entrega", {"state": "ZZ"})


@pytest.mark.parametrize("period,start,end", [
    ("entre janeiro e março de 2018", "2018-01-01", "2018-03-31"),
    ("de janeiro a março de 2018", "2018-01-01", "2018-03-31"),
    ("entre novembro de 2017 e fevereiro de 2018", "2017-11-01", "2018-02-28"),
    ("janeiro até fevereiro de 2020", "2020-01-01", "2020-02-29"),
])
def test_explicit_month_interval_covers_both_calendar_limits(period, start, end):
    filters = resolve_query("Quais reclamações " + period + "?")["filters"]
    assert filters["date_from"] == start
    assert filters["date_to"] == end


@pytest.mark.parametrize("period", [
    "janeiro e março de 2018", "entre janeiro e março", "entre março e janeiro de 2018",
    "janeiro, fevereiro e março de 2018", "janeiro de 2017 e 2018",
])
def test_ambiguous_month_lists_or_years_are_explicit_errors(period):
    with pytest.raises(ValueError, match="Período|período|ano|Mês"):
        resolve_query("Quantas reclamações em " + period + "?")
