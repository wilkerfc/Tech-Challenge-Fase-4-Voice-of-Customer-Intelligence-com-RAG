"""Transparent Portuguese intent and metadata filters, with conjunctive order joins.

Dates are inclusive ISO calendar dates and refer to purchase, never review date.
Natural-language and manual constraints are intersected; conflicting constraints
raise ValueError rather than silently broadening the requested population.
"""
import calendar
import re
import unicodedata
from datetime import date, timedelta


FILTER_KEYS = {"score_min", "score_max", "category", "date_from", "date_to",
               "late_only", "state", "uf", "delivery_status"}
SEARCH_KEYS = {"k", "min_similarity", "algorithm", "use_mmr", "mmr_lambda",
               "candidate_pool", "reranker", "min_lexical", "top_k", "rrf_k"}
STATES = {
    "acre": "AC", "alagoas": "AL", "amapa": "AP", "amazonas": "AM", "bahia": "BA",
    "ceara": "CE", "distrito federal": "DF", "espirito santo": "ES", "goias": "GO",
    "maranhao": "MA", "mato grosso do sul": "MS", "mato grosso": "MT",
    "minas gerais": "MG", "para": "PA", "paraiba": "PB", "parana": "PR",
    "pernambuco": "PE", "piaui": "PI", "rio de janeiro": "RJ", "rio grande do norte": "RN",
    "rio grande do sul": "RS", "rondonia": "RO", "roraima": "RR", "santa catarina": "SC",
    "sao paulo": "SP", "sergipe": "SE", "tocantins": "TO",
}
FURNITURE = ["moveis_colchao_e_estofado", "moveis_cozinha_area_de_servico_jantar_e_jardim",
             "moveis_decoracao", "moveis_escritorio", "moveis_quarto", "moveis_sala"]
CATEGORY_ALIASES = {
    "beleza e saude": ["beleza_saude"], "beleza saude": ["beleza_saude"],
    "moveis": FURNITURE, "cama mesa e banho": ["cama_mesa_banho"],
    "cama mesa banho": ["cama_mesa_banho"], "roupa de cama": ["cama_mesa_banho"],
    "informatica": ["informatica_acessorios"], "computadores": ["computadores"],
    "eletronicos": ["eletronicos"], "eletrodomesticos": ["eletrodomesticos", "eletrodomesticos_2"],
    "telefonia": ["telefonia", "telefonia_fixa"], "automotivo": ["automotivo"],
    "brinquedos": ["brinquedos"], "esporte e lazer": ["esporte_lazer"],
    "esporte lazer": ["esporte_lazer"], "relogios": ["relogios_presentes"],
    "livros": ["livros_tecnicos", "livros_interesse_geral", "livros_importados"],
    "pet shop": ["pet_shop"], "ferramentas": ["ferramentas_jardim"],
    "utilidades domesticas": ["utilidades_domesticas"], "perfumaria": ["perfumaria"],
    "papelaria": ["papelaria"], "bebes": ["bebes"],
}
MONTHS = {"janeiro": 1, "fevereiro": 2, "marco": 3, "abril": 4, "maio": 5,
          "junho": 6, "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10,
          "novembro": 11, "dezembro": 12}


def plain(text):
    return "".join(c for c in unicodedata.normalize("NFD", str(text).casefold())
                   if not unicodedata.combining(c))


def _many(value):
    if value is None or value == "":
        return []
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


def _categories(value):
    result = set()
    for item in _many(value):
        normalized = plain(item).replace("_", " ").strip()
        result.update(CATEGORY_ALIASES.get(normalized, [str(item)]))
        result.add(str(item))  # keep literal legacy categories as well as Olist aliases
    return result


def _date(value):
    value = value.isoformat() if hasattr(value, "isoformat") else str(value)
    try:
        date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Use uma data ISO válida (AAAA-MM-DD).") from error
    return value


def matched_order_metadata(doc, category=None, date_from=None, date_to=None,
                           late_only=False, state=None, uf=None, delivery_status=None, **_):
    """Every order constraint must hold for one and the same linked order.

    Never create a synthetic order with a category from SP and a date from RJ.
    Old index documents without order_metadata remain supported.
    """
    states = {str(x).upper() for x in _many(state or uf)}
    if state and uf:
        states = states.intersection(str(x).upper() for x in _many(uf))
        if not states:
            return []
    cats = _categories(category)
    statuses = set(_many(delivery_status))
    metadata = doc.get("order_metadata")
    if metadata is None:
        metadata = [{k: doc.get(k) for k in ("order_id", "categories", "purchase_date",
                                             "state", "uf", "delivery_status", "late_delivery")}]
    result = []
    for order in metadata:
        status = order.get("delivery_status")
        if not status:
            late = order.get("late_delivery")
            status = "late" if late is True else "on_time" if late is False else "unknown"
        purchase = order.get("purchase_date") or ""
        if cats and not cats.intersection(order.get("categories") or []):
            continue
        if states and (order.get("state") or order.get("uf") or "").upper() not in states:
            continue
        if date_from and (not purchase or purchase < str(date_from)):
            continue
        if date_to and (not purchase or purchase > str(date_to)):
            continue
        if late_only and status != "late":
            continue
        if statuses and status not in statuses:
            continue
        result.append(order)
    return result


def matches_filters(doc, score_min=1, score_max=5, category=None, date_from=None,
                    date_to=None, late_only=False, state=None, uf=None,
                    delivery_status=None, **extras):
    """Corpus/retrieval share a filter predicate; search controls are irrelevant.

    When conflicting copies of a review have several scores, all must satisfy a
    requested score range. This conservative choice avoids choosing one score.
    """
    scores = doc.get("scores") or [doc.get("score")]
    if not all(isinstance(s, (int, float)) and score_min <= s <= score_max for s in scores):
        return False
    return bool(matched_order_metadata(doc, category=category, date_from=date_from, date_to=date_to,
                                      late_only=late_only, state=state, uf=uf, delivery_status=delivery_status))


# Public spelling shared with retrieval; the older spelling remains explicit.
matching_order_metadata = matched_order_metadata


def _merge(inferred, manual, notes):
    result = dict(inferred)
    manual = {k: v for k, v in (manual or {}).items() if v is not None and v != ""}
    unknown = set(manual) - FILTER_KEYS - SEARCH_KEYS
    if unknown:
        raise ValueError("Filtro desconhecido: " + ", ".join(sorted(unknown)))
    # Search controls retain their values but never affect the analytics corpus.
    result.update({k: v for k, v in manual.items() if k in SEARCH_KEYS})
    if "uf" in manual:
        if "state" in manual and set(_many(manual["state"])) != set(_many(manual["uf"])):
            raise ValueError("Filtros manuais de UF conflitantes.")
        manual["state"] = manual.pop("uf")
    lower = max(int(inferred.get("score_min", 1)), int(manual.get("score_min", 1)))
    upper = min(int(inferred.get("score_max", 5)), int(manual.get("score_max", 5)))
    if not 1 <= lower <= upper <= 5:
        raise ValueError("O filtro de nota conflita com a pergunta ou tem intervalo inválido.")
    if lower != 1 or "score_min" in inferred or "score_min" in manual:
        result["score_min"] = lower
    if upper != 5 or "score_max" in inferred or "score_max" in manual:
        result["score_max"] = upper
    for key in ("category", "state", "delivery_status"):
        normalize = _categories if key == "category" else lambda x: set(_many(x))
        left = normalize(inferred.get(key))
        right = normalize(manual.get(key))
        if key == "state":
            right = {str(x).upper() for x in right}
            if right - set(STATES.values()):
                raise ValueError("UF inválida: use a sigla de um estado brasileiro.")
        if key == "delivery_status" and right - {"late", "on_time", "not_delivered", "unknown"}:
            raise ValueError("Situação da entrega desconhecida.")
        both = left.intersection(right) if left and right else left or right
        if left and right and not both:
            raise ValueError(f"O filtro manual de {key} conflita com a pergunta.")
        if both:
            result[key] = next(iter(both)) if len(both) == 1 else sorted(both)
    for key in ("date_from", "date_to"):
        left, right = inferred.get(key), manual.get(key)
        if right:
            right = _date(right)
        if left or right:
            result[key] = (max if key == "date_from" else min)(x for x in (left, right) if x)
    if result.get("date_from") and result.get("date_to") and result["date_from"] > result["date_to"]:
        raise ValueError("O período manual conflita com o período solicitado.")
    if inferred.get("late_only") or manual.get("late_only"):
        result["late_only"] = True
        statuses = set(_many(result.get("delivery_status")))
        if statuses and "late" not in statuses:
            raise ValueError("Filtros de situação da entrega conflitantes.")
        if statuses:
            result["delivery_status"] = "late"
    if inferred and manual:
        notes.append("Filtros da pergunta e da interface combinados por interseção; um controle padrão não amplia a pergunta.")
    return result


def resolve_query(question, manual_filters=None):
    query = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", question or "")).strip()
    if not query or len(query) > 2000:
        raise ValueError("A pergunta deve conter entre 1 e 2000 caracteres.")
    text, inferred, notes = plain(query), {}, []
    comparison = re.search(r"\bnotas?\s*(>=|<=|>|<|maiores? que|menores? que|acima de|abaixo de|a partir de|ate)\s*([1-5])\b", text)
    score = re.search(r"\bnota(?:s)?\s*(?:(?:de|entre|igual a)\s*)?([1-5])(?:\s*(?:a|ate|e|ou|[-–])\s*([1-5]))?\b", text)
    if comparison:
        operator, value = comparison[1], int(comparison[2])
        if operator in {">", "maior que", "maiores que", "acima de"}:
            inferred["score_min"] = value + 1
        elif operator in {"<", "menor que", "menores que", "abaixo de"}:
            inferred["score_max"] = value - 1
        elif operator in {">=", "a partir de"}:
            inferred["score_min"] = value
        else:
            inferred["score_max"] = value
    elif score:
        a, b = int(score[1]), int(score[2] or score[1])
        inferred.update(score_min=min(a, b), score_max=max(a, b))
    elif re.search(r"\b(insatisfeit\w*|insatisfatori\w*|negativ\w*|nota[s]? baixa[s]?)\b", text):
        inferred["score_max"] = 2
        notes.append("'Negativas/insatisfeitos/nota baixa' significa notas 1–2; a nota não determina a causa do comentário.")
    elif re.search(r"\b(nota[s]? alta[s]?|nota[s]? positiva[s]?)\b", text):
        inferred["score_min"] = 4
    month_names = "|".join(MONTHS)
    month_mentions = list(re.finditer(rf"\b(?:{month_names})\b", text))
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", text)]
    iso_dates = re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", text)
    month_range = None
    if len(month_mentions) > 1:
        # A list such as 'janeiro e março' may mean two separate months. Only
        # explicit interval syntax permits filling the intervening calendar.
        interval = re.search(
            rf"\b(?P<first>{month_names})(?:\s+(?:de\s+)?(?P<year_first>20\d{{2}}))?"
            rf"\s+(?P<link>e|a|ate)\s+(?P<last>{month_names})"
            rf"(?:\s+(?:de\s+)?(?P<year_last>20\d{{2}}))?\b", text)
        if (len(month_mentions) != 2 or not interval or iso_dates or
                (interval["link"] == "e" and not re.search(r"\bentre\s*$", text[:interval.start()]))):
            raise ValueError("Período com vários meses ambíguo. Use 'entre janeiro e março de 2018' ou duas datas ISO.")
        first_year = interval["year_first"] or interval["year_last"]
        last_year = interval["year_last"] or interval["year_first"]
        interval_years = [int(y) for y in (interval["year_first"], interval["year_last"]) if y]
        if not first_year or years != interval_years:
            raise ValueError("Informe o ano de cada limite do período, ou um único ano compartilhado.")
        y1, y2 = int(first_year), int(last_year)
        m1, m2 = MONTHS[interval["first"]], MONTHS[interval["last"]]
        if (y1, m1) > (y2, m2):
            raise ValueError("O início do período deve anteceder o fim. Informe anos explícitos para atravessar a virada do ano.")
        month_range = (f"{y1}-{m1:02}-01", f"{y2}-{m2:02}-{calendar.monthrange(y2, m2)[1]}")
    elif month_mentions and (not years or len(years) > 1 or iso_dates):
        raise ValueError("Mês com ano ausente ou ambíguo. Informe um mês e ano ou duas datas ISO.")
    if iso_dates:
        inferred["date_from"] = _date(iso_dates[0])
        inferred["date_to"] = _date(iso_dates[-1])
        if len(iso_dates) == 1 and re.search(r"\bdesde\b|\ba partir\b", text):
            inferred.pop("date_to")
        elif len(iso_dates) == 1 and re.search(r"\bate\b|\bantes de\b", text):
            inferred.pop("date_from")
            if "antes de" in text:
                inferred["date_to"] = (date.fromisoformat(inferred["date_to"]) - timedelta(days=1)).isoformat()
    elif month_range:
        inferred.update(date_from=month_range[0], date_to=month_range[1])
    elif month_mentions and years:
        y, m = years[0], MONTHS[month_mentions[0][0]]
        inferred.update(date_from=f"{y}-{m:02}-01", date_to=f"{y}-{m:02}-{calendar.monthrange(y, m)[1]}")
        if re.search(r"\bdesde\b|\ba partir\b", text):
            inferred.pop("date_to")
        elif re.search(r"\bate\b|\bantes de\b", text):
            inferred.pop("date_from")
            if "antes de" in text:
                inferred["date_to"] = (date(y, m, 1) - timedelta(days=1)).isoformat()
    elif years:
        inferred["date_from"] = f"{min(years)}-01-01"
        inferred["date_to"] = f"{max(years)}-12-31"
        if len(years) == 1 and re.search(r"\bdesde\b|\ba partir\b", text):
            inferred.pop("date_to")
        elif len(years) == 1 and re.search(r"\bate\b|\bantes de\b", text):
            inferred.pop("date_from")
            if "antes de" in text:
                inferred["date_to"] = f"{years[0] - 1}-12-31"
    if years or iso_dates:
        notes.append("O período filtra a data da compra; as duas datas-limite são inclusivas.")
    elif re.search(r"ultim[oa]s?\s+\w+\s+(meses|dias|anos)|este ano|ano passado", text):
        notes.append("Período relativo não inferido: informe datas de compra explícitas.")
    for alias in sorted(CATEGORY_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", text):
            inferred["category"] = CATEGORY_ALIASES[alias]
            break
    for name in sorted(STATES, key=len, reverse=True):
        # 'para' and 'amazonas' can occur in ordinary prose: require a place cue.
        if name == "para":
            found = re.search(r"\b(?:estado (?:do |de )?|no |do )para\b", text)
        else:
            found = re.search(rf"\b{re.escape(name)}\b", text)
        if found:
            inferred["state"] = STATES[name]
            break
    state_code = re.search(r"\b(?:em|de|do|no|uf\s*[:=]?)\s+(" + "|".join(sorted(STATES.values())).lower() + r")\b", text)
    if state_code:
        inferred["state"] = state_code[1].upper()
    if re.search(r"\b(entreg\w*|pedidos?)\s+(?:foram |que foram |com |registrad\w* como )?(atrasad\w*|atraso)\b", text):
        inferred["delivery_status"] = "late"
    elif re.search(r"\bentreg\w*\s+(?:registrad\w* )?(no prazo|dentro do prazo)\b", text):
        inferred["delivery_status"] = "on_time"
    if "delivery_status" in inferred:
        notes.append("Situação da entrega vem dos registros do pedido; menções no texto são contadas separadamente.")
    analytics = bool(re.search(r"\b(quantos?|quantas?|frequencia|frequentes?|recorrentes?|percentual|porcentagem|proporcao|maioria|distribuicao|ranking|principais|padroes)\b", text)
                     or re.search(r"\bmais\s+(?:\w+\s+){0,2}(?:elogia\w*|reclam\w*|relat\w*|cita\w*|frequent\w*|comun\w*)\b", text))
    filters = _merge(inferred, manual_filters, notes)
    return {"route": "analytics" if analytics else "rag", "filters": filters, "notes": notes, "query": query}
