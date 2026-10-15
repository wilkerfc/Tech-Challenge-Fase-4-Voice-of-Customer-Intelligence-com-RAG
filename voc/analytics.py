"""Deterministic counts over unique textual reviews, never over retrieved top-k.

Rules describe lexical mentions, not causes, ground truth product defects, or
the prevalence among all customers. A review can mention several themes.
Every numeric sentence is rendered from the same structured result checked by
validate_analytics; no LLM writes, changes, or estimates the numbers.
"""
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import re

from voc.query import (CATEGORY_ALIASES, FILTER_KEYS, MONTHS, STATES,
                       matched_order_metadata, matches_filters, plain)


TAXONOMY_VERSION = "lexical-2026-10-v2"


@dataclass(frozen=True)
class Theme:
    id: str
    label: str
    polarity: str
    group: str
    pattern: str
    check_negation: bool = False


TAXONOMY = (
    Theme("nao_recebido", "Pedido não recebido", "problem", "delivery",
          r"\b(?:nao|nunca|ainda nao) (?:recebi|recebemos|chegou|chegaram|foi entregue|entregaram)\b"
          r"(?! (?:a |o )?(?:nota|nf|resposta|retorno|boleto|email|e-mail|manual))"
          r"|\b(?:aguardando|esperando|aguardo) (?:o |a |meu |minha )?(?:produto|pedido|encomenda|entrega)"),
    Theme("atraso", "Atraso ou demora na entrega", "problem", "delivery",
          r"\batras\w*|\bdemor\w*|\bfora do prazo|\bdepois do prazo|\bnao cumpr\w* (?:o )?prazo", True),
    Theme("incompleto", "Pedido incompleto ou faltando itens", "problem", "delivery",
          r"\bincomplet\w*|\b(?:recebi|veio|chegou|entregaram) (?:so|apenas|somente) (?:uma? |\d|parte)"
          r"|\bfalt(?:ou|ando|aram) (?:uma? |o |a |os |as )?(?:item|produto|peca|parte|unidade)"),
    Theme("produto_errado", "Produto diferente ou errado", "problem", "product",
          r"\b(?:produto|item|cor|modelo|tamanho) (?:errad\w*|diferente|trocad\w*)"
          r"|\b(?:recebi|veio|chegou|enviaram) (?:um |uma |o |a )?(?:outro|outra|errado|errada)\b"
          r"|\bdiferente d[oa] (?:anunciad\w*|pedid\w*|comprad\w*|foto)"
          r"|\bnao (?:corresponde|era|e) (?:ao |o |a )?(?:anunciad\w*|que (?:eu )?(?:comprei|pedi))"
          r"|\bnao veio na cor"),
    Theme("danificado", "Produto danificado ou avariado", "problem", "product",
          r"\b(?:quebrad\w*|danificad\w*|avariad\w*|amassad\w*|trincad\w*|rachad\w*|estragad\w*)", True),
    Theme("defeito", "Defeito ou funcionamento inadequado", "problem", "product",
          r"\bdefeit\w*|\bnao (?:funciona\w*|funcionou|liga|ligou|carrega)\b|\bparou de funcionar", True),
    Theme("qualidade_ruim", "Qualidade abaixo do esperado", "problem", "product",
          r"\b(?:baixa|pessima|ma|ruim|horrivel) qualidade|\bqualidade (?:ruim|baixa|inferior|pessima)"
          r"|\bmaterial (?:fragil|fraco|ruim|fino)", True),
    Theme("atendimento_ruim", "Atendimento ou falta de resposta", "problem", "service",
          r"\b(?:pessimo|mau|ruim) atendimento|\bnao (?:respond\w*|atend\w*|retorn\w*)"
          r"|\bsem (?:resposta|retorno)|\b(?:contato|atendimento) (?:impossivel|ruim)"),
    Theme("troca_reembolso", "Troca, devolução ou reembolso", "problem", "after_sales",
          r"\btroc(?:a|ar|ando)\b|\bdevol\w*|\breembols\w*|\bestorn\w*|\bcancel\w*"),
    Theme("sem_estoque", "Indisponibilidade de estoque", "problem", "product",
          r"\bsem estoque|\bnao (?:tinha|tem|havia) (?:o )?(?:produto|estoque)|\bproduto indisponivel"),
    Theme("nota_fiscal", "Menção a nota fiscal", "neutral", "documentation",
          r"\bnota fiscal|\bnfe?\b|\bdanfe\b|\bcupom fiscal"),
    Theme("entrega_rapida", "Entrega rápida ou no prazo", "praise", "delivery",
          r"\bentrega (?:muito )?(?:rapida|eficiente)|\b(?:chegou|entregue|recebi) (?:bem |muito )?rapid\w*"
          r"|\bantes do prazo|\b(?:no|dentro do) prazo", True),
    Theme("qualidade_boa", "Boa qualidade do produto", "praise", "product",
          r"\b(?:boa|otima|excelente) qualidade|\bqualidade (?:boa|otima|excelente)"
          r"|\bproduto (?:muito )?(?:bom|otimo|excelente)\b", True),
    Theme("conforme_anunciado", "Conforme o anunciado", "praise", "product",
          r"\b(?:conforme|igual (?:ao|o)|como) (?:o )?(?:anunciad\w*|anuncio|descrit\w*|esperad\w*)", True),
    Theme("atendimento_bom", "Bom atendimento", "praise", "service",
          r"\b(?:bom|otimo|excelente) atendimento|\bbem atendid\w*|\batendimento (?:bom|otimo|excelente)", True),
    Theme("embalagem_boa", "Embalagem bem feita", "praise", "product",
          r"\bbem embalad\w*|\bembalagem (?:boa|otima|excelente|perfeita)", True),
    Theme("preco_bom", "Preço ou custo-benefício elogiado", "praise", "price",
          r"\b(?:bom|otimo|excelente) (?:preco|custo beneficio)|\bpreco (?:bom|otimo|justo|baixo)", True),
    Theme("recomenda", "Recomendação ou satisfação", "praise", "general",
          r"\brecomendo\b|\brecomendada?\b|\bsatisfeit\w*", True),
)


@lru_cache(maxsize=60000)
def label_themes(text):
    """Return stable, inspectable lexical labels; negation window is explicit."""
    normalized = plain(text)
    result = []
    for theme in TAXONOMY:
        for match in re.finditer(theme.pattern, normalized):
            before = normalized[max(0, match.start() - 30):match.start()]
            negated = re.search(r"\b(?:nao|nunca|jamais|sem|nem|nada)\b[^.!?;]{0,24}$", before)
            if theme.check_negation and negated:
                continue
            result.append(theme.id)
            break
    return tuple(result)


def unique_reviews(documents):
    """Merge copies/chunks without losing text variants or linked orders.

    Analytics requires corpus.json (full review text). It can still consume raw
    prepare() rows; passing chunks is detectable only when chunk is present and
    is rejected, because chunk dedup cannot reconstruct omitted original text.
    """
    # A caller may cache this full-corpus preparation once per immutable index.
    if isinstance(documents, (list, tuple)) and documents and all(d.get("_analytics_unique_review") for d in documents):
        return documents
    groups = defaultdict(list)
    for document in documents:
        if "chunk" in document:
            raise ValueError("A camada analítica exige corpus.json com textos completos; chunks não são uma população de avaliações.")
        groups[document.get("review_id") or document["id"]].append(document)
    result = []
    for review_id, copies in sorted(groups.items()):
        texts = sorted({d["text"] for d in copies if d.get("text")})
        scores = sorted({d["score"] for d in copies if isinstance(d.get("score"), (int, float))})
        metadata = {}
        for doc in copies:
            for order in doc.get("order_metadata") or [{k: doc.get(k) for k in
                    ("order_id", "purchase_date", "categories", "state", "uf", "delivery_status", "late_delivery")}]:
                oid = order.get("order_id") or doc["id"]
                if oid in metadata and metadata[oid] != order:
                    raise ValueError("Metadados conflitantes para o mesmo pedido; regenere a base.")
                metadata[oid] = order
        if not texts or not scores:
            continue
        # 'text' is a rendering of all variants, not a chosen source; variants
        # are matched separately to avoid joining words across documents.
        result.append({"id": str(review_id), "review_id": str(review_id), "_analytics_unique_review": True,
            "order_id": ", ".join(sorted(metadata)), "order_ids": sorted(metadata),
            "order_metadata": [metadata[k] for k in sorted(metadata)],
            "text": " || ".join(texts), "texts": texts,
            "score": scores[0] if len(scores) == 1 else None, "scores": scores,
            "purchase_date": "", "categories": sorted({c for o in metadata.values() for c in o.get("categories") or []}),
            "states": sorted({o.get("state") or o.get("uf") for o in metadata.values() if o.get("state") or o.get("uf")}),
            "late_delivery": None,
            "theme_ids": sorted({t for text in texts for t in label_themes(text)})})
    return result


# Focus words and question framing are separate: an unknown noun after 'sobre',
# 'mencionam' or elsewhere cannot silently become the generic theme ranking.
# These are positive vocabularies with word boundaries, not a topic denylist.
_ANALYTIC_FRAME = set("""
a o as os um uma uns umas de da do das dos em no na nos nas para por com e ou que se
ao aos pelo pela pelos pelas sua suas seu seus sobre acerca respeito quanto quanto
qual quais quantos quantas como onde ha sao ser foi foram estao esta entre ate desde
antes depois partir apenas somente todos todas todo toda algum alguma alguns algumas
avaliacao avaliacoes comentario comentarios base corpus conjunto recorte populacao
periodo periodos mes meses ano anos dia dias data datas compra compras clientes cliente
comprador compradores consumidor consumidores experiencia experiencias pedido pedidos
produto produtos item itens mercadoria mercadorias categoria categorias uf estado estados
nota notas baixa baixas alta altas negativas negativos positiva positivas insatisfeitos
insatisfeitas insatisfeito insatisfeita insatisfatorias insatisfatorios score min max
principal principais problema problemas reclamacao reclamacoes queixa queixas elogio elogios
aspecto aspectos padrao padroes tendencia tendencias frequencia frequencias frequente
frequentes recorrente recorrentes percentual percentuais porcentagem proporcao maioria
distribuicao ranking mais menos maior maiores menor menores melhor melhores pior piores
relato relatos relatam relata relatado relatados relatada relatadas mencionam menciona
mencionado mencionados mencionada mencionadas mencoes mencionados citam cita citado citados
falam fala dizem diz identificado identificados identificadas identificar encontrados
existe existem sustenta sustentam conclusao evidencias evidencia justificam demonstram
aparecem aparece ocorrencias ocorrencia apresentam apresenta relacionados relacionadas
relacionado relacionada elogiado elogiados elogiada elogiadas valorizam valorizado
publico publica unicos unica unicas unico textual textuais texto textos filtro filtros
tema temas assunto assuntos geral gerais podem pode podemos poderiam deve devem
no nao nunca sem ainda sim recebido recebidos recebida recebidas chegou chegaram recebi
receber receberam prazo prazos transporte transportadora transportadoras qualidade
rapida rapidas rapido rapidos rapidez funcionamento funcionar funciona funcionou
recomendacao recomendacoes recomenda recomendam satisfeito satisfeitos satisfeita satisfeitas
""".split())
_TOPIC_WORD = re.compile(
    r"(?:atras\w*|demor\w*|entreg\w*|logistica|defeit\w*|danific\w*|quebrad\w*|avariad\w*|"
    r"incomplet\w*|errad\w*|diferente\w*|atend\w*|suporte|sac|comunicacao|embalag\w*|"
    r"fiscal|fiscais|documentacao|documentos|troca\w*|devolu\w*|reembols\w*|estorn\w*|"
    r"cancel\w*|pos|venda|preco\w*|custo\w*|elogia\w*|valoriz\w*|reclam\w*|estoque|"
    r"anunciad\w*|boas?|bons?|otim\w*|excelente\w*|ruim|ruins|pessim\w*)")


def _unsupported_focus_terms(question, filters=None):
    text = plain(question).replace("_", " ")
    text = re.sub(r"\bcusto[- ]beneficio\b", " ", text)
    # Category/state/month values are explicit corpus filters, not new themes.
    filter_phrases = set(CATEGORY_ALIASES) | set(STATES) | set(MONTHS)
    for key in ("category", "state", "uf"):
        value = (filters or {}).get(key)
        values = value if isinstance(value, (list, tuple, set)) else [value]
        filter_phrases.update(plain(v).replace("_", " ") for v in values if v)
    for phrase in sorted(filter_phrases, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(phrase)}\b", " ", text)
    accepted = _ANALYTIC_FRAME | {uf.lower() for uf in STATES.values()}
    return sorted({word for word in re.findall(r"[a-z]+", text)
                   if word not in accepted and not _TOPIC_WORD.fullmatch(word)})


def _selected_themes(question, filters=None):
    text = plain(question)
    if _unsupported_focus_terms(question, filters):
        return []
    praise = bool(re.search(r"\belogi\w*|\bvaloriz\w*|\bpositiv\w*|\bmelhores\b", text))
    problem = bool(re.search(r"\bproblem\w*|\breclam\w*|\binsatisfeit\w*|\bnegativ\w*|\bqueixas?\b", text))
    polarity = "praise" if praise and not problem else "problem" if problem and not praise else None
    themes = [t for t in TAXONOMY if polarity is None or t.polarity == polarity]
    group_patterns = {
        "delivery": r"\bentreg\w*|\blogistica\b|\btransportador\w*",
        "service": r"\batend\w*|\bsuporte\b|\bsac\b|\bcomunicacao\b",
        # 'Produtos de beleza' is a category restriction; 'sobre os produtos'
        # or 'qualidade dos produtos' explicitly names the analytical subject.
        "product": r"\b(?:sobre|nos|dos) (?:os )?(?:produtos?|itens?|mercadorias?)\b|"
                   r"\b(?:qualidade|funcionamento) (?:dos? |nos? |de )?(?:produtos?|itens?)\b|"
                   r"\bdefeit\w*|\bdanific\w*|\bquebrad\w*|\bavariad\w*|\bembalag\w*",
        "documentation": r"\bnotas? fiscais?\b|\bnota fiscal\b|\bdocumentacao\b|\bdocumentos fiscais\b",
        "after_sales": r"\bpos[- ]venda\b|\btrocas?\b|\bdevolu\w*|\breembols\w*|\bestorn\w*|\bcancel\w*",
        "price": r"\bprecos?\b|\bcustos?(?:[- ]beneficio)?\b",
    }
    groups = {group for group, pattern in group_patterns.items() if re.search(pattern, text)}
    if (re.search(r"\b(?:qualidade|funcionamento)\b", text)
            and not re.search(r"\b(?:qualidade|funcionamento) (?:do|de|no) (?:atendimento|suporte|servico)\b", text)):
        groups.add("product")
    if not groups and re.search(r"\bprazos?\b", text):
        groups.add("delivery")
    # Narrow only explicit subjects, never guess a theme from an embedding top-k.
    specific = []
    triggers = {"atraso": r"atras|demora", "nao_recebido": r"nao receb|nao cheg|nunca cheg",
        "defeito": r"defeit|nao funciona", "danificado": r"danific|quebrad|avariad",
        "produto_errado": r"produto[s]? (?:errad|diferente)|item diferente",
        "incompleto": r"incomplet|parte dos itens", "nota_fiscal": r"nota fiscal",
        "troca_reembolso": r"reembols|estorn|devolu|trocas?\b"}
    if not re.search(r"\bprincipais\b|\bpadroes\b", text):
        specific = [tid for tid, pattern in triggers.items() if re.search(pattern, text)]
    if specific and len(groups) <= 1:
        themes = [t for t in TAXONOMY if t.id in specific
                  and (polarity is None or t.polarity in {polarity, "neutral"})]
    elif groups:
        themes = [t for t in TAXONOMY if t.group in groups
                  and (polarity is None or t.polarity in {polarity, "neutral"})]
    return themes


def _percent(count, denominator):
    return round(100 * count / denominator, 2) if denominator else None


def render_analytics(analytics):
    """Canonical numeric answer: structured facts are the only input."""
    n = analytics["denominator"]
    count = analytics["count"]
    if not n:
        return "Não há avaliações textuais únicas no recorte informado. Ajuste os filtros."
    if analytics.get("reason") == "unsupported_analytic_subject":
        return ("O foco solicitado não é coberto com segurança pela taxonomia analítica. "
                "Não foi publicado um ranking de outros assuntos. Use uma pergunta específica no modo RAG "
                "ou um dos temas documentados para consultar evidências.")
    if analytics["status"] == "no_evidence":
        return (f"O recorte contém {n} avaliações textuais únicas, mas nenhum tema lexical selecionado foi identificado. "
                "As regras têm cobertura limitada: isso não prova ausência do assunto. Reformule ou consulte exemplos específicos.")
    if analytics["status"] != "ok":
        return (f"O recorte contém {n} avaliações textuais únicas. O mínimo definido para publicar percentuais é "
                f"{analytics['min_base']}; a base é insuficiente para esta análise agregada.")
    lines = [f"Base analisada: {n} avaliações textuais únicas no recorte; contagem sobre o corpus completo filtrado."]
    if not analytics["rows"]:
        lines.append("Nenhum dos temas lexicais selecionados apareceu no recorte. Isso não prova ausência do assunto: as regras têm cobertura limitada.")
    for row in analytics["rows"][:analytics["top_n"]]:
        percent = f"{row['percentage']:.2f}".replace(".", ",")
        refs = " ".join(f"[{c}]" for c in row["citations"])
        lines.append(f"- {row['label']}: {row['count']} avaliações ({percent}% da base textual filtrada). {refs}".rstrip())
    coverage = f"{analytics['coverage_percentage']:.2f}".replace(".", ",")
    lines.append(f"Cobertura de todos os temas selecionados: {count}/{n} avaliações ({coverage}%). A lista mostra até {analytics['top_n']} temas; uma avaliação pode aparecer em vários temas e os percentuais não devem ser somados.")
    lines.append("São menções identificadas por regras lexicais, com exemplos auditáveis; não são causas comprovadas nem proporções de todos os clientes. Avaliações sem texto ficam fora desta base.")
    lines.append("As contagens usam todas as avaliações elegíveis, não a amostra recuperada por similaridade. A nota é apenas um filtro explícito e não prova uma causa.")
    if analytics["multiple_order_reviews"]:
        lines.append(f"Há {analytics['multiple_order_reviews']} avaliações ligadas a mais de um pedido; cada review_id conta uma vez. Todos os filtros de pedido precisam coincidir no mesmo pedido.")
    if analytics["conflicting_score_reviews"]:
        lines.append(f"Há {analytics['conflicting_score_reviews']} avaliações com notas conflitantes; todos os valores foram preservados e precisam satisfazer o filtro de nota.")
    return "\n\n".join(lines)


def validate_analytics(result):
    """Reject swapped counts/percentages/answer text, even with valid citations.

    This is a consistency check, not semantic validation of lexical rules.
    """
    facts = result["analytics"]
    n = facts["denominator"]
    if not isinstance(n, int) or n < 0 or facts["unique_reviews"] != n:
        raise ValueError("Denominador analítico inconsistente.")
    covered = set()
    theme_ids = set()
    evidence = {e["citation"]: e for e in result["evidence"]}
    taxonomy = {theme.id: theme for theme in TAXONOMY}
    for row in facts["rows"]:
        ids = row["review_ids"]
        if row["theme"] in theme_ids or len(ids) != len(set(ids)) or row["count"] != len(ids) or not 0 <= row["count"] <= n:
            raise ValueError("Contagem analítica inconsistente.")
        if row["theme"] not in taxonomy or row["label"] != taxonomy[row["theme"]].label or row["rule"] != taxonomy[row["theme"]].pattern:
            raise ValueError("Tema analítico difere da taxonomia versionada.")
        expected_percentage = _percent(row["count"], n) if facts["percentages_published"] else None
        if row["percentage"] != expected_percentage:
            raise ValueError("Percentual analítico inconsistente.")
        if not all(c in evidence and evidence[c]["review_id"] in ids for c in row["citations"]):
            raise ValueError("Exemplo analítico não pertence à contagem do tema.")
        if not all(row["theme"] in {tid for text in evidence[c].get("texts", [evidence[c]["text"]]) for tid in label_themes(text)} for c in row["citations"]):
            raise ValueError("Exemplo analítico não satisfaz a regra publicada.")
        theme_ids.add(row["theme"])
        covered.update(ids)
    expected_coverage = _percent(len(covered), n) if facts["percentages_published"] else None
    if facts["count"] != len(covered) or facts["coverage_percentage"] != expected_coverage:
        raise ValueError("Cobertura analítica inconsistente.")
    if result["answer"] != render_analytics(facts):
        raise ValueError("Texto analítico difere dos fatos determinísticos.")
    if sorted(result["citations"]) != sorted(evidence):
        raise ValueError("Citações analíticas inconsistentes.")
    return result


def analyze(documents, question, filters=None, top_n=5, min_base=30):
    """Count lexical themes on all eligible review IDs; no model/network call."""
    if not 1 <= int(top_n) <= len(TAXONOMY) or not isinstance(min_base, int) or min_base < 1:
        raise ValueError("Use top_n válido e mínimo de base positivo.")
    filters = {k: v for k, v in (filters or {}).items() if k in FILTER_KEYS}
    full = unique_reviews(documents)
    eligible = [d for d in full if matches_filters(d, **filters)]
    n = len(eligible)
    themes = _selected_themes(question, filters)
    rows = []
    for theme in themes:
        matching = [d for d in eligible if theme.id in d["theme_ids"]]
        if matching:
            rows.append({"theme": theme.id, "label": theme.label, "polarity": theme.polarity,
                "count": len(matching), "percentage": _percent(len(matching), n),
                "review_ids": [d["review_id"] for d in matching], "citations": [],
                "rule": theme.pattern, "negation_check": theme.check_negation})
    rows.sort(key=lambda row: (-row["count"], row["theme"]))
    evidence, citation_by_review = [], {}
    by_id = {d["review_id"]: d for d in eligible}
    for row in rows[:int(top_n)]:
        for rid in row["review_ids"][:2]:
            if rid not in citation_by_review:
                citation = f"E{len(evidence) + 1}"
                citation_by_review[rid] = citation
                doc = dict(by_id[rid], citation=citation)
                doc["matching_orders"] = matched_order_metadata(doc, **filters)
                evidence.append(doc)
            row["citations"].append(citation_by_review[rid])
    covered = {rid for row in rows for rid in row["review_ids"]}
    status = "no_evidence" if not n or not rows else "insufficient_evidence" if n < min_base else "ok"
    facts = {"taxonomy_version": TAXONOMY_VERSION, "status": status,
             "population": "unique_textual_reviews", "denominator": n, "unique_reviews": n,
             "corpus_unique_reviews": len(full), "count": len(covered),
             "coverage_percentage": _percent(len(covered), n), "rows": rows,
             "selected_themes": [t.id for t in themes], "displayed_themes": [r["theme"] for r in rows[:int(top_n)]],
             "filters": filters, "min_base": min_base, "top_n": int(top_n),
             "multiple_order_reviews": sum(len(d["order_ids"]) > 1 for d in eligible),
             "conflicting_score_reviews": sum(len(d["scores"]) > 1 for d in eligible),
             "overlap": True, "denominator_scope": "all_filtered_unique_textual_review_ids",
             "notes": ["Temas por regras, não avaliação humana; cobertura não mede recall.",
                       "A mesma avaliação pode receber múltiplos temas.",
                       "Percentuais referem-se somente à base textual filtrada."]}
    if not themes:
        facts["reason"] = "unsupported_analytic_subject"
        facts["unsupported_terms"] = _unsupported_focus_terms(question, filters)
    if status != "ok":
        # Small groups may show examples, but do not publish unstable percentages.
        for row in rows:
            row["percentage"] = None
        facts["coverage_percentage"] = None
        facts["percentages_published"] = False
    else:
        facts["percentages_published"] = True
    result = {"answer": render_analytics(facts), "evidence": evidence,
              "citations": [e["citation"] for e in evidence], "insufficient": status != "ok",
              "mode": "analytics", "status": status, "analytics": facts}
    if facts.get("reason"):
        result["reason"] = facts["reason"]
    return validate_analytics(result)


answer_analytics = analyze
