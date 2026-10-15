"""Interface de investigação e análise de voz do cliente."""
import json
import os
import streamlit as st
from voc.retrieval import Retriever
from voc.pipeline import answer_question

st.set_page_config(page_title="Olist · Voz do Cliente", page_icon="💬", layout="wide")
st.markdown("""<style>
.block-container{max-width:1350px;padding-top:2.2rem}h1{letter-spacing:-.04em}
[data-testid="stMetric"]{border:1px solid #263448;border-radius:12px;padding:16px}
[data-testid="stSidebar"]{border-right:1px solid #263448}
</style>""", unsafe_allow_html=True)
st.caption("OLIST  /  INTELIGÊNCIA DA VOZ DO CLIENTE")
st.title("Escute os relatos. Investigue com evidências.")
st.write("Pergunte sobre experiências de compra ou compare os temas mencionados na base histórica da Olist.")


@st.cache_resource
def load_index(path):
    return Retriever(path)


with st.sidebar:
    st.header("Sua investigação")
    mode = st.selectbox("Resposta", ["extractive", "llm"], format_func=lambda x: "Explorar evidências" if x == "extractive" else "Síntese verificada com LLM")
    st.caption("Perguntas de frequência usam contagens calculadas sobre todo o recorte, independentemente do modo LLM.")
    scores = st.slider("Notas", 1, 5, (1, 5))
    with st.expander("Busca e configuração"):
        path = st.text_input("Diretório do índice", "data/index")
        algorithm = st.selectbox("Estratégia", ["hybrid", "dense"], format_func=lambda x: {"hybrid":"Híbrida · vetores + BM25", "dense":"Semântica · vetores"}[x])
        k = st.slider("Máximo de avaliações", 2, 12, 6)
        threshold = st.slider("Similaridade mínima", 0.0, 1.0, 0.30, 0.05)
        route = st.selectbox("Tipo de investigação", ["auto", "rag", "analytics"], format_func=lambda x: {"auto":"Identificar pela pergunta", "rag":"Investigar relatos", "analytics":"Contar temas na base"}[x])
        st.caption("LLM: " + os.environ.get("LLM_MODEL", "qwen2.5:7b") + " · configure as variáveis conforme o README.")
try:
    retriever = load_index(path)
except Exception as error:
    st.info("Construa o índice para começar: python -m voc.cli ingest --raw data/raw")
    st.caption(f"Índice indisponível: {type(error).__name__}")
    st.stop()

categories = sorted({c for d in retriever.docs for c in d["categories"]})
with st.sidebar:
    category = st.selectbox("Categoria", ["Todas"] + categories)
    state = st.selectbox("UF do cliente", ["Todas"] + sorted({uf for d in retriever.docs for uf in d.get("states", [])}))
    delivery = st.selectbox("Situação da entrega", ["Todas", "late", "on_time", "not_delivered", "unknown"], format_func=lambda x: {"Todas":"Todas", "late":"Atraso confirmado", "on_time":"Dentro da data prevista", "not_delivered":"Pedido não entregue", "unknown":"Sem informação suficiente"}[x])
    use_dates = st.checkbox("Filtrar período da compra")
    date_from = st.date_input("Compra a partir de") if use_dates else None
    date_to = st.date_input("Compra até") if use_dates else None

stats = retriever.manifest.get("stats", {})
a, b, c = st.columns(3)
a.metric("Documentos textuais", f'{stats.get("indexed_documents", stats.get("documents", len(retriever.docs))):,}'.replace(",", "."))
b.metric("Avaliações únicas", f'{len({d["review_id"] for d in retriever.docs}):,}'.replace(",", "."))
c.metric("Base histórica", "2016–2018")
st.caption("Os filtros da pergunta e dos controles são combinados por interseção. Conflitos precisam ser corrigidos; a resolução aparece no resultado.")

examples = ["O que os clientes relatam sobre atrasos na entrega?", "Quais os temas mais frequentes nas avaliações com notas 1 e 2 em SP?", "O que os clientes elogiam sobre a rapidez da entrega?", "Quais relatos mencionam produtos quebrados ou com defeito?"]
selected_example = st.selectbox("Sugestão de pergunta", examples)
with st.form("question"):
    question = st.text_area("Pergunta", selected_example, max_chars=2000, height=100)
    submitted = st.form_submit_button("Investigar", type="primary")
if submitted:
    if use_dates and date_from > date_to:
        st.error("A data inicial deve ser anterior à final.")
    else:
        filters = {"k": k, "algorithm": algorithm, "min_similarity": threshold}
        if scores != (1, 5):
            filters.update(score_min=scores[0], score_max=scores[1])
        if category != "Todas":
            filters["category"] = category
        if state != "Todas":
            filters["state"] = state
        if delivery != "Todas":
            filters["delivery_status"] = delivery
        if use_dates:
            filters.update(date_from=str(date_from), date_to=str(date_to))
        try:
            with st.spinner("Investigando os comentários e conferindo as fontes…"):
                st.session_state["result"] = answer_question(retriever, question, mode=mode, route=route, **filters)
        except ValueError as error:
            st.error(str(error))

result = st.session_state.get("result")
if result:
    st.divider()
    status = result.get("status", "ok")
    st.caption(f'{result.get("mode", mode).upper()}  ·  {result.get("elapsed_seconds", 0):.2f}s  ·  {status}')
    st.subheader("Resultado da investigação")
    if result["insufficient"]:
        st.info(result["answer"])
    else:
        st.write(result["answer"])
    resolved = result.get("resolved_query")
    if resolved:
        with st.expander("Como a pergunta foi interpretada"):
            for note in resolved.get("notes", []):
                st.caption(note)
            st.json(resolved["filters"])
    analytics = result.get("analytics")
    if analytics:
        rows = analytics.get("rows", [])
        if rows:
            table = [{"Tema": row["label"], "Avaliações": row["count"],
                      "% da base textual": row["percentage"],
                      "Fontes de exemplo": ", ".join(row["citations"])}
                     for row in rows[:analytics.get("top_n", 5)]]
            st.dataframe(table, hide_index=True, width="stretch")
        st.caption("Categorias de temas são regras lexicais sobre comentários e podem se sobrepor. Consulte cobertura e denominador no resultado; não são estimativas de causas nem de toda a clientela.")
    st.subheader("Avaliações de origem")
    for e in result["evidence"]:
        metric = f' · cosseno {e["similarity"]:.3f}' if "similarity" in e else " · exemplo do tema"
        with st.expander(f'[{e["citation"]}] Nota {e["score"]}{metric}', expanded=True):
            st.text(e["text"])
            st.caption(f'Review: {e["review_id"]}  |  Pedido: {e["order_id"]}')
            if len(e.get("order_ids", [])) > 1:
                st.caption("Este review está ligado a mais de um pedido. Os pedidos que satisfazem o recorte aparecem em matching_orders; o texto não identifica a qual deles se refere.")
            st.json({key: e[key] for key in ["purchase_date", "categories", "states", "delivery_status", "late_delivery", "matching_orders", "order_metadata"] if key in e})
    st.download_button("Baixar resposta, filtros e fontes", json.dumps(result, ensure_ascii=False, indent=2), "resultado-voc.json", "application/json")
st.divider()
st.caption("Tech Challenge · Fase 4  |  Dados Olist / Kaggle  |  Respostas apoiadas nos comentários históricos, com limites explícitos.")
