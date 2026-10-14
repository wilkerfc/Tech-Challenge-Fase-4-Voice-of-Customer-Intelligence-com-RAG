import json
import streamlit as st
from voc.retrieval import Retriever
from voc.pipeline import answer_question

st.set_page_config(page_title="Olist | Voz do Cliente", page_icon="💬", layout="wide")
st.title("Voz do Cliente · Olist")
st.caption("Explore experiências de compra com busca semântica e evidências rastreáveis.")


@st.cache_resource
def load_index(path):
    return Retriever(path)


with st.sidebar:
    st.header("Exploração")
    path = st.text_input("Diretório do índice", "data/index")
    mode = st.selectbox("Resposta", ["extractive", "llm"], format_func=lambda x: "Evidências sem LLM" if x == "extractive" else "Síntese com LLM")
    scores = st.slider("Notas", 1, 5, (1, 5))
    k = st.slider("Máximo de avaliações", 2, 12, 6)
    threshold = st.slider("Similaridade mínima", 0.0, 1.0, 0.30, 0.05)
    late = st.checkbox("Somente entregas com atraso confirmado")
try:
    retriever = load_index(path)
except Exception as error:
    st.info("Primeiro construa o índice: python -m voc.cli ingest --raw data/raw")
    st.caption(f"Índice indisponível: {type(error).__name__}")
    st.stop()
categories = sorted({c for d in retriever.docs for c in d["categories"]})
category = st.sidebar.selectbox("Categoria", ["Todas"] + categories)
use_dates = st.sidebar.checkbox("Filtrar período da compra")
date_from = st.sidebar.date_input("Compra a partir de") if use_dates else None
date_to = st.sidebar.date_input("Compra até") if use_dates else None
stats = retriever.manifest["stats"]
a, b, c = st.columns(3)
a.metric("Avaliações indexadas", stats.get("indexed_documents", len({d["id"] for d in retriever.docs})))
b.metric("Fragmentos", len(retriever.docs))
c.metric("Dimensão dos embeddings", retriever.manifest["dimension"])
st.caption("As notas e os textos refletem a base histórica. Resultados top-k não representam a frequência de temas na população.")
with st.form("question"):
    question = st.text_area("Pergunta", "O que os clientes relatam sobre atrasos na entrega?", max_chars=2000)
    submitted = st.form_submit_button("Investigar", type="primary")
if submitted:
    if use_dates and date_from > date_to:
        st.error("A data inicial deve ser anterior à final.")
    else:
        try:
            with st.spinner("Recuperando avaliações..."):
                result = answer_question(retriever, question, mode=mode, k=k, min_similarity=threshold, score_min=scores[0], score_max=scores[1],
                                         category=None if category == "Todas" else category, date_from=str(date_from) if date_from else None,
                                         date_to=str(date_to) if date_to else None, late_only=late)
            st.session_state["result"] = result
        except ValueError as error:
            st.error(str(error))
result = st.session_state.get("result")
if result:
    st.subheader("Resposta")
    st.write(result["answer"])
    st.subheader("Evidências")
    for e in result["evidence"]:
        with st.expander(f'[{e["citation"]}] Nota {e["score"]} · Similaridade {e["similarity"]:.3f}', expanded=True):
            st.write(e["text"])
            st.json({key: e[key] for key in ["review_id", "order_id", "purchase_date", "categories", "late_delivery"]})
    st.download_button("Exportar resposta e evidências", json.dumps(result, ensure_ascii=False, indent=2), "resultado-voc.json", "application/json")
