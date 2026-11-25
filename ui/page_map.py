"""Map of every answer (select points to read them) and search by meaning."""
import numpy as np
import plotly.express as px
import streamlit as st

from ui.common import TONE_COLOURS, get_embedder


def _map(ctx):
    st.caption("Each point is one answer. Points close together mean similar meaning. Use the box or lasso tool to "
               "select a group of points and read them below.")
    colour = st.radio("Colour by", ["Theme", "Tone", "Region", "Question"], horizontal=True)
    col = {"Theme": "theme", "Tone": "tone", "Region": "region", "Question": "question_id"}[colour]
    if col == "region" and "region" not in ctx.view:
        st.info("This dataset has no region information. Colouring by theme instead.")
        col = "theme"
    pdf = ctx.view.copy()
    pdf["theme"] = pdf.theme_id.map(lambda i: ctx.labels.get(i, "too short"))
    pdf["row_id"] = pdf.index
    pdf["snippet"] = pdf.text.str.slice(0, 110)
    cmap = TONE_COLOURS if col == "tone" else (ctx.colour_map() if col == "theme" else None)
    fig = px.scatter(pdf, x="x", y="y", color=col, custom_data=["row_id"], color_discrete_map=cmap,
                     hover_data={"participant_id": True, "snippet": True, "x": False, "y": False}, height=600)
    fig.update_traces(marker=dict(size=7, opacity=0.8))
    fig.update_layout(template="plotly_white", xaxis_visible=False, yaxis_visible=False, legend_title="")
    try:
        ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode=("box", "lasso", "points"),
                             key="answer_map")
        ids = [p["customdata"][0] for p in (ev["selection"]["points"] if ev else []) if p.get("customdata")]
    except TypeError:                      # older Streamlit without selection support
        st.plotly_chart(fig, width="stretch")
        ids = []
    if ids:
        picked = pdf.loc[pdf.row_id.isin(ids)]
        st.markdown(f"**{len(picked)} answers selected from {picked.participant_id.nunique()} participants**")
        st.dataframe(picked[["participant_id", "question_id", "theme", "tone", "text"]], hide_index=True,
                     width="stretch", column_config={"text": st.column_config.TextColumn("Answer", width="large")})


def _search(ctx):
    st.caption("Searches by meaning, so 'nobody listened to me' also finds answers that use different words.")
    query = st.text_input("Search", placeholder="waiting for mental health help")
    top_n = st.slider("Results", 5, 50, 15)
    if not query.strip():
        st.info("Type a phrase above. Results respect the sidebar filters.")
        return
    q = get_embedder(str(ctx.results_dir)).encode([query])[0]
    idx = ctx.view.index.to_numpy()
    sims = ctx.emb[idx] @ q
    best = np.argsort(-sims)[:top_n]
    res = ctx.view.loc[idx[best]].copy()
    res["similarity"] = np.round(sims[best], 2)
    res["theme"] = res.theme_id.map(lambda i: ctx.labels.get(i, "too short"))
    st.dataframe(res[["similarity", "participant_id", "question_id", "theme", "tone", "text"]], hide_index=True,
                 width="stretch", column_config={"text": st.column_config.TextColumn("Answer", width="large")})
    st.caption("Similarity is cosine similarity (1.0 = identical meaning). Scores differ between models, so judge "
               "results by reading them.")


def render(ctx):
    st.title("Map and search")
    if ctx.view.empty:
        st.warning("No answers match the current filters.")
        return
    t1, t2 = st.tabs(["Map of answers", "Search by meaning"])
    with t1:
        _map(ctx)
    with t2:
        _search(ctx)
