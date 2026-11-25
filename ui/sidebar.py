"""Sidebar: dataset, navigation, theme settings and filters. Returns the Ctx for the pages."""
import streamlit as st

from themes import label
from ui import common
from ui.common import Ctx, GROUPS


PAGES = ["Start here", "Overview", "Explore a theme", "Map and search", "Patterns", "Quality checks",
         "Review and report", "Methods and choices", "Upload transcripts"]

def build(datasets):
    """datasets: {display name: (results_dir, source_dir)}. Returns a Ctx."""
    # NarrativeVoice branding at the top of the sidebar
    st.sidebar.image(
        "ui/assets/narrativevoice_logo.png",
        width=200
    )

   # st.sidebar.markdown("## NarrativeVoice")
    st.sidebar.caption("Synthetic or uploaded transcripts only. The AI proposes, the researcher decides.")
    name = st.sidebar.selectbox("Dataset", list(datasets)) if len(datasets) > 1 else next(iter(datasets))
    results_dir, source_dir = datasets[name]
    st.sidebar.radio("Navigate", PAGES, key="page")

    full0, emb, meta = common.load_results(str(results_dir))
    with st.sidebar.expander("Theme settings"):
        auto = st.radio("Number of themes", ["Automatic", "Manual"], horizontal=True) == "Automatic"
        k = st.slider("Themes (k)", 2, max(3, min(20, len(full0) - 1)), min(9, max(3, len(full0) // 10)),
                      disabled=auto)
        min_words = st.slider("Ignore answers shorter than (words)", 1, 10, 5)
    full, info = common.themed(str(results_dir), None if auto else k, min_words)
    labels = {t: label(info, t) for t in range(info["k"])}

    group_cols = [(c, n) for c, n in GROUPS if c in full and full[c].nunique() > 1]
    view = full
    with st.sidebar.expander("Filters", expanded=False):
        for col, title in group_cols:
            chosen = st.multiselect(title, sorted(full[col].dropna().unique()))
            if chosen:
                view = view[view[col].isin(chosen)]
        qs = {common.short_question(full, q, 45): q for q in sorted(full.question_id.unique(), key=lambda s: int(s[1:]))}
        chosen_q = st.multiselect("Interview question", list(qs))
        if chosen_q:
            view = view[view.question_id.isin([qs[x] for x in chosen_q])]
        chosen_t = st.multiselect("Tone", ["negative", "positive", "uncertain"])
        if chosen_t:
            view = view[view.tone.isin(chosen_t)]
        if not st.checkbox("Include interviewer follow-up answers", True):
            view = view[~view.is_followup]
    if len(view) < len(full):
        st.sidebar.info(f"Filters on: {len(view)} of {len(full)} answers shown.")

    with st.sidebar.expander("How reliable are these themes?"):
        st.write(f"Stability: **{info['stability']:.2f}** (1.0 means identical themes on every run)")
        if info["scores"]:
            st.write(f"Silhouette at k={info['k']}: **{info['scores'][info['k']]:.2f}**")
        st.write(f"Embedder `{meta['backend']}`, tone `{meta['tone_method']}`")
        if info["stability"] < 0.5:
            st.warning("Low stability: treat these themes as rough.")
        if meta["tone_method"].startswith("lexicon"):
            st.warning("Tone uses a crude word list because the sentiment model was unavailable.")

    st.sidebar.markdown("""
    <style>
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        display: flex;
        flex-direction: column;
        height: 100vh;
    }

    [data-testid="stSidebar"] [data-testid="stSidebarContent"] > div:last-child {
        margin-top: auto;
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.image(
        "ui/assets/rav_logo.png",
        width=180
    )

    return Ctx(name, source_dir, results_dir, full, view, emb, info, meta, labels, group_cols)
