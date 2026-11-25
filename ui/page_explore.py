"""Explore one theme: quotes, tone, the full interview around a quote, and an optional AI-drafted insight."""
import pandas as pd
import plotly.express as px
import streamlit as st

from evidence import verify_frame
from insights import insight_for
from themes import representative_rows
from ui.common import TONE_COLOURS


def show_interview(ctx, participant, highlight_line):
    """Read one participant's whole interview, with the chosen answer highlighted and each answer tagged."""
    rows = ctx.full[ctx.full.participant_id == participant].sort_values("line_no")
    last_q = None
    for r in rows.itertuples():
        if r.question_id != last_q:
            st.markdown(f"<span class='nv-muted'>{r.question_id}: {r.question}</span>", unsafe_allow_html=True)
            last_q = r.question_id
        tag = ctx.labels.get(r.theme_id, "too short to theme")
        css = "nv-quote hl" if r.line_no == highlight_line else "nv-quote"
        st.markdown(f"<div class='{css}'>{r.text}<br><span class='nv-muted'>line {r.line_no} | {tag} | "
                    f"{r.tone}</span></div>", unsafe_allow_html=True)


def render(ctx):
    st.title("Explore a theme")
    tv = ctx.themed_view
    t = st.selectbox("Theme", range(ctx.k), format_func=lambda i: ctx.labels[i])
    sub = tv[tv.theme_id == t]
    if sub.empty:
        st.info("No answers from this theme match the current filters.")
        return

    a, b, c = st.columns(3)
    a.metric("Participants", sub.participant_id.nunique())
    b.metric("Answers", len(sub))
    c.metric("Share of participants in view", f"{sub.participant_id.nunique() / ctx.view.participant_id.nunique():.0%}")
    st.markdown("**Distinctive words:** " + ", ".join(ctx.info["keywords"][t]))

    left, right = st.columns([2, 1])
    with right:
        fig = px.pie(sub, names="tone", color="tone", color_discrete_map=TONE_COLOURS, hole=0.5, height=260)
        fig.update_layout(template="plotly_white", margin=dict(t=10, b=10), legend_title="")
        st.plotly_chart(fig, width="stretch")
    with left:
        n_show = st.slider("Quotes to show", 3, 40, 8)
        order = st.radio("Order", ["Most typical first", "Negative tone first", "Positive tone first"], horizontal=True)
    reps = representative_rows(sub, ctx.emb, ctx.info["centers"][t], n=len(sub))
    if order != "Most typical first":
        first = "negative" if order.startswith("Neg") else "positive"
        reps = pd.concat([reps[reps.tone == first], reps[reps.tone != first]])
    shown = reps.head(n_show).copy()
    shown["verified"] = ["yes" if v else "NO" for v in verify_frame(shown, ctx.source_dir)]

    st.subheader("Supporting quotes")
    st.caption("One answer per participant, so one voice cannot dominate. 'verified' means the text was found word for "
               "word at that line of the original transcript.")
    cols = ["participant_id", "line_no", "question_id", "tone", "text", "verified"]
    if "region" in shown:
        cols.insert(4, "region")
    st.dataframe(shown[cols], hide_index=True, width="stretch",
                 column_config={"text": st.column_config.TextColumn("Answer", width="large"),
                                "participant_id": "Participant", "line_no": "Line", "question_id": "Question"})

    st.subheader("Read it in context")
    pick = st.selectbox("Choose a quote to see the whole interview around it",
                        [f"{r.participant_id}, line {r.line_no}" for r in shown.itertuples()])
    pid, line = pick.split(", line ")
    show_interview(ctx, pid, int(line))

    st.subheader("AI-drafted insight (optional, local model)")
    st.caption("A local model sees only the quotes above. Every claim must cite its evidence and pass automatic checks. "
               "It is an unreviewed draft: read each claim against its quotes.")
    if st.button("Draft insight from these quotes"):
        ev = [{"participant": r.participant_id, "line": int(r.line_no), "text": r.text} for r in shown.itertuples()]
        with st.spinner("Asking the local model..."):
            st.session_state[f"insight_{ctx.name}_{ctx.k}_{t}"] = insight_for(ev)
    res = st.session_state.get(f"insight_{ctx.name}_{ctx.k}_{t}")
    if res:
        if res["claims"]:
            for cl in res["claims"]:
                refs = ", ".join(f"{e['participant']} line {e['line']}" for e in cl["evidence"])
                st.write(f"- {cl['text']}  *(evidence: {refs})*")
        else:
            st.warning(res["error"])
