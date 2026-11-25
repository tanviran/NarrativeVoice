"""Overview: how many people raised each theme, tone within themes, and which questions produced them."""
import pandas as pd
import plotly.express as px
import streamlit as st

from ui.common import TONE_COLOURS, short_question


def render(ctx):
    st.title("Overview")
    st.caption("A theme counts as widespread when many different participants raised it, not when a few said a lot.")
    tv = ctx.themed_view
    if tv.empty:
        st.warning("No themed answers match the current filters.")
        return

    by = tv.groupby("theme_id").participant_id.nunique().reindex(range(ctx.k), fill_value=0)
    names = [ctx.labels[t] for t in by.index]
    fig = px.bar(x=by.values, y=names, orientation="h", color=names, color_discrete_map=ctx.colour_map(),
                 labels={"x": "participants", "y": ""}, height=70 + 36 * ctx.k)
    fig.update_layout(showlegend=False, template="plotly_white")
    fig.update_yaxes(autorange="reversed")
    st.subheader("Participants per theme")
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Tone within each theme")
        pct = st.toggle("Show as percentage", True)
        tt = tv.groupby(["theme_id", "tone"]).size().reset_index(name="answers")
        tt["theme"] = tt.theme_id.map(ctx.labels)
        fig = px.bar(tt, x="answers", y="theme", color="tone", orientation="h", color_discrete_map=TONE_COLOURS,
                     height=70 + 36 * ctx.k)
        if pct:
            fig.update_layout(barnorm="percent")
        fig.update_layout(template="plotly_white", legend_title="")
        fig.update_yaxes(autorange="reversed", title="")
        st.plotly_chart(fig, width="stretch")
        st.caption("Themes group by topic, so one theme can hold both good and bad experiences. Tone shows the split.")
    with right:
        st.subheader("Which questions produced which themes")
        ct = pd.crosstab(tv.theme_id.map(ctx.labels), tv.question_id)
        fig = px.imshow(ct, text_auto=True, aspect="auto", color_continuous_scale="Blues", height=70 + 36 * ctx.k,
                        labels={"x": "question", "y": "", "color": "answers"})
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, width="stretch")
        with st.expander("Question list"):
            for q in sorted(ctx.full.question_id.unique(), key=lambda s: int(s[1:])):
                st.write(short_question(ctx.full, q, 90))
