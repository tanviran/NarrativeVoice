"""Methods and choices: what was chosen, what else was possible, and measured comparisons."""
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from config import RESULTS

CHOICES = [
    ("Unit of analysis", "One answer (a transcript line)", "Whole interview, single sentence",
     "Interviews mix topics, sentences are too short to carry meaning, an answer is quotable and has a line number."),
    ("Interviewer question", "Stored with each answer, not mixed into the text", "Ignore it, or embed question and answer together",
     "Short answers need their question. Embedding the question would make answers cluster by question wording."),
    ("Representing meaning", "Sentence embeddings (MiniLM, local)", "TF-IDF, topic models (LDA), larger model, hosted API",
     "Matches meaning not words, runs on a laptop, and no participant text leaves the machine."),
    ("Finding themes", "K-means, k chosen by silhouette, adjustable live", "Agglomerative, HDBSCAN, BERTopic, LLM coding",
     "Simple to explain, assigns every answer, gives an objective 'most typical' answer. HDBSCAN leaves many unassigned."),
    ("Labelling themes", "Distinctive words (TF-IDF)", "LLM-written names",
     "Computed from the data so it cannot invent framing. The researcher renames themes in review."),
    ("Evidence", "Typical answers, one per participant, verified verbatim", "Random quotes, LLM-picked quotes",
     "Objective, repeatable, one voice cannot dominate, and altered quotes are rejected."),
    ("Tone", "Small sentiment model, shown separately", "Ignore tone, LLM tone, manual only",
     "Topic embeddings mix good and bad experiences, so tone makes the split visible. It is approximate."),
    ("Theme links", "Lift plus Fisher test with correction", "Raw counts, LLM-extracted causes",
     "Raw counts make big themes look linked to everything. Cause cannot be verified from text."),
    ("Insight text", "Optional local LLM behind hard checks", "No LLM, hosted API",
     "Meets the 'AI-generated insight' need while keeping data local. Claims must cite evidence and pass checks."),
]


def _pick(df, name):
    return df[df.method.str.startswith(name)] if "method" in df else df


def render(ctx):
    st.title("Methods and choices")
    st.caption("What I chose, what else I could have done, and why. Full reasoning is in docs/decisions.md.")
    st.dataframe(pd.DataFrame(CHOICES, columns=["Decision", "Chosen", "Alternatives", "Why"]), hide_index=True,
                 width="stretch", column_config={"Why": st.column_config.TextColumn(width="large")})

    st.header("Measured comparison")
    path = RESULTS / "evaluation.json"
    if not path.exists():
        st.info("Run  python src/evaluate.py  to measure the options on this machine. The results will appear here.")
        return
    ev = json.loads(path.read_text())
    st.caption(f"Generated {ev['generated']} on {ev['n_answers']} synthetic answers generated from "
               f"{ev['planted_topics']} planted topics. These scores compare methods on easy, template-built data. "
               "They do not show real-world accuracy.")
    t1, t2, t3 = st.tabs(["Embedding models", "Clustering methods", "Tone"])
    with t1:
        st.dataframe(pd.DataFrame(ev["embedders"]), hide_index=True, width="stretch")
        c = pd.DataFrame(ev["clustering"])
        km = c[c.method == "k-means (k=8)"]
        fig = px.bar(km, x="embedder", y=["ARI", "NMI"], barmode="group", height=320,
                     title="Recovery of the planted topics with k-means (higher is better)")
        fig.update_layout(template="plotly_white", legend_title="")
        st.plotly_chart(fig, width="stretch")
    with t2:
        c = pd.DataFrame(ev["clustering"])
        st.dataframe(c, hide_index=True, width="stretch")
        fig = px.scatter(c.dropna(subset=["ARI"]), x="assigned_pct", y="ARI", color="method", symbol="embedder",
                         size_max=14, height=380, title="Accuracy against how many answers were given a theme")
        fig.update_traces(marker=dict(size=13))
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, width="stretch")
        st.caption("A method can score well by leaving most answers unassigned. HDBSCAN often does this: "
                   "check the assigned percentage before comparing accuracy.")
    with t3:
        st.dataframe(pd.DataFrame(ev["tone"]), hide_index=True, width="stretch")
        st.caption("Agreement is measured only on clear-cut positive or negative answers.")
    v = ev["verification"]
    st.metric("Answers verified word for word at their stated line", f"{v['verified']} of {v['total']}")
