"""Shared dashboard pieces: data loading (cached), colours, and the Ctx object every page receives."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from embed import Embedder
from evidence import verify_frame
from themes import find_themes

# Colour-blind-safe palettes (Okabe-Ito and Paul Tol), so meaning never depends on red versus green.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#999999",
           "#332288", "#44AA99", "#88CCEE", "#CC6677", "#DDCC77", "#117733", "#882255", "#AA4499"]
TONE_COLOURS = {"negative": "#D55E00", "positive": "#0072B2", "uncertain": "#B0B0B0"}
GROUPS = [("region", "Region"), ("age_band", "Age band"), ("time_in_care", "Time in care"),
          ("ethnic_group", "Ethnic group")]

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
h1 {font-weight: 700; letter-spacing: -0.5px;}
[data-testid="stMetric"] {background: #F2F5F9; border-radius: 10px; padding: 12px 16px;}
.nv-quote {border-left: 4px solid #2F5D8A; background: #F2F5F9; padding: 8px 14px; border-radius: 4px; margin: 6px 0;}
.nv-quote.hl {border-left-color: #E69F00; background: #FFF6E0;}
.nv-muted {color: #5b6b7b; font-size: 0.9rem;}
</style>
"""


@dataclass
class Ctx:
    """Everything a page needs. `full` is all answers; `view` is what the sidebar filters leave."""
    name: str
    source_dir: Path
    results_dir: Path
    full: pd.DataFrame
    view: pd.DataFrame
    emb: np.ndarray
    info: dict
    meta: dict
    labels: dict
    group_cols: list

    @property
    def k(self):
        return self.info["k"]

    @property
    def themed_view(self):
        return self.view[self.view.theme_id >= 0]

    def colour_map(self):
        return {self.labels[t]: PALETTE[t % len(PALETTE)] for t in self.labels}


def goto(page):
    """Button callback: jump to another page."""
    st.session_state["page"] = page


@st.cache_data(show_spinner="Loading results...")
def load_results(results_dir):
    d = Path(results_dir)
    return (pd.read_csv(d / "answers.csv"), np.load(d / "embeddings.npy"),
            json.loads((d / "run_meta.json").read_text()))


@st.cache_data(show_spinner="Finding themes...")
def themed(results_dir, k, min_words):
    df, emb, _ = load_results(results_dir)
    return find_themes(df, emb, k=k, min_words=min_words)


@st.cache_resource
def get_embedder(results_dir):
    df, _, meta = load_results(results_dir)
    e = Embedder(meta["backend"])
    if meta["backend"] == "tfidf":
        e.fit_encode(df.text.tolist())       # refit on the same corpus to rebuild the same space
    return e


@st.cache_data
def verification_rate(results_dir, source_dir):
    df, _, _ = load_results(results_dir)
    ok = verify_frame(df, source_dir)
    return sum(ok), len(ok)


def short_question(df, qid, width=60):
    return f"{qid}: {df.loc[df.question_id == qid, 'question'].iloc[0][:width]}"
