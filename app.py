"""NarrativeVoice dashboard. Run: streamlit run app.py

This file only wires things together. Each page lives in ui/page_*.py."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import streamlit as st

from config import DATA, RESULTS, TRANSCRIPTS
from ui import (common, page_explore, page_map, page_methods, page_overview, page_patterns, page_quality,
                page_review, page_start, page_upload, sidebar)

st.set_page_config(page_title="NarrativeVoice", layout="wide")
st.markdown(common.CSS, unsafe_allow_html=True)

PAGE_FUNCS = {"Start here": page_start, "Overview": page_overview, "Explore a theme": page_explore,
              "Map and search": page_map, "Patterns": page_patterns, "Quality checks": page_quality,
              "Review and report": page_review, "Methods and choices": page_methods,
              "Upload transcripts": page_upload}

datasets = {}
if (RESULTS / "answers.csv").exists():
    datasets["Demo: synthetic interviews"] = (RESULTS, TRANSCRIPTS)
if (RESULTS / "uploads" / "answers.csv").exists():
    datasets["My uploaded transcripts"] = (RESULTS / "uploads", DATA / "uploads")

if not datasets:
    st.error("No results yet. In Terminal run:  python src/generate_data.py  then  python src/pipeline.py")
    st.stop()

ctx = sidebar.build(datasets)
PAGE_FUNCS[st.session_state["page"]].render(ctx)
