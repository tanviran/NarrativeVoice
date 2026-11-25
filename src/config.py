"""Central settings. Change things here, not inside the other files."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TRANSCRIPTS = DATA / "transcripts"
PARTICIPANTS = DATA / "participants.csv"
GROUND_TRUTH = DATA / "ground_truth.csv"   # planted labels, used ONLY by evaluate.py
RESULTS = ROOT / "results"
CACHE = RESULTS / "cache"

SEED = 42
BACKEND = os.environ.get("NV_EMBEDDER", "minilm")   # minilm | mpnet | tfidf
MIN_WORDS = 5            # answers shorter than this are kept but not themed
K_RANGE = (6, 16)

# Generic interviewer follow-ups. They carry no topic, so the answer keeps the
# last MAIN question as its context.
PROBES = {
    "Can you say more about that?",
    "What was that like for you?",
    "How did that affect you?",
}
