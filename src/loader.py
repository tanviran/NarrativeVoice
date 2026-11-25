"""Read transcripts into one row per participant answer, WITH its interviewer question."""
import re
from pathlib import Path
import pandas as pd
from config import TRANSCRIPTS, PARTICIPANTS, PROBES

LINE = re.compile(r"^(INTERVIEWER|PARTICIPANT):\s*(.+)$")


def is_probe(text):
    """A probe is a generic follow-up ('Can you say more?'), not a new topic."""
    return text.strip() in PROBES or len(text.split()) <= 6


def load_segments(folder=TRANSCRIPTS, meta=PARTICIPANTS):
    rows, q_ids = [], {}
    for path in sorted(Path(folder).glob("*.txt")):
        pid = path.stem
        question, followup = "(no question recorded)", False
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            m = LINE.match(raw.strip())
            if not m:
                continue                      # header comments and blank lines
            speaker, text = m.group(1), m.group(2).strip()
            if speaker == "INTERVIEWER":
                if is_probe(text):
                    followup = True           # keep the earlier main question as context
                else:
                    question, followup = text, False
                continue
            q_id = q_ids.setdefault(question, f"Q{len(q_ids) + 1}")
            rows.append({"participant_id": pid, "line_no": line_no, "question_id": q_id,
                         "question": question, "is_followup": followup,
                         "text": text, "n_words": len(text.split())})
            followup = False
    if not rows:
        raise SystemExit(f"No transcript lines found in {folder}. Run: python src/generate_data.py")
    df = pd.DataFrame(rows)
    if Path(meta).exists():
        df = df.merge(pd.read_csv(meta), on="participant_id", how="left")
    return df


if __name__ == "__main__":
    df = load_segments()
    print(f"{len(df)} answers from {df.participant_id.nunique()} participants")
    print(df.groupby(["question_id", "question"]).size().to_string())
    print(df.head(4).to_string())
