"""Check that a quote really appears at that participant and line of the original transcript."""
from functools import lru_cache
from pathlib import Path
from config import TRANSCRIPTS


@lru_cache(maxsize=1024)
def _lines(participant_id, folder):
    path = Path(folder) / f"{participant_id}.txt"
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def verify_quote(participant_id, line_no, text, folder=TRANSCRIPTS):
    """True only if `text` is on that line of that participant's file, spoken by the PARTICIPANT."""
    lines = _lines(participant_id, str(folder))
    if not (1 <= line_no <= len(lines)):
        return False
    line = lines[line_no - 1]
    return line.startswith("PARTICIPANT:") and text.strip() in line


def verify_frame(df, folder=TRANSCRIPTS):
    return [verify_quote(r.participant_id, int(r.line_no), r.text, folder) for r in df.itertuples()]
