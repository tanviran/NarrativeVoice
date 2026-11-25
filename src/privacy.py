"""A simple privacy scan: flag text that may identify a person, so a human can check before analysis.

This is a pattern check, not anonymisation. It misses many things (nicknames, school names, unusual
spellings), so it supports a human check and never replaces one."""
import re
import pandas as pd

PATTERNS = {
    "email address": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone number": re.compile(r"(?<!\d)(?:\+44\s?\d{3,4}|0\d{3,4})[\s-]?\d{3}[\s-]?\d{3,4}(?!\d)"),
    "UK postcode": re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b", re.I),
    "name given": re.compile(r"\b(?:my name is|i am called|call me)\s+[A-Za-z]+", re.I),
    "titled name": re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+[A-Z][a-z]+"),
    "street address": re.compile(r"\b\d{1,4}\s+[A-Z][a-z]+\s+(?:Road|Street|Lane|Avenue|Close|Drive)\b"),
}


def scan(df):
    """One row per possible identifier found in an answer."""
    rows = []
    for r in df.itertuples():
        for kind, rx in PATTERNS.items():
            for m in rx.finditer(r.text):
                rows.append({"participant_id": r.participant_id, "line_no": r.line_no, "type": kind,
                             "match": m.group(0), "text": r.text})
    return pd.DataFrame(rows, columns=["participant_id", "line_no", "type", "match", "text"])
