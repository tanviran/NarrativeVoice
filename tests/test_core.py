"""Fast tests for the parts that must never silently break. Run: python -m pytest tests -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from evidence import verify_quote
from insights import validate
from loader import is_probe, load_segments
from privacy import scan
from relations import theme_links
from quality import contrast_cases


def write(tmp_path, name, text):
    (tmp_path / name).write_text(text, encoding="utf-8")


def test_loader_keeps_main_question_through_probes(tmp_path):
    write(tmp_path, "P1.txt", "# header\nINTERVIEWER: How was school for you in care?\nPARTICIPANT: It was hard to keep up.\n"
                              "INTERVIEWER: Can you say more about that?\nPARTICIPANT: I missed many exams.\n")
    df = load_segments(tmp_path, tmp_path / "none.csv")
    assert list(df.is_followup) == [False, True]
    assert df.question.nunique() == 1 and df.line_no.tolist() == [3, 5]


def test_probe_detection():
    assert is_probe("What was that like for you?")
    assert not is_probe("Tell me about leaving care and what that was like for you.")


def test_verify_quote_rejects_altered_text_and_wrong_line(tmp_path):
    write(tmp_path, "P1.txt", "INTERVIEWER: Q?\nPARTICIPANT: I waited a year for counselling.\n")
    assert verify_quote("P1", 2, "I waited a year for counselling.", tmp_path)
    assert not verify_quote("P1", 2, "I waited a month for counselling.", tmp_path)
    assert not verify_quote("P1", 1, "Q?", tmp_path)          # interviewer line is not participant evidence
    assert not verify_quote("P1", 9, "anything", tmp_path)


def test_claim_checks_reject_unsupported_claims():
    ev = {"E1": {"participant": "P001", "line": 3, "text": "I waited almost a year for counselling."}}
    assert validate([{"text": "Some waited a long time.", "evidence": ["E1"]}], ev)[1] is None
    assert "number" in validate([{"text": "People waited 12 months.", "evidence": ["E1"]}], ev)[1]
    assert "unknown" in validate([{"text": "Waiting was long.", "evidence": ["E9"]}], ev)[1]
    assert "participant ID" in validate([{"text": "P001 waited.", "evidence": ["E1"]}], ev)[1]


def test_privacy_scan_finds_identifiers():
    df = pd.DataFrame({"participant_id": ["P1"] * 3, "line_no": [1, 2, 3],
                       "text": ["Email me at sam@example.com", "my name is Jordan and I live at 12 Park Road",
                                "I felt fine about school."]})
    kinds = set(scan(df).type)
    assert {"email address", "name given", "street address"} <= kinds
    assert 3 not in set(scan(df).line_no)


def test_theme_links_lift_matches_hand_calculation():
    # 10 people raise both themes, 10 only theme 0, 10 only theme 1 (people with no theme are not counted)
    rows = ([(p, t) for p in range(10) for t in (0, 1)] + [(p, 0) for p in range(10, 20)]
            + [(p, 1) for p in range(20, 30)])
    links, n = theme_links(pd.DataFrame(rows, columns=["participant_id", "theme_id"]))
    assert n == 30 and links.n_shared.iloc[0] == 10
    assert links.lift.iloc[0] == 0.75          # 10 * 30 / (20 * 20)


def test_contrast_cases_flags_minority_tone():
    df = pd.DataFrame({"theme_id": [0] * 5, "tone": ["negative"] * 4 + ["positive"]})
    cases, majority = contrast_cases(df)
    assert majority == {0: "negative"} and len(cases) == 1
