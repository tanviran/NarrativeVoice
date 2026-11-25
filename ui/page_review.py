"""Researcher review: accept, rename or reject each theme, keep an audit trail, export a report."""
import datetime
import json

import streamlit as st

from evidence import verify_frame
from themes import representative_rows

STATUSES = ["pending_review", "accepted", "needs_edit", "rejected"]


def _paths(ctx):
    return ctx.results_dir / "reviews.json", ctx.results_dir / "audit_log.jsonl"


def _log(path, reviewer, ctx, theme, field, old, new):
    entry = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "reviewer": reviewer, "k": ctx.k,
             "theme": theme, "field": field, "old": old, "new": new}
    with open(path, "a") as fh:
        fh.write(json.dumps(entry) + "\n")


def _report(ctx, reviews):
    full = ctx.full
    lines = ["# Reviewed findings", "", f"{full.participant_id.nunique()} participants. "
             "Only researcher-accepted themes are included. Draft AI output until accepted.", ""]
    for t in range(ctx.k):
        r = reviews.get(f"k{ctx.k}_t{t}")
        if not r or r["status"] != "accepted":
            continue
        s = full[full.theme_id == t]
        tones = ", ".join(f"{k} {v}" for k, v in s.tone.value_counts().items())
        lines += [f"## {r['label'] or ctx.labels[t]}",
                  f"- Raised by {s.participant_id.nunique()} participants "
                  f"({s.participant_id.nunique() / full.participant_id.nunique():.0%} of all). Tone of answers: {tones}."]
        if r["notes"]:
            lines.append(f"- Researcher notes: {r['notes']}")
        lines.append("- Supporting quotes (one per participant, checked against the transcript):")
        top = representative_rows(s, ctx.emb, ctx.info["centers"][t], n=4)
        for q, ok in zip(top.itertuples(), verify_frame(top, ctx.source_dir)):
            lines.append(f"  - \"{q.text}\" ({q.participant_id}, line {q.line_no}, {q.question_id})"
                         + ("" if ok else " NOT VERIFIED"))
        lines.append("")
    return "\n".join(lines)


def render(ctx):
    st.title("Review and report")
    st.caption("The AI proposes themes. You decide what they mean. Reviews belong to the current theme set (same k).")
    rev_path, log_path = _paths(ctx)
    reviews = json.loads(rev_path.read_text()) if rev_path.exists() else {}
    done = sum(reviews.get(f"k{ctx.k}_t{t}", {}).get("status", "pending_review") != "pending_review" for t in range(ctx.k))
    st.progress(done / ctx.k, text=f"{done} of {ctx.k} themes reviewed")
    t1, t2, t3 = st.tabs(["Review themes", "Report", "Audit log"])

    with t1:
        reviewer = st.text_input("Your name (needed to save)")
        t = st.selectbox("Theme", range(ctx.k), format_func=lambda i: ctx.labels[i])
        key = f"k{ctx.k}_t{t}"
        cur = reviews.get(key, {"status": "pending_review", "label": "", "notes": ""})
        sub = ctx.full[ctx.full.theme_id == t]
        st.markdown(f"**{sub.participant_id.nunique()} participants, {len(sub)} answers.** Distinctive words: "
                    + ", ".join(ctx.info["keywords"][t]))
        for q in representative_rows(sub, ctx.emb, ctx.info["centers"][t], n=3).itertuples():
            st.markdown(f"<div class='nv-quote'>{q.text}<br><span class='nv-muted'>{q.participant_id}, line "
                        f"{q.line_no}, {q.tone}</span></div>", unsafe_allow_html=True)
        with st.form(f"form_{key}"):
            status = st.radio("Decision", STATUSES, index=STATUSES.index(cur["status"]), horizontal=True)
            label = st.text_input("Your label for this theme", value=cur["label"])
            notes = st.text_area("Notes (mixes two topics, contrast cases, needs context...)", value=cur["notes"])
            if st.form_submit_button("Save"):
                if not reviewer.strip():
                    st.warning("Enter your name first.")
                else:
                    for field, old, new in (("status", cur["status"], status), ("label", cur["label"], label),
                                            ("notes", cur["notes"], notes)):
                        if old != new:
                            _log(log_path, reviewer.strip(), ctx, t, field, old, new)
                    reviews[key] = {"status": status, "label": label, "notes": notes}
                    rev_path.write_text(json.dumps(reviews, indent=2))
                    st.success("Saved and logged.")
                    st.rerun()

    with t2:
        report = _report(ctx, reviews)
        st.markdown(report)
        st.download_button("Download report (Markdown)", report, file_name="reviewed_findings.md")

    with t3:
        if log_path.exists():
            st.code(log_path.read_text())
        else:
            st.info("No changes recorded yet.")
