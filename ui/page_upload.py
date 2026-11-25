"""Upload your own transcripts, check the format, and analyse them with the same pipeline."""
import re
import shutil

import streamlit as st

from config import DATA, RESULTS
from loader import LINE
from pipeline import run

UPLOAD_DIR, UPLOAD_RESULTS = DATA / "uploads", RESULTS / "uploads"
EXAMPLE = ("INTERVIEWER: How did being in care affect your education?\n"
           "PARTICIPANT: I changed schools three times and fell behind in most subjects.\n"
           "INTERVIEWER: Can you say more about that?\n"
           "PARTICIPANT: Nobody explained how to catch up, so I stopped trying for a while.\n")


def _check(name, text):
    """Return (n_participant_lines, n_interviewer_lines, problems) for one uploaded file."""
    parts = ints = 0
    problems = []
    for i, raw in enumerate(text.splitlines(), 1):
        if not raw.strip() or raw.startswith("#"):
            continue
        m = LINE.match(raw.strip())
        if not m:
            problems.append(f"line {i} does not start with INTERVIEWER: or PARTICIPANT:")
        elif m.group(1) == "PARTICIPANT":
            parts += 1
        else:
            ints += 1
    if parts == 0:
        problems.append("no PARTICIPANT lines found")
    return parts, ints, problems


def render(ctx):
    st.title("Upload transcripts")
    st.caption("Use synthetic or non-sensitive text only. Files stay on this computer.")
    with st.expander("Required format", expanded=True):
        st.markdown("One `.txt` file per participant. Each line starts with `INTERVIEWER:` or `PARTICIPANT:`. "
                    "The file name becomes the participant ID. Lines starting with `#` are ignored.")
        st.code(EXAMPLE)
        st.download_button("Download an example file", EXAMPLE, file_name="example_participant.txt")
    files = st.file_uploader("Transcript files", type="txt", accept_multiple_files=True)
    meta = st.file_uploader("Optional: participants.csv (participant_id, region, age_band, time_in_care, ethnic_group)",
                            type="csv")
    if not files:
        return

    report, valid = [], []
    for f in files:
        text = f.getvalue().decode("utf-8", errors="replace")
        parts, ints, problems = _check(f.name, text)
        report.append({"file": f.name, "participant answers": parts, "interviewer lines": ints,
                       "problems": "; ".join(problems[:3]) or "none"})
        if parts:
            valid.append((f.name, text))
    st.dataframe(report, hide_index=True, width="stretch")
    st.write(f"**{len(valid)} of {len(files)} files can be analysed.** Lines that do not match the format are skipped.")
    if len(valid) < 5:
        st.warning("Themes need more material. With fewer than about 5 participants the results will be unreliable.")
    if valid and st.button("Analyse these transcripts", type="primary"):
        if UPLOAD_DIR.exists():
            shutil.rmtree(UPLOAD_DIR)
        UPLOAD_DIR.mkdir(parents=True)
        for name, text in valid:
            stem = re.sub(r"[^A-Za-z0-9_-]", "_", name.rsplit(".", 1)[0])
            (UPLOAD_DIR / f"{stem}.txt").write_text(text, encoding="utf-8")
        meta_path = DATA / "uploads_participants.csv"
        if meta is not None:
            meta_path.write_bytes(meta.getvalue())
        elif meta_path.exists():
            meta_path.unlink()
        with st.status("Analysing...", expanded=True) as status:
            run(folder=UPLOAD_DIR, meta=meta_path, out=UPLOAD_RESULTS, log=st.write)
            status.update(label="Done", state="complete")
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Finished. Choose 'My uploaded transcripts' under Dataset in the sidebar.")
