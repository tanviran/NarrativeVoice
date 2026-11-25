"""Quality checks: evidence, voices that do not fit, and a privacy scan."""
import pandas as pd
import streamlit as st

from evidence import verify_frame, verify_quote
from privacy import scan
from quality import contrast_cases, voices_that_dont_fit

TEXT = {"text": st.column_config.TextColumn("Answer", width="large")}


def _evidence(ctx):
    ok = verify_frame(ctx.view, ctx.source_dir)
    st.metric("Answers found word for word in the source transcripts", f"{sum(ok)} of {len(ok)}")
    bad = ctx.view[[not v for v in ok]]
    if len(bad):
        st.error("These answers could not be verified:")
        st.dataframe(bad[["participant_id", "line_no", "text"]], hide_index=True, width="stretch")
    st.markdown("**Live test: does the checker catch a changed quote?**")
    if st.button("Alter a real quote and check it"):
        r = ctx.view.iloc[0]
        fake = r.text.replace(r.text.split()[-1], "invented")
        st.write(f"Original: _{r.text}_ -> accepted: **{verify_quote(r.participant_id, int(r.line_no), r.text, ctx.source_dir)}**")
        st.write(f"Altered: _{fake}_ -> accepted: **{verify_quote(r.participant_id, int(r.line_no), fake, ctx.source_dir)}**")
        st.caption("The altered version is rejected: a quote must appear exactly at that participant and line.")


def _outliers(ctx):
    st.caption("Summaries smooth over untidy voices. These answers sit furthest from their theme's centre, and these "
               "are answers whose tone is the opposite of their theme's majority. Read them before accepting a theme.")
    far = voices_that_dont_fit(ctx.full, ctx.emb, ctx.info)
    far = far[far.index.isin(ctx.view.index)]
    st.subheader(f"Voices that do not fit ({len(far)})")
    far = far.assign(theme=far.theme_id.map(ctx.labels))
    st.dataframe(far[["participant_id", "question_id", "theme", "tone", "text"]].head(40), hide_index=True,
                 width="stretch", column_config=TEXT)
    cases, majority = contrast_cases(ctx.view)
    st.subheader(f"Contrast cases ({len(cases)})")
    if len(cases):
        cases = cases.assign(theme=cases.theme_id.map(ctx.labels),
                             theme_majority=cases.theme_id.map(majority))
        st.dataframe(cases[["participant_id", "question_id", "theme", "theme_majority", "tone", "text"]].head(60),
                     hide_index=True, width="stretch", column_config=TEXT)
    else:
        st.info("No contrast cases in this view.")
    st.caption("Tone comes from a small model trained on film reviews, so a few of these will be tone errors "
               "and not true contrasts.")


def _privacy(ctx):
    hits = scan(ctx.full)
    st.metric("Possible identifiers found", len(hits))
    if hits.empty:
        st.success("No emails, phone numbers, postcodes, names or street addresses matched.")
    else:
        st.dataframe(hits, hide_index=True, width="stretch", column_config=TEXT)
        st.bar_chart(hits.type.value_counts())
    st.caption("This is a simple pattern scan, not anonymisation. It misses nicknames, school names and unusual "
               "spellings, so a person must still check transcripts before analysis.")


def render(ctx):
    st.title("Quality checks")
    st.caption("Ways to question the AI before trusting it.")
    if ctx.view.empty:
        st.warning("No answers match the current filters.")
        return
    t1, t2, t3 = st.tabs(["Evidence check", "Voices that do not fit", "Privacy scan"])
    with t1:
        _evidence(ctx)
    with t2:
        _outliers(ctx)
    with t3:
        _privacy(ctx)
