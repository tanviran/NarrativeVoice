"""Landing page: what this is, how to use it, and a quick health check."""
import streamlit as st

from ui.common import goto, verification_rate


def render(ctx):
    st.title("NarrativeVoice")
    st.markdown("#### From interview transcripts to evidence-grounded themes, with a researcher in control.")
    st.caption(f"Dataset: {ctx.name}")

    ok, total = verification_rate(str(ctx.results_dir), str(ctx.source_dir))
    a, b, c, d = st.columns(4)
    a.metric("Participants", ctx.full.participant_id.nunique())
    b.metric("Answers analysed", len(ctx.full))
    c.metric("Themes found", ctx.k)
    d.metric("Quotes verified in source", f"{ok / max(total, 1):.0%}")

    st.markdown("### How to work with it")
    steps = [("1. See the patterns", "Overview", "Which themes were raised by how many different people, and in which tone."),
             ("2. Read the evidence", "Explore a theme", "Real quotes, one per participant, each checked against the transcript."),
             ("3. Question the AI", "Quality checks", "Voices that do not fit, contrast cases, a privacy scan, and a live evidence check."),
             ("4. Decide", "Review and report", "Accept, rename or reject each theme. Every change is logged.")]
    for col, (title, page, text) in zip(st.columns(4), steps):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.markdown(f"<span class='nv-muted'>{text}</span>", unsafe_allow_html=True)
            st.button(f"Open {page}", key=f"go_{page}", on_click=goto, args=(page,))

    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown("**What the system does**")
        st.markdown("- Pairs every answer with the interviewer question it responded to\n"
                    "- Turns answers into numbers that capture meaning, then groups similar ones into themes\n"
                    "- Shows tone separately, because a theme can hold good and bad experiences\n"
                    "- Links every finding to a participant and line in the original transcript")
    with right.container(border=True):
        st.markdown("**What it does not do**")
        st.markdown("- Decide what an experience means: that is the researcher's job\n"
                    "- Prove cause: links between themes show co-occurrence only\n"
                    "- Replace reading the transcripts\n"
                    "- Guarantee accuracy on real data: the demo data is synthetic")
    st.button("Want your own transcripts? Upload them", on_click=goto, args=("Upload transcripts",))
