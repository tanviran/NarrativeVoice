"""Patterns across people: which themes co-occur, and whether groups differ."""
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from relations import participant_matrix, theme_links


def _links(ctx):
    links, n = theme_links(ctx.view)
    if links.empty:
        st.info("Not enough data in this view.")
        return
    a, b, c = st.columns(3)
    min_lift = a.slider("Minimum lift", 1.0, 3.0, 1.3, 0.1,
                        help="Lift 1.0 = themes meet by chance. 2.0 = twice as often as chance.")
    sig = b.toggle("Only statistically significant", True)
    min_shared = c.slider("Minimum shared participants", 1, 30, 5)
    keep = links[(links.lift >= min_lift) & (links.n_shared >= min_shared)]
    if sig:
        keep = keep[keep.p_adj < 0.05]
    pm = participant_matrix(ctx.view)
    g = nx.Graph()
    for t in pm.columns:
        g.add_node(t, size=int(pm[t].sum()))
    for r in keep.itertuples():
        g.add_edge(r.theme_a, r.theme_b, w=r.n_shared, lift=r.lift)
    pos = nx.circular_layout(g)
    cmap = ctx.colour_map()
    fig = go.Figure()
    for u, v, d in g.edges(data=True):
        fig.add_trace(go.Scatter(x=[pos[u][0], pos[v][0]], y=[pos[u][1], pos[v][1]], mode="lines", showlegend=False,
                                 line=dict(width=1 + d["w"] / 4, color="rgba(110,110,110,0.5)"), hoverinfo="text",
                                 text=f"{ctx.labels[u]}<br>{ctx.labels[v]}<br>shared {d['w']}, lift {d['lift']}"))
    fig.add_trace(go.Scatter(x=[pos[t][0] for t in g], y=[pos[t][1] for t in g], mode="markers+text", showlegend=False,
                             text=[f"T{t}" for t in g], textposition="middle center", hoverinfo="text",
                             hovertext=[f"{ctx.labels[t]}<br>{g.nodes[t]['size']} participants" for t in g],
                             marker=dict(size=[24 + g.nodes[t]["size"] * 0.6 for t in g],
                                         color=[cmap[ctx.labels[t]] for t in g], opacity=0.85)))
    fig.update_layout(template="plotly_white", height=520, xaxis_visible=False, yaxis_visible=False,
                      margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    st.markdown(f"**{len(keep)} links shown** out of {len(links)} pairs ({n} participants).")
    if len(keep):
        tbl = keep.copy()
        tbl["theme_a"], tbl["theme_b"] = tbl.theme_a.map(ctx.labels), tbl.theme_b.map(ctx.labels)
        st.dataframe(tbl.sort_values("lift", ascending=False)[["theme_a", "theme_b", "n_shared", "lift", "p_adj"]],
                     hide_index=True, width="stretch")
    st.caption("Lift corrects for big themes overlapping with everything. p is adjusted for testing many pairs. "
               "Co-occurrence is not cause: read the transcripts before saying one experience leads to another.")


def _groups(ctx):
    if not ctx.group_cols:
        st.info("This dataset has no participant information (region, age and so on) to compare.")
        return
    names = dict(ctx.group_cols)
    g = st.selectbox("Compare by", list(names), format_func=names.get)
    tone = st.selectbox("Count answers with tone", ["any", "negative", "positive"])
    tv = ctx.themed_view
    base = tv if tone == "any" else tv[tv.tone == tone]
    sizes = ctx.view.groupby(g).participant_id.nunique()
    rate = (base.groupby([g, "theme_id"]).participant_id.nunique().unstack(fill_value=0)
            .reindex(index=sizes.index, columns=range(ctx.k), fill_value=0).div(sizes, axis=0) * 100).round(0)
    rate.columns = [ctx.labels[c] for c in rate.columns]
    rate.index = [f"{i} (n={sizes[i]})" for i in rate.index]
    fig = px.imshow(rate, text_auto=True, aspect="auto", color_continuous_scale="Purples",
                    labels={"color": "% of group"}, height=130 + 55 * len(rate))
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, width="stretch")
    small = sizes[sizes < 10]
    if len(small):
        st.warning("Small groups (under 10 participants): " + ", ".join(f"{i} (n={n})" for i, n in small.items())
                   + ". Differences here are often noise.")
    st.caption("Each cell is the share of that group's participants who raised the theme. Treat gaps as prompts to "
               "investigate. No significance test is applied and many cells are compared at once.")


def render(ctx):
    st.title("Patterns")
    if ctx.view.empty:
        st.warning("No answers match the current filters.")
        return
    t1, t2 = st.tabs(["Which themes go together", "Do groups differ"])
    with t1:
        _links(ctx)
    with t2:
        _groups(ctx)
