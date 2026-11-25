"""Checks that help a researcher question the AI instead of trusting it."""
import numpy as np


def voices_that_dont_fit(df, emb, info, pct=95):
    """Answers far from their theme centre: the untidy voices a summary tends to smooth over."""
    d = df[df.theme_id >= 0].copy()
    d["distance"] = np.linalg.norm(emb[d.index.to_numpy()] - info["centers"][d.theme_id.to_numpy()], axis=1)
    cutoff = d.groupby("theme_id").distance.transform(lambda s: np.percentile(s, pct))
    return d[d.distance > cutoff].sort_values("distance", ascending=False)


def contrast_cases(df, min_share=0.6):
    """Answers whose tone is the opposite of their theme's majority tone (for example a good placement
    story inside a mostly negative placements theme). Returns (cases, majority_tone_by_theme)."""
    d = df[(df.theme_id >= 0) & df.tone.isin(["positive", "negative"])]
    majority = {}
    for t, g in d.groupby("theme_id"):
        share = g.tone.value_counts(normalize=True)
        if share.iloc[0] >= min_share:
            majority[t] = share.index[0]
    out = d[d.apply(lambda r: r.theme_id in majority and r.tone != majority[r.theme_id], axis=1)]
    return out, majority
