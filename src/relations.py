"""Which themes tend to appear in the same people? Uses lift and Fisher's exact test, not raw counts."""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


def participant_matrix(df):
    d = df[df.theme_id >= 0]
    return pd.crosstab(d.participant_id, d.theme_id).gt(0)


def theme_links(df):
    """One row per pair of themes. lift > 1: appear together more than chance; < 1: less."""
    M = participant_matrix(df)
    N = len(M)
    rows = []
    for a, b in combinations(M.columns, 2):
        A, B = M[a], M[b]
        n_ab, n_a, n_b = int((A & B).sum()), int(A.sum()), int(B.sum())
        table = [[n_ab, n_a - n_ab], [n_b - n_ab, N - n_a - n_b + n_ab]]
        p = fisher_exact(table)[1]
        rows.append({"theme_a": a, "theme_b": b, "n_shared": n_ab, "n_a": n_a, "n_b": n_b,
                     "lift": round((n_ab * N) / (n_a * n_b), 2) if n_a * n_b else np.nan,
                     "jaccard": round(n_ab / (n_a + n_b - n_ab), 2) if (n_a + n_b - n_ab) else 0.0,
                     "p": p})
    links = pd.DataFrame(rows)
    if len(links):
        links["p_adj"] = (links.p * len(links)).clip(upper=1.0)   # Bonferroni: many pairs tested at once
    return links, N
