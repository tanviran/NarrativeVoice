"""Find themes: k-means on embeddings, k chosen by silhouette, labelled with distinctive words."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from config import SEED, MIN_WORDS, K_RANGE

STOP = list(ENGLISH_STOP_WORDS | {"felt", "feel", "just", "really", "like", "got", "didn", "don", "wasn", "ve", "ll"})


def choose_k(X, k_range=K_RANGE, seed=SEED):
    scores = {}
    lo = min(k_range[0], max(2, len(X) // 4))      # small uploads need a smaller starting k
    hi = min(k_range[1], max(lo + 1, len(X) // 6), len(X) - 1)   # keep themes big enough to read
    for k in range(lo, hi + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        scores[k] = float(silhouette_score(X, labels))
    return max(scores, key=scores.get), scores


def stability(X, k, seeds=(1, 2, 3, 4, 5)):
    """Re-run k-means with different random starts. 1.0 = identical themes every time."""
    runs = [KMeans(n_clusters=k, n_init=3, random_state=s).fit_predict(X) for s in seeds]
    aris = [adjusted_rand_score(runs[i], runs[j]) for i in range(len(runs)) for j in range(i + 1, len(runs))]
    return float(np.mean(aris))


def keywords(texts_by_theme, n=5):
    ids = sorted(texts_by_theme)
    vec = TfidfVectorizer(stop_words=STOP, ngram_range=(1, 2), sublinear_tf=True)
    M = vec.fit_transform([" ".join(texts_by_theme[t]) for t in ids])
    terms = np.array(vec.get_feature_names_out())
    out = {}
    for row, t in enumerate(ids):
        chosen = []
        for term in terms[np.argsort(-M[row].toarray()[0])]:
            if any(term in c or c in term for c in chosen):
                continue                      # skip "worker" when "social worker" is chosen
            chosen.append(term)
            if len(chosen) == n:
                break
        out[t] = chosen
    return out


def find_themes(df, emb, k=None, min_words=MIN_WORDS, seed=SEED, k_range=K_RANGE):
    """Return (df with theme_id, info). Rows that are too short get theme_id -1."""
    eligible = (df.n_words >= min_words).to_numpy()
    X = emb[eligible]
    scores = {}
    if k is None:
        k, scores = choose_k(X, k_range, seed)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    ids = np.full(len(df), -1)
    ids[eligible] = km.labels_
    out = df.copy()
    out["theme_id"] = ids
    kw = keywords({t: out.loc[out.theme_id == t, "text"].tolist() for t in range(k)})
    info = {"k": int(k), "scores": scores, "stability": stability(X, k), "centers": km.cluster_centers_,
            "keywords": kw, "n_excluded": int((~eligible).sum()), "min_words": min_words}
    return out, info


def label(info, t):
    return f"T{t}: " + ", ".join(info["keywords"][t][:3])


def representative_rows(sub, emb, center, n=5):
    """Most typical answers for a theme, at most one per participant (so one voice cannot dominate)."""
    d = np.linalg.norm(emb[sub.index.to_numpy()] - center, axis=1)
    order = sub.assign(_d=d).sort_values("_d")
    return order.drop_duplicates("participant_id").head(n).drop(columns="_d")


def summarise(out, info):
    total = out.participant_id.nunique()
    rows = []
    for t in range(info["k"]):
        s = out[out.theme_id == t]
        rows.append({"theme_id": t, "label": label(info, t), "n_answers": len(s),
                     "n_participants": s.participant_id.nunique(),
                     "share_of_participants": round(s.participant_id.nunique() / total, 2)})
    return rows
