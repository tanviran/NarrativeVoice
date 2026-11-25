"""Compare methods against the planted ground truth, and write the numbers for the dashboard and docs.

    python src/evaluate.py                       minilm, mpnet, tfidf
    python src/evaluate.py minilm tfidf          choose which embedders to compare

Writes results/evaluation.json (read by the dashboard's Methods page) and docs/model_comparison.md.

CAUTION: the data is synthetic and template-built, so absolute scores look better than real interviews
would. Use the results to compare methods with each other, not to claim real-world accuracy."""
import datetime
import json
import sys
import time

import pandas as pd
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from config import GROUND_TRUTH, RESULTS, ROOT
from embed import MODELS, Embedder
from evidence import verify_frame
from loader import load_segments
from themes import find_themes
from tone import score_tone

PLANTED_K = 8   # the data was generated from 8 topics


def agreement(truth, labels):
    return round(adjusted_rand_score(truth, labels), 3), round(normalized_mutual_info_score(truth, labels), 3)


def cluster_with(method, X):
    """Return cluster labels for X using the named method."""
    if method == "agglomerative (k=8)":
        return AgglomerativeClustering(n_clusters=PLANTED_K, linkage="ward").fit_predict(X)
    if method == "HDBSCAN":
        return HDBSCAN(min_cluster_size=15).fit_predict(X)
    raise ValueError(method)


def evaluate_embedder(backend, df):
    """Return (embedder row, list of clustering rows) for one embedding method."""
    texts = df.text.tolist()
    t0 = time.perf_counter()
    emb = Embedder(backend).fit_encode(texts, use_cache=False)   # no cache, so the timing is honest
    embed_s = time.perf_counter() - t0
    elig = (df.n_words >= 5).to_numpy()
    X, topic = emb[elig], df.topic[elig].to_numpy()
    scorable = topic != "none"
    rows = []

    for label, k in (("k-means (auto k)", None), ("k-means (k=8)", PLANTED_K)):
        t0 = time.perf_counter()
        out, info = find_themes(df, emb, k=k)
        secs = time.perf_counter() - t0
        lab = out.theme_id[elig].to_numpy()
        ari, nmi = agreement(topic[scorable], lab[scorable])
        rows.append({"embedder": backend, "method": label, "k_found": info["k"], "ARI": ari, "NMI": nmi,
                     "assigned_pct": 100.0, "stability": round(info["stability"], 3), "seconds": round(secs, 2)})

    for label in ("agglomerative (k=8)", "HDBSCAN"):
        t0 = time.perf_counter()
        lab = cluster_with(label, X)
        secs = time.perf_counter() - t0
        keep = scorable & (lab != -1)                      # HDBSCAN leaves some answers unassigned
        ari, nmi = agreement(topic[keep], lab[keep]) if keep.sum() > 1 else (None, None)
        rows.append({"embedder": backend, "method": label, "k_found": int(len(set(lab)) - (1 if -1 in lab else 0)),
                     "ARI": ari, "NMI": nmi, "assigned_pct": round(100 * keep.sum() / scorable.sum(), 1),
                     "stability": None, "seconds": round(secs, 2)})
    emb_row = {"embedder": backend, "dimensions": int(emb.shape[1]), "embed_seconds": round(embed_s, 1),
               "model": MODELS.get(backend, "TF-IDF + SVD (no neural network)")}
    return emb_row, rows


def evaluate_tone(df):
    """Agreement with the planted positive/negative stance, on clear-cut answers only."""
    sub = df[df.stance.isin(["neg", "pos"])]
    rows = []
    for method in ("model", "lexicon"):
        t0 = time.perf_counter()
        tone, name = score_tone(sub.text.tolist(), method=method)
        secs = time.perf_counter() - t0
        pred = pd.Series([t[0] for t in tone], index=sub.index)
        mapped = pred.map({"negative": "neg", "positive": "pos"})
        rows.append({"method": name, "agrees_pct": round(100 * (mapped == sub.stance).mean(), 1),
                     "uncertain_pct": round(100 * (pred == "uncertain").mean(), 1), "seconds": round(secs, 2)})
    return rows


def md_table(rows, cols):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "\n".join("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |" for r in rows)
    return head + body


def main(backends):
    df = load_segments().merge(pd.read_csv(GROUND_TRUTH), on=["participant_id", "line_no"], how="left")
    emb_rows, clu_rows = [], []
    for b in backends:
        try:
            e, c = evaluate_embedder(b, df)
        except Exception as ex:
            print(f"skipping {b}: {type(ex).__name__}: {ex}")
            continue
        emb_rows.append(e)
        clu_rows += c
    ok = verify_frame(df)
    result = {"generated": datetime.date.today().isoformat(), "n_answers": len(df), "planted_topics": PLANTED_K,
              "embedders": emb_rows, "clustering": clu_rows, "tone": evaluate_tone(df),
              "verification": {"verified": int(sum(ok)), "total": len(ok)}}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "evaluation.json").write_text(json.dumps(result, indent=2))

    print("\nEMBEDDERS")
    print(pd.DataFrame(emb_rows).to_string(index=False))
    print("\nCLUSTERING (ARI/NMI: higher is better, 1.0 = perfect recovery of the planted topics)")
    print(pd.DataFrame(clu_rows).to_string(index=False))
    print("\nTONE")
    print(pd.DataFrame(result["tone"]).to_string(index=False))
    print(f"\nQUOTE VERIFICATION: {sum(ok)}/{len(ok)}")

    doc = ["# Model and method comparison", "",
           f"Generated {result['generated']} by `python src/evaluate.py` on {len(df)} synthetic answers "
           f"(planted topics: {PLANTED_K}). Scores compare methods on easy, template-built data. "
           "They are not claims of real-world accuracy.", "",
           "## Embedding models", "", md_table(emb_rows, ["embedder", "model", "dimensions", "embed_seconds"]), "",
           "## Clustering (recovery of planted topics)", "",
           md_table(clu_rows, ["embedder", "method", "k_found", "ARI", "NMI", "assigned_pct", "stability", "seconds"]), "",
           "## Tone", "", md_table(result["tone"], ["method", "agrees_pct", "uncertain_pct", "seconds"]), "",
           "## Quote verification", "", f"{sum(ok)} of {len(ok)} answers were found verbatim at their stated line.", ""]
    (ROOT / "docs" / "model_comparison.md").write_text("\n".join(doc))
    print("\nWrote results/evaluation.json and docs/model_comparison.md")


if __name__ == "__main__":
    main(sys.argv[1:] or ["minilm", "mpnet", "tfidf"])
