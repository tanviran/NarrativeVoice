"""Tone (positive / negative / uncertain) for each answer.

Tone is NOT the same as theme: embeddings group by topic, so one theme can hold both
good and bad experiences. Tone lets a researcher see that split.
Primary: a small sentiment model. Fallback: a tiny word list (flagged as such)."""
NEG = {"waited", "never", "nobody", "alone", "anxious", "lost", "missed", "cancelled", "fell", "behind", "stopped",
       "hardly", "overloaded", "cliff", "debt", "grieve", "punishment", "difficult", "passed", "cancelled", "no",
       "dropped", "kept", "changing", "paid", "manage", "ended", "left", "rush", "unpacking", "miles"}
POS = {"helped", "trust", "stable", "steady", "believed", "improved", "quickly", "proud", "close", "relief", "listened",
       "difference", "prepared", "seriously", "honest", "heard", "stayed", "confidence", "meant", "safely", "regularly"}


def _lexicon(texts):
    out = []
    for t in texts:
        w = set(t.lower().replace(",", " ").replace(".", " ").split())
        n, p = len(w & NEG), len(w & POS)
        out.append(("negative", 0.6) if n > p else ("positive", 0.6) if p > n else ("uncertain", 0.5))
    return out


def score_tone(texts, method="auto"):
    """Return (list of (label, score), method_name). method: auto | model | lexicon."""
    texts = list(texts)
    if method == "lexicon":
        return _lexicon(texts), "lexicon"
    try:
        from transformers import pipeline
        clf = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                       truncation=True)
        res = []
        for r in clf(texts, batch_size=32):
            label = r["label"].lower()
            res.append((label if r["score"] >= 0.8 else "uncertain", float(r["score"])))
        return res, "distilbert-sst2"
    except Exception as e:                      # offline, missing model, etc.
        print(f"  (sentiment model unavailable: {type(e).__name__}; using word-list fallback)")
        return _lexicon(texts), "lexicon-fallback"
