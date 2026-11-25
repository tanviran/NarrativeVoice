"""Run the heavy steps once and save the results. The dashboard reads what this writes.

    python src/pipeline.py            analyse the synthetic demo data
The upload page of the dashboard calls run() for your own transcripts."""
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from config import RESULTS, BACKEND, SEED, TRANSCRIPTS, PARTICIPANTS
from loader import load_segments
from embed import Embedder
from tone import score_tone


def run(folder=TRANSCRIPTS, meta=PARTICIPANTS, out=RESULTS, backend=BACKEND, log=print):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log("1/4 Loading transcripts")
    df = load_segments(folder, meta)
    log(f"    {len(df)} answers, {df.participant_id.nunique()} participants, {df.question_id.nunique()} questions")
    log(f"2/4 Embedding with '{backend}'")
    emb = Embedder(backend).fit_encode(df.text.tolist())
    log("3/4 Scoring tone")
    tone, method = score_tone(df.text.tolist())
    df["tone"], df["tone_score"] = [t[0] for t in tone], [round(t[1], 3) for t in tone]
    log("4/4 Building the 2D map")
    if len(df) >= 10:   # t-SNE needs enough points; tiny uploads use a plain projection instead
        xy = TSNE(n_components=2, perplexity=min(30, (len(df) - 1) / 3), init="pca", random_state=SEED).fit_transform(emb)
    else:
        xy = PCA(n_components=2, random_state=SEED).fit_transform(emb)
    df["x"], df["y"] = xy[:, 0], xy[:, 1]
    df.to_csv(out / "answers.csv", index=False)
    np.save(out / "embeddings.npy", emb)
    (out / "run_meta.json").write_text(json.dumps({"backend": backend, "tone_method": method, "n": len(df)}))
    log(f"Saved results to {out}. Tone method: {method}")
    return df


if __name__ == "__main__":
    run(backend=sys.argv[1] if len(sys.argv) > 1 else BACKEND)
