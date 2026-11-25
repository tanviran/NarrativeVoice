"""Turn text into vectors (embeddings). Backends: minilm (default), mpnet, tfidf (offline baseline)."""
import hashlib
import numpy as np
from config import CACHE

MODELS = {"minilm": "all-MiniLM-L6-v2", "mpnet": "all-mpnet-base-v2"}


def _norm(Z):
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.maximum(n, 1e-9)


class Embedder:
    def __init__(self, backend="minilm"):
        if backend not in MODELS and backend != "tfidf":
            raise ValueError(f"Unknown embedder '{backend}'. Use minilm, mpnet or tfidf.")
        self.backend, self._model, self._tfidf, self._svd = backend, None, None, None

    def _st(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(MODELS[self.backend])
        return self._model

    def fit_encode(self, texts, use_cache=True):
        """Embed the whole corpus. Neural embeddings are cached on disk unless use_cache=False."""
        texts = list(texts)
        if self.backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            self._tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=2)
            X = self._tfidf.fit_transform(texts)
            self._svd = TruncatedSVD(n_components=max(2, min(100, X.shape[1] - 1, X.shape[0] - 1)), random_state=42)
            return _norm(self._svd.fit_transform(X))
        key = hashlib.md5((self.backend + "\n".join(texts)).encode()).hexdigest()[:16]
        path = CACHE / f"emb_{self.backend}_{key}.npy"
        if use_cache and path.exists():
            return np.load(path)
        Z = self._st().encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
        if use_cache:
            CACHE.mkdir(parents=True, exist_ok=True)
            np.save(path, Z)
        return Z

    def encode(self, texts):
        """Embed new text (for search) in the same space as the corpus."""
        texts = list(texts)
        if self.backend == "tfidf":
            return _norm(self._svd.transform(self._tfidf.transform(texts)))
        return self._st().encode(texts, normalize_embeddings=True)
