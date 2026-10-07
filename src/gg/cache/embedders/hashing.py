import hashlib
import math
import re
import unicodedata
from collections.abc import Sequence

_WORD_RE = re.compile(r"\w+")


class HashingEmbedder:
    """deterministic bag of words and character trigrams hashed into signed buckets; tests and dev only"""

    def __init__(self, dim: int = 256) -> None:
        if dim < 8:
            raise ValueError("hashing embedder needs dim >= 8")
        self._dim = dim

    @property
    def name(self) -> str:
        return f"hashing-{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def _features(self, text: str) -> list[str]:
        words = _WORD_RE.findall(unicodedata.normalize("NFKC", text).lower())
        grams = [f"#{w[i : i + 3]}" for w in words for i in range(max(1, len(w) - 2))]
        return words + grams

    def embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self._dim
            vector[bucket] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in vector))
        return [x / norm for x in vector] if norm else vector

    async def embed(self, texts: Sequence[str], /) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]
