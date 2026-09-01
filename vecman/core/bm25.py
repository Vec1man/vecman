"""Minimal BM25 (Okapi) implementation for hybrid keyword + dense search.

Dependency-free; tokenization is a lowercase word-character split, which is
adequate for fusing keyword evidence with dense retrieval via RRF. Swap in
a proper analyzer if your corpus needs stemming or CJK segmentation.
"""

import math
import re
from collections import Counter
from typing import Dict, List

import numpy as np

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: List[Counter] = []
        self.doc_lens: List[int] = []
        term_doc_count: Counter = Counter()
        for doc in docs:
            tokens = tokenize(doc or "")
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            self.doc_lens.append(len(tokens))
            term_doc_count.update(freqs.keys())
        self.n_docs = len(docs)
        self.avg_len = (sum(self.doc_lens) / self.n_docs) if self.n_docs else 0.0
        self.idf: Dict[str, float] = {
            term: math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for term, df in term_doc_count.items()
        }

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of the query against every document."""
        out = np.zeros(self.n_docs, dtype=np.float32)
        terms = tokenize(query)
        if not terms or self.n_docs == 0:
            return out
        for i, (freqs, doc_len) in enumerate(zip(self.doc_freqs, self.doc_lens)):
            if doc_len == 0:
                continue
            norm = self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_len, 1e-9))
            score = 0.0
            for term in terms:
                tf = freqs.get(term)
                if not tf:
                    continue
                score += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / (tf + norm)
            out[i] = score
        return out
