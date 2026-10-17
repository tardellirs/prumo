"""Lexical relevance computed only from the request itself: BM25 with IDF over the request's own candidates
(no corpus statistics). Iteration order is fixed so float sums are reproducible."""
import math
from collections import Counter

from prumo_rules.common import tokens, uniq
from prumo_rules.params import P as PRM


def bm25_scores(query, docs, k1=PRM["bm25.k1"], b=PRM["bm25.b"]):
    """query: str; docs: list[str] -> list[float]. IDF from these docs only."""
    q = uniq(tokens(query))
    dt = [tokens(d) for d in docs]
    n = len(dt)
    if not n or not q:
        return [0.0] * n
    avg = sum(len(d) for d in dt) / n or 1.0
    df = Counter(w for d in dt for w in uniq(d))
    out = []
    for d in dt:
        tf = Counter(d)
        s = 0.0
        for w in q:
            if w not in tf:
                continue
            idf = math.log(1 + (n - df[w] + PRM["bm25.idf_smoothing"]) / (df[w] + PRM["bm25.idf_smoothing"]))
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg))
        out.append(s)
    return out
