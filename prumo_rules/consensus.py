"""Capability: pick the candidate statement closest to a group's opinions (lexical centroid): mean BM25 similarity,
normalised per opinion, between each candidate and every participant's opinion (IDF over the candidates)."""
from prumo_rules.common import crit_text, from_scores
from prumo_rules.textmatch import bm25_scores
from prumo_rules.params import P as PRM

OPINION_KEYS = ("participant_opinions", "opinions", "participant_statements", "responses", "views")


def solve(state, questions):
    if not isinstance(state, dict):
        return None
    ops = next((state[k] for k in OPINION_KEYS if isinstance(state.get(k), list) and len(state[k]) >= 2), None)
    if not ops:
        return None
    out = {}
    for k, q in questions.items():
        if q["type"] != "choice" or len(q["criteria"]) < 2:
            continue
        keys = list(q["criteria"])
        docs = [crit_text(q["criteria"][o]) for o in keys]
        tot = {o: 0.0 for o in keys}
        for op in ops:
            sc = bm25_scores(op if isinstance(op, str) else crit_text(op), docs)
            m = max(sc) or 1.0
            for o, s in zip(keys, sc):
                tot[o] += s / m
        out[k] = from_scores({o: v / len(ops) for o, v in tot.items()}, temp=PRM["consensus.temperature"])
    return out or None
