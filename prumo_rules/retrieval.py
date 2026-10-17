"""Capability: rank candidates against a query with BM25 (IDF over the request's own candidates).

- yes/no per candidate ("is this candidate relevant to the query?"): the candidate is given with the question;
  P(yes) is a monotone squash of its BM25 score, so the probabilities order the candidates by BM25.
- graded relevance of one item to a search query (options described as exact / partial / substitute /
  irrelevant ...): share of the query's terms that the item's title/name covers.
"""
import math

from prumo_rules.common import crit_text, peaked, tokens, uniq, yes_no_keys
from prumo_rules.textmatch import bm25_scores

QUERY_KEYS = ("query", "search_query", "search", "question", "claim", "information_need")


def _cand(q):
    ins = q.get("instructions")
    if isinstance(ins, dict):
        c = ins.get("candidate") or ins.get("document") or ins.get("passage")
        if isinstance(c, (str, dict)):
            return c if isinstance(c, str) else crit_text(c)
    return None


def _query(state):
    if isinstance(state, dict):
        for k in QUERY_KEYS:
            if isinstance(state.get(k), str) and state[k].strip():
                return state[k]
    return None


def _item(state):
    for k in ("product", "item", "listing", "result"):
        if isinstance(state.get(k), dict):
            return state[k]
    return None


def _grade(state, q):
    if not isinstance(state, dict):
        return None
    item = _item(state)
    query = _query(state)
    if item is None or query is None:
        return None
    title = " ".join(str(item.get(k, "")) for k in ("title", "name", "product_name", "brand", "category"))
    qt = uniq(tokens(query))
    if not qt:
        return None
    tt = set(tokens(title + " " + str(item.get("bullet_point", "") or item.get("description", ""))[:300]))
    cov = sum(1 for w in qt if any(w == t or (len(w) >= 4 and t.startswith(w[:4])) for t in sorted(tt))) / len(qt)
    opts = {k: crit_text(v).lower() for k, v in q["criteria"].items()}
    exact = [k for k, v in opts.items() if v.startswith("exact")]
    irr = [k for k, v in opts.items() if v.startswith(("irrelevant", "not relevant"))]
    if len(exact) != 1 or len(irr) != 1:
        return None
    if cov >= 0.5:
        return peaked(list(opts), exact, 0.6)
    if cov == 0:
        return peaked(list(opts), irr, 0.6)
    return peaked(list(opts), exact, 0.45)


def solve(state, questions):
    out = {}
    query = _query(state)
    cands = {}
    for k, q in questions.items():
        if q["type"] != "choice":
            continue
        yn = yes_no_keys(q)
        c = _cand(q)
        if yn and c and query:
            cands[k] = (c, yn)
        else:
            g = _grade(state, q)
            if g:
                out[k] = g
    if cands:
        keys = list(cands)
        sc = bm25_scores(query, [cands[k][0] for k in keys])
        for k, s in zip(keys, sc):
            p = 1 - math.exp(-s / 8.0)
            p = min(0.98, max(0.02, 0.02 + 0.96 * p))
            yes, no = cands[k][1]
            out[k] = {yes: p, no: 1 - p}
    return out or None
