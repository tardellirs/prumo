"""Capability: choose among many short labels (intent names, categories) by weighted word overlap between the
message and each label's words (IDF over the request's own label set, prefix match for inflections). An
out-of-scope / none option wins when no label shares a content word with the message."""
import math
import re
from collections import Counter

from prumo_rules.common import crit_text, from_scores, tokens, uniq
from prumo_rules.params import P as PRM

OOS = ("oos", "out of scope", "out-of-scope", "none of them", "none of these", "none", "other", "no match", "unknown")
OOS_SCORE = PRM["labels.oos_score"]
MSG_KEYS = ("text", "message", "utterance", "query", "request", "user_message", "input")


def _message(state, q):
    ins = crit_text(q.get("instructions"))
    m = re.search(r'"([^"]{3,})"', ins)
    if m:
        return m.group(1)
    if "\n" in ins:
        return ins.split("\n", 1)[1]
    if isinstance(state, dict):
        for k in MSG_KEYS:
            if isinstance(state.get(k), str):
                return state[k]
    if isinstance(state, str):
        return state
    return None


def _is_oos(v):
    t = v.strip().lower().replace("_", " ")
    return t in OOS or t.startswith(("out of scope", "out-of-scope", "none of")) or "none of the listed" in t or "no matching" in t


def _labelish(q):
    vals = [crit_text(v) for v in q["criteria"].values()]
    return len(vals) >= 8 and all(0 < len(v.replace("_", " ").split()) <= 12 and len(v) <= 80 for v in vals)


def _match(a, b):
    return a == b or (len(a) >= 4 and len(b) >= 4 and (a.startswith(b[:5]) or b.startswith(a[:5])))


def solve(state, questions):
    out = {}
    for k, q in questions.items():
        if q["type"] != "choice" or not _labelish(q):
            continue
        msg = _message(state, q)
        if not msg:
            continue
        mt = uniq(tokens(msg))
        labels = {o: uniq(tokens(crit_text(v).replace("_", " "))) for o, v in q["criteria"].items()}
        df = Counter(w for ws in labels.values() for w in ws)
        n = len(labels)
        oos = [o for o, v in q["criteria"].items() if _is_oos(crit_text(v))]
        sc = {}
        for o, ws in labels.items():
            if o in oos or not ws:
                sc[o] = 0.0
                continue
            hit = sum(math.log(1 + n / df[w]) for w in ws if any(_match(w, m) for m in mt))
            tot = sum(math.log(1 + n / df[w]) for w in ws)
            sc[o] = hit / tot * (1 + PRM["labels.hit_bonus"] * hit) if tot else 0.0
        best = max(sc.values())
        if best <= 0:
            if oos:
                out[k] = from_scores({o: (1.0 if o in oos else 0.0) for o in sc}, temp=PRM["labels.temperature_oos"])
            continue
        for o in oos:
            sc[o] = OOS_SCORE
        out[k] = from_scores(sc, temp=PRM["labels.temperature"])
    return out or None
