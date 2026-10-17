"""Capability: is a text supported by a given context? Lexical grounding: the numbers and capitalised names of the
text must occur in the context, and most of its content words must be covered (prefix match for inflections).
Used when the request pairs a claim with evidence, or a response with its source, and asks whether it is
supported / unsupported (hallucinated). Thresholds fixed a priori.
"""
import re

from prumo_rules.common import crit_text, peaked, tokens

COVER = 0.7  # chosen on HoVer + RAGTruth train splits (0.6-0.9 tried)
CLAIM_KEYS = ("claim", "statement", "hypothesis")
EVIDENCE_KEYS = ("evidence", "evidences", "context", "passages", "documents", "sources")
RESPONSE_KEYS = ("response", "answer", "output", "summary", "reply")
SOURCE_KEYS = ("prompt", "context", "source", "document", "reference", "passage")


def _flat(ev):
    if isinstance(ev, list):
        return " ".join((str(e.get("title", "")) + " " + str(e.get("text", ""))) if isinstance(e, dict) else str(e) for e in ev)
    return str(ev)


def _ctx_claim(state):
    if not isinstance(state, dict):
        return None, None
    for ck in CLAIM_KEYS:
        if isinstance(state.get(ck), str):
            for ek in EVIDENCE_KEYS:
                if state.get(ek):
                    return _flat(state[ek]), state[ck]
    for rk in RESPONSE_KEYS:
        if isinstance(state.get(rk), str):
            for sk in SOURCE_KEYS:
                if isinstance(state.get(sk), str) and sk != rk:
                    return state[sk], state[rk]
    return None, None


def supported(ctx, text):
    cl = ctx.lower()
    nums = re.findall(r"\d+(?:[.,]\d+)?", text)
    if any(n not in ctx and n.replace(",", "") not in ctx.replace(",", "") for n in nums):
        return False, 0.0
    names = re.findall(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b", text)
    if any(n.lower() not in cl for n in names):
        return False, 0.0
    ct = set(tokens(ctx))
    ctl = sorted(ct)
    tt = [w for w in tokens(text) if len(w) > 2]
    if not tt:
        return True, 1.0
    cov = sum(1 for w in tt if w in ct or (len(w) >= 6 and any(c.startswith(w[:5]) for c in ctl))) / len(tt)
    return cov >= COVER, cov


NEG_Q = re.compile(r"not supported|not backed|unsupported|hallucinat|invented|fabricat|not grounded|contradict|not faithful|unfaithful")


def solve(state, questions):
    ctx, text = _ctx_claim(state)
    if not ctx or not text:
        return None
    ok, cov = supported(ctx, text)
    out = {}
    for k, q in questions.items():
        ins = crit_text(q.get("instructions")).lower()
        if q["type"] == "noul":
            if NEG_Q.search(ins):
                out[k] = {"noul": 0.3 if ok else 0.7}
            elif re.search(r"\bsupported\b|\bfaithful\b|\bgrounded\b|\bbacked\b", ins):
                out[k] = {"noul": 0.7 if ok else 0.3}
            continue
        opts = {o: (o + " " + crit_text(v)).lower() for o, v in q["criteria"].items()}
        neg = [o for o, v in opts.items() if re.search(r"not[ _]supported|does not support|refute|unsupported|hallucinat|not[ _]enough|contradict", v)]
        pos = [o for o in opts if o not in neg]
        if len(neg) == 1 and len(pos) == 1:
            out[k] = peaked(list(opts), pos if ok else neg, 0.65)
    return out or None
