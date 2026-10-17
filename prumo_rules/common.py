"""Shared helpers: text of a state, tokens, numbers, probability shaping, deterministic tie-breaking.

Everything here is deterministic: no set iteration where order could change a float sum, no wall-clock reads.
"""
import hashlib
import json
import math
import re

# A short hand-written English stop-word list (no corpus statistics).
STOP = frozenset("""a an the of to in on for and or but is are was were be been being it its this that these those with as at by
from into about than then so such can could would should will shall may might must do does did done have has had i you he she
we they me my your our their his her them us what which who whom whose when where why how not no yes if else there here all
any each some more most other own same very just also only up down out over under again further once please""".split())


def text_of(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False, sort_keys=True)


def strings_in(x, depth=0):
    """All string leaves of a JSON value, in document order."""
    if depth > 6:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [s for v in x.values() for s in strings_in(v, depth + 1)]
    if isinstance(x, list):
        return [s for v in x for s in strings_in(v, depth + 1)]
    return []


def stem(w):
    for suf in ("ings", "ing", "ies", "ied", "ed", "es", "s", "ly"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf in ("ies", "ied") else "")
    return w


def tokens(s, keep_stop=False):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s or "")
    ws = re.findall(r"[a-z0-9]+", s.lower().replace("_", " "))
    return [stem(w) for w in ws if keep_stop or w not in STOP]


def uniq(xs):
    """Order-preserving unique (deterministic replacement for set())."""
    return list(dict.fromkeys(xs))


NUM_RE = re.compile(r"(?<![\w.])-?\$?\d[\d,]*(?:\.\d+)?%?")


def numbers(s):
    out = []
    for m in NUM_RE.finditer(s or ""):
        t = m.group(0).replace("$", "").replace(",", "")
        pct = t.endswith("%")
        t = t.rstrip("%")
        try:
            v = float(t)
        except ValueError:
            continue
        out.append(v / 100 if pct else v)
        if pct:
            out.append(float(t))
    return out


def to_number(s):
    if not isinstance(s, str):
        return None
    t = s.strip().replace(",", "").replace("$", "").rstrip("%").strip()
    try:
        return float(t)
    except ValueError:
        m = re.fullmatch(r"(-?\d+)\s*/\s*(\d+)", t)
        if m and int(m.group(2)):
            return int(m.group(1)) / int(m.group(2))
        return None


def hpick(keys, salt=""):
    """Deterministic pseudo-random pick (hash of the option set): ties, or nothing known (expected score = chance)."""
    keys = sorted(keys)
    h = int(hashlib.sha256((salt + "|" + "|".join(keys)).encode()).hexdigest(), 16)
    return keys[h % len(keys)]


def from_scores(scores, temp=1.0):
    """Softmax over solver scores (higher = better)."""
    m = max(scores.values())
    e = {k: math.exp((v - m) / temp) for k, v in scores.items()}
    z = sum(e.values())
    return {k: v / z for k, v in e.items()}


def peaked(keys, best, mass=0.9):
    """`mass` on the chosen set `best`, the rest spread uniformly."""
    keys = list(keys)
    bs = set(best)
    best = [k for k in keys if k in bs]
    if not best:
        return {k: 1 / len(keys) for k in keys}
    rest = [k for k in keys if k not in bs]
    if not rest:
        return {k: 1 / len(keys) for k in keys}
    return {k: (mass / len(best) if k in bs else (1 - mass) / len(rest)) for k in keys}


def crit_text(v):
    """Description of an option (criteria values may be strings, null or objects)."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False, sort_keys=True)


def opt_texts(q):
    return {k: crit_text(v).strip() for k, v in q["criteria"].items()}


YES_WORDS = ("yes", "true", "valid", "correct", "y")
NO_WORDS = ("no", "false", "invalid", "incorrect", "n")


def yes_no_keys(q):
    """(yes_key, no_key) when a two-option question is a yes/no (by key or by the start of the description)."""
    if q.get("type") != "choice" or len(q["criteria"]) != 2:
        return None
    yes, no = [], []
    for k, v in q["criteria"].items():
        t = crit_text(v).strip().lower().strip(" .!")
        kk = k.strip().lower()
        first = re.split(r"[\s,.:;!]+", t, maxsplit=1)[0] if t else ""
        if kk in YES_WORDS or first in YES_WORDS:
            yes.append(k)
        elif kk in NO_WORDS or first in NO_WORDS:
            no.append(k)
    return (yes[0], no[0]) if len(yes) == 1 and len(no) == 1 else None


def bool_answer(q, val, mass=0.95):
    """Probabilities for a yes/no style question (choice with yes/no-like options, or noul)."""
    if q.get("type") == "noul":
        return {"noul": mass if val else 1 - mass}
    yn = yes_no_keys(q)
    if not yn:
        return None
    return peaked(list(q["criteria"]), [yn[0] if val else yn[1]], mass)


def main_text(state, q=None):
    """The request's free text: string state, or its string fields, plus the question's instructions."""
    parts = []
    if isinstance(state, str):
        parts.append(state)
    elif isinstance(state, dict):
        parts.extend(v for v in state.values() if isinstance(v, str))
    if q is not None:
        ins = q.get("instructions")
        if isinstance(ins, str):
            parts.append(ins)
    return "\n".join(p for p in parts if p)
