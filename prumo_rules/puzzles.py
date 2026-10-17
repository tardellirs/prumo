"""Capabilities: small exact puzzles stated in text. Each one parses the WHOLE puzzle from its structure and
computes the answer; if any sentence cannot be read, it passes (no guess).

- ordering constraints: a list of N entities ("...: A, B, C and D.") and sentences that each place one entity
  (leftmost, third from the right, second-cheapest, finished last, ...) or compare two (left of, older than,
  cheaper than, finished above, ...) on one linear axis -> all permutations are checked; the unique solution
  decides which option (a positional statement) is true.
- swap tracking: "<A> V <x>, <B> V <y>, and <C> V <z>" (the same verb phrase V in every clause) followed by
  pairwise swaps/trades/switches -> who ends with what.
- movement: steps forward/backward/left/right or north/south/east/west and turns -> back at the start?
- free interval: busy time ranges "from T1 to T2" and option ranges -> the unique option that lies within the
  times mentioned and overlaps no busy range.
- SVG path: vertices of a <path d="..."> -> polygon/line/sector by geometry.
- brackets: the option that closes an open bracket sequence, or whether a sequence is balanced.
- sorting: the option that is the alphabetical order of a word list.
"""
import itertools
import math
import re

from prumo_rules.common import crit_text, main_text, opt_texts, peaked, yes_no_keys

ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
       "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6, "7th": 7, "8th": 8}
NUMW = {"zero": 0, "no": 0, "one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10}


def _pick(q, pred, mass=0.95):
    o = opt_texts(q)
    hits = [k for k, v in o.items() if pred(v)]
    if not hits or len(hits) == len(o):
        return None
    return peaked(list(o), hits, mass)


def _yn(q, val, mass=0.95):
    yn = yes_no_keys(q)
    return peaked(list(q["criteria"]), [yn[0] if val else yn[1]], mass) if yn else None


# ------------------------------------------------------------------------------------ ordering constraints --
# one linear axis per puzzle: index 0 is the "low" end
LOW_END = {"left": "lr", "leftmost": "lr", "oldest": "age", "cheapest": "price", "least expensive": "price", "first": "rank"}
HIGH_END = {"right": "lr", "rightmost": "lr", "newest": "age", "youngest": "age", "most expensive": "price", "priciest": "price",
            "last": "rank"}
LT = ("to the left of", "left of", "older than", "cheaper than", "less expensive than", "less costly than", "finished above",
      "finished before", "finished ahead of", "came before", "came in ahead of", "ahead of", "above", "before")
GT = ("to the right of", "right of", "newer than", "younger than", "more expensive than", "pricier than", "costlier than",
      "finished below", "finished after", "finished behind", "came after", "behind", "below", "after")


def _entities(text):
    """The entity list: the first 'something: X, Y, ... and Z.' with 3-9 short items, or None."""
    for m in re.finditer(r":\s*([^:.\n]+?)\.(?:\s|$)", text):
        items = [re.sub(r"^(?:a|an|the)\s+", "", x.strip(), flags=re.I) for x in re.split(r",\s*(?:and\s+)?|\s+and\s+", m.group(1))]
        items = [x for x in items if x]
        if 3 <= len(items) <= 9 and all(1 <= len(x.split()) <= 4 for x in items) and len(set(x.lower() for x in items)) == len(items):
            return [x.lower() for x in items], m.end()
    return None, None


def _position(t, n):
    """Index asserted by a one-entity sentence, or None."""
    m = re.search(r"\b(\w+)[- ](?:from the |from )?(left|right)\b", t)
    if m and m.group(1) in ORD:
        k = ORD[m.group(1)]
        return k - 1 if m.group(2) == "left" else n - k
    m = re.search(r"\b(\w+)[- ]to[- ]last\b", t)
    if m and m.group(1) in ORD:
        return n - ORD[m.group(1)]
    if re.search(r"\bsecond[- ]to[- ]last\b|\bpenultimate\b", t):
        return n - 2
    m = re.search(r"\b(\w+)[- ](cheapest|least expensive|oldest|newest|youngest|most expensive|priciest)\b", t)
    if m and m.group(1) in ORD:
        k = ORD[m.group(1)]
        return k - 1 if m.group(2) in ("cheapest", "least expensive", "oldest") else n - k
    for w in ("leftmost", "cheapest", "least expensive", "oldest"):
        if re.search(r"\b" + w + r"\b", t):
            return 0
    for w in ("rightmost", "most expensive", "priciest", "newest", "youngest"):
        if re.search(r"\b" + w + r"\b", t):
            return n - 1
    m = re.search(r"\b(?:finished|came|placed|ranked|is|was)\s+(?:in\s+)?(\w+)(?:\s+place)?\b", t)
    if m:
        if m.group(1) == "last":
            return n - 1
        if m.group(1) in ORD:
            return ORD[m.group(1)] - 1
    return None


def _found(t, objs):
    hits = []
    for o in objs:
        for mm in re.finditer(r"(?<![\w-])" + re.escape(o) + r"(?![\w-])", t):
            hits.append((mm.start(), mm.end(), o))
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    out, last_end = [], -1
    for s, e, o in hits:
        if s >= last_end:
            out.append((s, e, o))
            last_end = e
    return out


def _constraint(sent, objs, n):
    t = sent.lower().strip().rstrip(".")
    neg = bool(re.search(r"\b(?:not|isn't|wasn't|didn't|doesn't)\b", t))
    f = _found(t, objs)
    names = list(dict.fromkeys(o for _, _, o in f))
    if len(names) == 2:
        a, b = names
        mid = t[f[0][1]: [x for x in f if x[2] == b][0][0]]
        if any(r in mid for r in GT):
            c = ("gt", a, b)
        elif any(r in mid for r in LT):
            c = ("lt", a, b)
        else:
            return None
        return ("not",) + c if neg else c
    if len(names) == 1:
        rest = t[: f[0][0]] + " " + t[f[0][1]:]
        idx = _position(rest, n)
        if idx is None or not 0 <= idx < n:
            return None
        return ("notat", names[0], idx) if neg else ("at", names[0], idx)
    return None


def _holds(c, perm):
    if c[0] == "not":
        return not _holds(c[1:], perm)
    if c[0] == "notat":
        return perm[c[1]] != c[2]
    if c[0] == "lt":
        return perm[c[1]] < perm[c[2]]
    if c[0] == "gt":
        return perm[c[1]] > perm[c[2]]
    return perm[c[1]] == c[2]


def ordering(state, q):
    text = main_text(state, q)
    objs, end = _entities(text)
    if not objs:
        return None
    n = len(objs)
    if n > 7:
        return None
    body = re.split(r"\n\s*(?:options?|choices?|answer choices?)\s*:", text[end:], flags=re.I)[0]
    cons = []
    for sent in re.split(r"(?<=[.!?])\s+|\n+", body):
        s = sent.strip()
        if not s or s.endswith("?") or re.match(r"^\(?[A-Za-z]\)\s", s):
            continue
        c = _constraint(s, objs, n)
        if c is None:
            if _found(s.lower(), objs):
                return None  # a sentence about the entities that we cannot read
            continue
        cons.append(c)
    if len(cons) < 2:
        return None
    sols = []
    for p in itertools.permutations(range(n)):
        perm = dict(zip(objs, p))
        if all(_holds(c, perm) for c in cons):
            sols.append(perm)
            if len(sols) > 1:
                return None
    if len(sols) != 1:
        return None
    perm = sols[0]
    o = opt_texts(q)
    hits = []
    for k, v in o.items():
        c = _constraint(v, objs, n)
        if c is None:
            return None
        if _holds(c, perm):
            hits.append(k)
    if len(hits) != 1:
        return None
    return peaked(list(o), hits, 0.95)


# ------------------------------------------------------------------------------------------- swap tracking --
SWAP = r"(?:swap|swaps|trade|trades|switch|switches|exchange|exchanges)"


def swaps(state, q):
    text = main_text(state, q)
    if not re.search(r"\b" + SWAP + r"\b", text):
        return None
    init = None
    for m in re.finditer(r"([A-Z][a-z]+) ([a-z][a-z ]{0,30}?) ((?:a |an |the )?[^,.]{1,60}?),\s*([A-Z][a-z]+) \2 ", text):
        verb = m.group(2)
        span_start = m.start()
        sent_end = text.find(".", span_start)
        clause = text[span_start: sent_end if sent_end > 0 else len(text)]
        parts = re.split(r",\s*(?:and\s+)?|\s+and\s+(?=[A-Z])", clause)
        owner = {}
        for p in parts:
            mm = re.fullmatch(r"\s*([A-Z][a-z]+) " + re.escape(verb) + r" (?:a |an |the )?(.+?)\s*", p)
            if not mm:
                owner = None
                break
            owner[mm.group(1)] = mm.group(2).strip()
        if owner and len(owner) >= 2:
            init = (owner, verb, sent_end)
            break
    if not init:
        return None
    owner, verb, pos = init
    rest = text[pos:]
    for a, b in re.findall(r"([A-Z][a-z]+) and ([A-Z][a-z]+) " + SWAP, rest):
        if a not in owner or b not in owner:
            return None
        owner[a], owner[b] = owner[b], owner[a]
    finals = re.findall(r"([A-Z][a-z]+) (?:" + re.escape(verb) + r"|ends up with|has|is holding|is left with)\b", rest)
    finals = [f for f in finals if f in owner]
    if not finals:
        return None
    item = owner[finals[-1]].lower()
    norm = lambda v: re.sub(r"^(?:a|an|the)\s+", "", v.lower().strip(" ."))
    return _pick(q, lambda v: norm(v) == item or item.endswith(norm(v)) or norm(v).endswith(item))


# ------------------------------------------------------------------------------------------------ movement --
DIRS = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0), "up": (0, 1), "down": (0, -1)}
VEC = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def movement(state, q):
    text = main_text(state, q)
    if not re.search(r"\b(?:start(?:ing)? (?:point|position|place)|origin|where (?:you|they|he|she|we|it) started)\b", text, re.I):
        return None
    if not re.search(r"\b(?:steps?|paces?|units?|meters?|metres?|blocks?)\b", text, re.I):
        return None
    body = re.split(r"\n\s*options?\s*:", text, flags=re.I)[0]
    x = y = d = 0
    moved = 0
    for sent in re.split(r"(?<=[.;?!])\s+|\n+", body):
        s = sent.strip().lower().rstrip(".;")
        if not s or s.endswith("?"):
            continue
        for part in re.split(r",\s*(?:then\s+)?|\s+then\s+|\s+and\s+", s):
            p = part.strip()
            if not p or re.search(r"\b(?:start(?:ing)?|origin|instructions?|always face|answer|options?)\b", p):
                continue
            if re.fullmatch(r"(?:turn|rotate) (?:to the )?left", p):
                d = (d + 3) % 4; continue
            if re.fullmatch(r"(?:turn|rotate) (?:to the )?right", p):
                d = (d + 1) % 4; continue
            if re.fullmatch(r"(?:turn|rotate) around|about[- ]face", p):
                d = (d + 2) % 4; continue
            m = re.fullmatch(r"face (north|south|east|west)", p)
            if m:
                d = VEC.index(DIRS[m.group(1)]); continue
            m = re.fullmatch(r"(?:take|move|walk|go|step|run)(?: forward| backward| back| left| right)? (\d+|" + "|".join(NUMW) + r") (?:steps?|paces?|units?|meters?|metres?|blocks?)"
                             r"(?: (forward|forwards|ahead|backward|backwards|back|left|right|north|south|east|west|up|down))?", p)
            if not m:
                return None
            k = int(m.group(1)) if m.group(1).isdigit() else NUMW[m.group(1)]
            word = m.group(2) or (re.search(r"^(?:take|move|walk|go|step|run) (forward|backward|back|left|right)\b", p) or [None, None])[1]
            if word in DIRS:
                dx, dy = DIRS[word]
            else:
                rel = {None: 0, "forward": 0, "forwards": 0, "ahead": 0, "right": 1, "backward": 2, "backwards": 2, "back": 2, "left": 3}[word]
                dx, dy = VEC[(d + rel) % 4]
            x += k * dx
            y += k * dy
            moved += 1
    if moved < 1:
        return None
    return _yn(q, x == 0 and y == 0)


# ------------------------------------------------------------------------------------------- free interval --
TIME = r"(\d{1,2}(?::\d\d)?\s*(?:am|pm|a\.m\.|p\.m\.)|\d{1,2}:\d\d)"


def _hour(t):
    t = t.strip().lower().replace(".", "")
    m = re.fullmatch(r"(\d{1,2})(?::(\d\d))?\s*(am|pm)?", t)
    if not m:
        return None
    h = int(m.group(1))
    if m.group(3):
        h = h % 12 + (12 if m.group(3) == "pm" else 0)
    return h + (int(m.group(2)) / 60 if m.group(2) else 0)


def intervals(state, q):
    text = main_text(state, q)
    if not re.search(r"\b(?:between what times|what times? could|when could|which (?:time|period) could)\b", text, re.I):
        return None
    o = opt_texts(q)
    rng = {}
    for k, v in o.items():
        m = re.fullmatch(r"(?:from\s+)?" + TIME + r"\s*(?:to|-|–|until)\s*" + TIME, v.strip(), re.I)
        if not m:
            return None
        rng[k] = (_hour(m.group(1)), _hour(m.group(2)))
    body = re.split(r"\n\s*options?\s*:", text, flags=re.I)[0]
    busy = [(_hour(a), _hour(b)) for a, b in re.findall(r"from " + TIME + r" (?:to|until) " + TIME, body, re.I)]
    times = [_hour(t) for t in re.findall(TIME, body, re.I)]
    if len(busy) < 2 or any(a is None or b is None for a, b in busy) or any(t is None for t in times):
        return None
    lo, hi = min(times), max(times)
    ok = [k for k, (a, b) in rng.items() if a is not None and b is not None and lo <= a < b <= hi and all(b <= x or a >= y for x, y in busy)]
    if len(ok) != 1:
        return None
    return peaked(list(o), ok, 0.95)


# ------------------------------------------------------------------------------------------------ SVG path --
def shapes(state, q):
    text = main_text(state, q)
    m = re.search(r'<path d="([^"]+)"', text)
    if not m:
        return None
    d = m.group(1)
    o = opt_texts(q)
    low = {k: v.lower().strip(" .") for k, v in o.items()}
    pick = lambda names: (peaked(list(o), [k for k, v in low.items() if v in names], 0.95)
                          if 0 < sum(v in names for v in low.values()) < len(o) else None)
    if re.search(r"(?:^|\s)A\s", " " + d):
        return pick(("sector",) if re.search(r"\bL\b", d) else ("ellipse", "circle"))
    pts = []
    for cmd, xs, ys in re.findall(r"([ML])\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", d):
        p = (round(float(xs), 2), round(float(ys), 2))
        if p not in pts:
            pts.append(p)
    if len(pts) == 2:
        return pick(("line",))
    names = {3: "triangle", 5: "pentagon", 6: "hexagon", 7: "heptagon", 8: "octagon"}
    if len(pts) in names:
        return pick((names[len(pts)],))
    if len(pts) == 4:
        a, b, c, e = pts
        dist = lambda p, r: math.hypot(p[0] - r[0], p[1] - r[1])
        sides = [dist(a, b), dist(b, c), dist(c, e), dist(e, a)]
        dot = lambda p, r, t: (p[0] - r[0]) * (t[0] - r[0]) + (p[1] - r[1]) * (t[1] - r[1])
        if all(abs(dot(x, y, z)) < 1e-2 * max(sides) ** 2 for x, y, z in ((e, a, b), (a, b, c), (b, c, e))):
            sq = abs(sides[0] - sides[1]) < 1e-2 * max(sides)
            return pick(("square", "rectangle") if sq else ("rectangle",))
        tol = 0.02 * max(sides)
        if (abs(sides[0] - sides[1]) < tol and abs(sides[2] - sides[3]) < tol) or (abs(sides[1] - sides[2]) < tol and abs(sides[3] - sides[0]) < tol):
            return pick(("kite",))
        return pick(("trapezoid", "kite"))
    return None


# ------------------------------------------------------------------------------------------------ brackets --
PAIRS = {")": "(", "]": "[", "}": "{", ">": "<"}
OPEN = {v: k for k, v in PAIRS.items()}


def _stack(seq):
    st = []
    for ch in seq:
        if ch in OPEN:
            st.append(ch)
        elif ch in PAIRS:
            if not st or st[-1] != PAIRS[ch]:
                return None
            st.pop()
    return st


def brackets(state, q):
    text = main_text(state, q)
    seqs = re.findall(r"(?:[()\[\]{}<>]\s*){4,}", text)
    if len(seqs) != 1:
        return None
    seq = re.sub(r"\s+", "", seqs[0])
    o = opt_texts(q)
    if all(re.fullmatch(r"[\s()\[\]{}<>]+", v) for v in o.values()):
        st = _stack(seq)
        if st is None or not st:
            return None
        want = "".join(OPEN[c] for c in reversed(st))
        return _pick(q, lambda v: re.sub(r"\s+", "", v) == want)
    if yes_no_keys(q) and re.search(r"\b(?:balanced|properly (?:closed|nested)|well[- ]formed|valid)\b", text, re.I):
        st = _stack(seq)
        return _yn(q, st is not None and not st)
    return None


# ------------------------------------------------------------------------------------------------- sorting --
def sorting(state, q):
    text = main_text(state, q)
    if not re.search(r"\b(?:alphabetical(?:ly)?|sort(?:ed)?|lexicographic)\b", text, re.I):
        return None
    o = opt_texts(q)
    seqs = {k: re.findall(r"[A-Za-z][\w'-]*", v) for k, v in o.items()}
    bags = {tuple(sorted(w.lower() for w in s)) for s in seqs.values()}
    if len(bags) != 1 or len(next(iter(bags))) < 3:
        return None
    rev = bool(re.search(r"\b(?:reverse|descending|z to a)\b", text, re.I))
    return _pick(q, lambda v: [w.lower() for w in re.findall(r"[A-Za-z][\w'-]*", v)] == sorted((w.lower() for w in re.findall(r"[A-Za-z][\w'-]*", v)), reverse=rev))


SOLVERS = (ordering, swaps, movement, intervals, shapes, brackets, sorting)


def solve(state, questions):
    out = {}
    for k, q in questions.items():
        if q["type"] != "choice":
            continue
        for fn in SOLVERS:
            try:
                r = fn(state, q)
            except Exception:
                r = None
            if r:
                out[k] = r
                break
    return out or None
