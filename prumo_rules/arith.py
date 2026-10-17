"""Capability: numeric answers.

1. Exact arithmetic: an expression of numbers, + - * / and parentheses in the text (at least two operators) is
   parsed (no eval) and evaluated with exact fractions; the option equal to the value wins.
2. Word problems with numeric options: which option can be *derived* from the numbers stated in the problem?
   Numbers are read from the text (digits, number words, a few unit words that imply a constant: dozen, percent,
   hour, week, half...). Values reachable by combining the stated numbers with + - * / are enumerated by dynamic
   programming over subsets, under a fixed operation budget (deterministic, bounded latency). An option scores by
   the richest derivation that reaches it; an option that is just a stated number scores low; unreachable
   options score 0. Pure arithmetic search; nothing learned.
"""
import itertools
import re
from fractions import Fraction

from prumo_rules.common import crit_text, from_scores, main_text, numbers, opt_texts, peaked, to_number

# ------------------------------------------------------------------------------------------- exact arithmetic --
EXPR = re.compile(r"[\d(\-][\d\s+\-*/×÷().]*[\d)]")


def _eval(tokens):
    pos = [0]

    def peek():
        return tokens[pos[0]] if pos[0] < len(tokens) else None

    def eat():
        pos[0] += 1
        return tokens[pos[0] - 1]

    def expr():
        v = term()
        while peek() in ("+", "-"):
            op = eat()
            r = term()
            v = v + r if op == "+" else v - r
        return v

    def term():
        v = factor()
        while peek() in ("*", "/"):
            op = eat()
            r = factor()
            if op == "/":
                if r == 0:
                    raise ZeroDivisionError
                v = v / r
            else:
                v = v * r
        return v

    def factor():
        t = peek()
        if t == "-":
            eat()
            return -factor()
        if t == "+":
            eat()
            return factor()
        if t == "(":
            eat()
            v = expr()
            if eat() != ")":
                raise ValueError
            return v
        if t is None or not re.fullmatch(r"\d+(?:\.\d+)?", t):
            raise ValueError
        eat()
        return Fraction(t)

    v = expr()
    if pos[0] != len(tokens):
        raise ValueError
    return v


def exact(state, q):
    text = main_text(state, q).replace("×", "*").replace("÷", "/").replace("−", "-")
    vals = []
    for m in EXPR.finditer(text):
        s = m.group(0)
        toks = re.findall(r"\d+(?:\.\d+)?|[+\-*/()]", s)
        ops = sum(1 for t in toks if t in "+-*/")
        if ops < 2 or len(toks) < 5 or re.search(r"\d\s+\d", s):
            continue
        try:
            vals.append(_eval(toks))
        except (ValueError, ZeroDivisionError, IndexError, RecursionError):
            continue
    if len(vals) != 1:
        return None
    o = opt_texts(q)
    nums = {k: to_number(v) for k, v in o.items()}
    if any(v is None for v in nums.values()):
        return None
    target = float(vals[0])
    hits = [k for k, v in nums.items() if abs(v - target) < 1e-9 * max(1.0, abs(target))]
    if len(hits) != 1:
        return None
    return peaked(list(o), hits, 0.95)


# ------------------------------------------------------------------------------------------ reachability --
POOL = 6            # at most this many stated numbers are combined
MAXK = 4            # at most this many numbers in one derivation
BUDGET = 60000      # binary operations per question (fixed, deterministic)
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
         "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000, "half": 0.5, "twice": 2, "double": 2, "triple": 3,
         "thrice": 3, "quarter": 0.25, "dozen": 12, "pair": 2}
UNIT_CONST = {"percent": 100, "%": 100, "hour": 60, "minute": 60, "week": 7, "dozen": 12, "month": 12, "year": 12,
              "day": 24, "cent": 100, "half": 2, "twice": 2, "double": 2, "quarter": 4, "triple": 3}


def stated_numbers(text):
    vals = list(numbers(text))
    low = text.lower()
    for w, v in WORDS.items():
        vals.extend(float(v) for _ in re.finditer(r"\b" + w + r"\b", low))
    consts = sorted({float(v) for w, v in UNIT_CONST.items() if w in low})
    out = []
    for v in vals:
        if out.count(v) < 2:
            out.append(v)
    return out, consts[:3]


def _ops(a, b):
    yield a + b
    yield a * b
    yield a - b
    yield b - a
    if b:
        yield a / b
    if a:
        yield b / a


def _key(v):
    return round(v, 6)


def reachable(nums, maxk=MAXK, budget=BUDGET):
    """-> ({value: largest number of stated numbers whose combination reaches it}, values using all numbers).
    Subsets are visited in a fixed order and the expansion stops after `budget` binary operations."""
    n = len(nums)
    R = {1 << i: [_key(v)] for i, v in enumerate(nums)}
    best = {}
    for v in nums:
        best.setdefault(_key(v), 1)
    spent = 0
    for size in range(2, min(maxk, n) + 1):
        for combo in itertools.combinations(range(n), size):
            mask = sum(1 << i for i in combo)
            vals = {}
            sub = (mask - 1) & mask
            while sub and spent < budget:
                other = mask ^ sub
                if sub < other and sub in R and other in R:
                    for a in R[sub]:
                        for b in R[other]:
                            spent += 1
                            for v in _ops(a, b):
                                if abs(v) < 1e8:
                                    vals[_key(v)] = None
                        if spent >= budget:
                            break
                sub = (sub - 1) & mask
            R[mask] = list(vals)
            for v in R[mask]:
                if best.get(v, 0) < size:
                    best[v] = size
            if spent >= budget:
                break
        if spent >= budget:
            break
    full = set(R.get((1 << n) - 1, [])) if n <= maxk else set()
    return best, full


def _problem_text(state):
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return " ".join(str(state[k]) for k in ("document", "context", "passage", "question", "problem", "text") if isinstance(state.get(k), str))
    return ""


def derive(state, q):
    text = _problem_text(state)
    if not text or len(q["criteria"]) < 2:
        return None
    vals = {o: to_number(crit_text(v)) for o, v in q["criteria"].items()}
    if any(v is None for v in vals.values()):
        return None
    nums, consts = stated_numbers(text)
    if not nums:
        return None
    pool = nums[:POOL]
    best, full = reachable(pool, min(len(pool), MAXK + (1 if len(pool) <= 5 else 0)))
    if consts and len(pool) <= 4:
        b2, f2 = reachable(pool + consts[:1], len(pool) + 1)
        for v, kk in b2.items():
            if best.get(v, 0) < kk - 1:
                best[v] = kk - 1
        full = full | f2
    stated = {_key(v) for v in nums}
    sc = {}
    for o, v in vals.items():
        cands = [_key(v), _key(round(v, 2)), _key(round(v))]
        if any(c in full for c in cands) and _key(v) not in stated:
            sc[o] = 2.0
        elif _key(v) in stated:
            sc[o] = 0.3
        else:
            hit = max((best.get(c, 0) for c in cands), default=0)
            sc[o] = 0.0 if hit < 2 else 0.5 + 0.2 * hit
    if max(sc.values()) <= 0:
        return None
    return from_scores(sc, temp=0.25)


def solve(state, questions):
    out = {}
    for k, q in questions.items():
        if q["type"] != "choice":
            continue
        for fn in (exact, derive):
            try:
                r = fn(state, q)
            except Exception:
                r = None
            if r:
                out[k] = r
                break
    return out or None
