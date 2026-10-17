"""Capabilities: exact logic.

1. Boolean expressions: a span made only of True / False / and / or / not / parentheses is parsed (no eval) and
   evaluated; options True/False (or yes/no) are matched to the value.
2. Propositional logic in English: premises and a claim are parsed into formulas over atomic sentences (if/then,
   only if, unless, either/or (or both | but not both), neither/nor, not both, it is (not) the case / true / false
   that, and, or, if and only if, sentence negation) and decided by truth table: the claim follows (true in every
   model of the premises), is contradicted (false in every model) or is undetermined. Claimed only when at least
   one premise is compound and every atom of the claim occurs in the premises.
3. Truth-teller puzzles: "X lies / tells the truth", "Y says (that) X lies / tells the truth / is a liar / is
   honest" are constraints over who is truthful (a speaker is truthful iff the statement is true); solved by
   enumeration, answered only when the queried person's status is unique.
"""
import itertools
import re

from prumo_rules.common import crit_text, main_text, opt_texts, peaked, yes_no_keys

# ----------------------------------------------------------------------------------------- boolean expressions --
BOOL_SPAN = re.compile(r"(?:\bnot\b|\band\b|\bor\b|\bTrue\b|\bFalse\b|[()])(?:[\s()]*(?:\bnot\b|\band\b|\bor\b|\bTrue\b|\bFalse\b|[()]))+")


def _bool_parse(toks):
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def eat(t=None):
        x = peek()
        if x is None or (t and x != t):
            raise ValueError("parse")
        pos[0] += 1
        return x

    def p_or():
        v = p_and()
        while peek() == "or":
            eat()
            r = p_and()
            v = v or r
        return v

    def p_and():
        v = p_not()
        while peek() == "and":
            eat()
            r = p_not()
            v = v and r
        return v

    def p_not():
        if peek() == "not":
            eat()
            return not p_not()
        if peek() == "(":
            eat("(")
            v = p_or()
            eat(")")
            return v
        t = eat()
        if t not in ("True", "False"):
            raise ValueError("atom")
        return t == "True"

    v = p_or()
    if pos[0] != len(toks):
        raise ValueError("trailing")
    return v


def boolean(state, q):
    text = main_text(state, q)
    spans = []
    for m in BOOL_SPAN.finditer(text):
        s = m.group(0).strip()
        toks = re.findall(r"\(|\)|\w+", s)
        while toks and toks[-1] in ("and", "or", "not", "("):
            toks.pop()
        while toks and toks[0] in ("and", "or", ")"):
            toks.pop(0)
        if len(toks) >= 3 and any(t in ("and", "or", "not") for t in toks) and any(t in ("True", "False") for t in toks):
            spans.append(toks)
    if len(spans) != 1:
        return None
    try:
        val = _bool_parse(spans[0])
    except (ValueError, RecursionError):
        return None
    o = opt_texts(q)
    tf = {k: v.lower().strip(" .") for k, v in o.items()}
    t_k = [k for k, v in tf.items() if v == "true"]
    f_k = [k for k, v in tf.items() if v == "false"]
    if len(t_k) == 1 and len(f_k) == 1:
        return peaked(list(o), t_k if val else f_k, 0.95)
    yn = yes_no_keys(q)
    if yn:
        ask_false = re.search(r"\b(?:evaluate|evaluates|is it|is this|equal)\b[^.?]*\bfalse\b", text, re.I)
        v = (not val) if ask_false else val
        return peaked(list(o), [yn[0] if v else yn[1]], 0.95)
    return None


# ------------------------------------------------------------------------------------- propositional logic --
NEG_VERB = re.compile(r"\b(is not|are not|isn't|aren't|was not|wasn't|were not|weren't|has not|hasn't|have not|haven't|"
                      r"does not|doesn't|do not|don't|did not|didn't|cannot|can not|can't|will not|won't)\b")
NEG_POS = {"is not": "is", "are not": "are", "isn't": "is", "aren't": "are", "was not": "was", "wasn't": "was", "were not": "were",
           "weren't": "were", "has not": "has", "hasn't": "has", "have not": "have", "haven't": "have", "does not": "",
           "doesn't": "", "do not": "", "don't": "", "did not": "", "didn't": "", "cannot": "can", "can not": "can",
           "can't": "can", "will not": "will", "won't": "will"}


def _norm_atom(s):
    s = s.strip().strip(".,;:!").strip()
    s = re.sub(r"^(?:that|the case that)\s+", "", s, flags=re.I)
    neg = False
    m = NEG_VERB.search(s)
    if m:
        s = (s[: m.start()] + NEG_POS[m.group(1).lower()] + s[m.end():])
        neg = True
    s = re.sub(r"\s+", " ", s.lower()).strip()
    # third-person verb forms after do-support removal: "the light does not shine" -> "the light shine" ~ "the light shines"
    s = " ".join(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in s.split())
    return s, neg


class Unparsed(ValueError):
    pass


def _formula(s, atoms, depth=0):
    if depth > 12:
        raise Unparsed(s)
    s = s.strip().rstrip(".").strip()
    s = re.sub(r"^(?:and|but|also|moreover|furthermore|however),?\s+", "", s, flags=re.I)
    f = lambda x: _formula(x, atoms, depth + 1)
    m = re.fullmatch(r"(?:if|whenever|when)\s+(.+?),?\s+then\s+(.+)", s, re.I) or re.fullmatch(r"(?:if|whenever)\s+(.+?),\s+(.+)", s, re.I)
    if m:
        a, b = f(m.group(1)), f(m.group(2))
        return lambda v: (not a(v)) or b(v)
    m = re.fullmatch(r"(.+?),?\s+if and only if\s+(.+)", s, re.I)
    if m:
        a, b = f(m.group(1)), f(m.group(2))
        return lambda v: a(v) == b(v)
    m = re.fullmatch(r"(.+?),?\s+only if\s+(.+)", s, re.I)
    if m:
        a, b = f(m.group(1)), f(m.group(2))
        return lambda v: (not a(v)) or b(v)
    m = re.fullmatch(r"(.+?),?\s+unless\s+(.+)", s, re.I)
    if m:
        a, b = f(m.group(1)), f(m.group(2))
        return lambda v: a(v) or b(v)
    m = re.fullmatch(r"(.+?),?\s+(?:implies that|implies)\s+(.+)", s, re.I)
    if m:
        a, b = f(m.group(1)), f(m.group(2))
        return lambda v: (not a(v)) or b(v)
    m = re.fullmatch(r"either\s+(.+?)\s+or\s+(.+?),?\s*\(?\s*but not both\s*\)?", s, re.I)
    if m:
        a, b = f(m.group(1)), f(m.group(2))
        return lambda v: a(v) != b(v)
    m = re.fullmatch(r"either\s+(.+?)\s+or\s+(.+?)(?:,?\s*\(?\s*or both\s*\)?)?", s, re.I)
    if m:
        a, b = f(m.group(1)), f(m.group(2))
        return lambda v: a(v) or b(v)
    m = re.fullmatch(r"neither\s+(.+?)\s+nor\s+(.+)", s, re.I)
    if m:
        a, b = f(m.group(1)), f(m.group(2))
        return lambda v: not a(v) and not b(v)
    m = re.fullmatch(r"(?:it is not the case that|it is not true that|it is false that|it's not the case that|not)\s+both\s+(.+?)\s+and\s+(.+)", s, re.I)
    if m:
        a, b = f(m.group(1)), f(m.group(2))
        return lambda v: not (a(v) and b(v))
    m = re.fullmatch(r"(?:it is true that|it is the case that|it's true that)\s+(.+)", s, re.I)
    if m:
        return f(m.group(1))
    m = re.fullmatch(r"(?:it is false that|it is not the case that|it is not true that|it's not the case that|it's false that)\s+(.+)", s, re.I)
    if m:
        a = f(m.group(1))
        return lambda v: not a(v)
    m = re.fullmatch(r"(?:both\s+)?(.+?),?\s+and\s+(.+)", s, re.I)
    if m:
        a, b = f(m.group(1)), f(m.group(2))
        return lambda v: a(v) and b(v)
    m = re.fullmatch(r"(.+?),?\s+or\s+(.+)", s, re.I)
    if m:
        a, b = f(m.group(1)), f(m.group(2))
        return lambda v: a(v) or b(v)
    if re.search(r"\b(if|either|then|unless|neither|nor|only if|iff)\b", s, re.I) or len(s.split()) < 2:
        raise Unparsed(s)
    key, neg = _norm_atom(s)
    if not key:
        raise Unparsed(s)
    atoms.setdefault(key, len(atoms))
    i = atoms[key]
    return (lambda v: not v[i]) if neg else (lambda v: v[i])


COMPOUND = re.compile(r"\b(if|either|unless|neither|only if|implies|or|not the case|false that|not true that)\b", re.I)


def _sentences(text):
    return [p for p in re.split(r"(?<=[.!;])\s+|\n+", text.strip()) if p.strip()]


def _premises_claim(state, q):
    claim = prem = None
    if isinstance(state, dict):
        for k in ("claim", "conclusion", "statement", "hypothesis", "query"):
            if isinstance(state.get(k), str):
                claim = state[k]
                break
        for k in ("premises", "facts", "rules", "context", "given"):
            v = state.get(k)
            if isinstance(v, str):
                prem = v
                break
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                prem = " ".join(x.strip().rstrip(".") + "." for x in v)
                break
    ins = crit_text(q.get("instructions"))
    if claim is None:
        m = re.search(r"(?:claim|conclusion|statement|hypothesis)\s*:\s*(.+?)(?:\n|$)", ins, re.I)
        if m:
            claim = m.group(1)
    if prem is None:
        m = re.search(r"premises?\s*:\s*(.+?)(?:\n\s*\n|(?:claim|conclusion|statement|hypothesis)\s*:|$)", ins, re.I | re.S)
        if m:
            prem = m.group(1)
    return prem, claim


def _verdict_keys(q):
    o = {k: (k + " " + crit_text(v)).lower() for k, v in q["criteria"].items()}
    km = {}
    for k, v in o.items():
        if re.search(r"undetermin|unknown|cannot be determined|can't be determined|not enough|uncertain|true or false|may or may not|might or might not|neither|neutral|not determined|undecid", v):
            km.setdefault("undetermined", []).append(k)
        elif re.search(r"contradict|rule[sd]? (?:the claim |it )?out|must be false|refute|\bfalse\b|impossible|inconsistent", v):
            km.setdefault("contradicted", []).append(k)
        elif re.search(r"follow|entail|guarantee|must be true|\btrue\b|valid|proved|implied", v):
            km.setdefault("follows", []).append(k)
    if all(len(km.get(x, [])) == 1 for x in ("follows", "contradicted", "undetermined")) and len(o) == 3:
        return {x: km[x][0] for x in km}
    return None


def propositional(state, q):
    prem, claim = _premises_claim(state, q)
    if not prem or not claim:
        return None
    atoms = {}
    try:
        sents = _sentences(prem)
        if not any(COMPOUND.search(s) for s in sents):
            return None
        P = [_formula(s, atoms) for s in sents]
        n_prem_atoms = len(atoms)
        C = _formula(claim, atoms)
    except (Unparsed, RecursionError):
        return None
    if len(atoms) != n_prem_atoms or len(atoms) > 14 or not atoms:
        return None
    vals = set()
    for v in itertools.product((False, True), repeat=len(atoms)):
        if all(p(v) for p in P):
            vals.add(C(v))
            if len(vals) == 2:
                break
    if not vals:
        return None
    verdict = "follows" if vals == {True} else "contradicted" if vals == {False} else "undetermined"
    km = _verdict_keys(q)
    if km:
        return peaked(list(q["criteria"]), [km[verdict]], 0.93)
    yn = yes_no_keys(q) if q["type"] == "choice" else None
    ins = crit_text(q.get("instructions")).lower()
    if q["type"] == "noul" or yn:
        if re.search(r"\b(?:follow|entail|guarantee|must be true|necessarily true|implied)", ins):
            val = verdict == "follows"
        elif re.search(r"\b(?:contradict|rule[sd]? out|must be false|necessarily false)", ins):
            val = verdict == "contradicted"
        else:
            return None
        if q["type"] == "noul":
            return {"noul": 0.93 if val else 0.07}
        return peaked(list(q["criteria"]), [yn[0] if val else yn[1]], 0.93)
    o = {k: crit_text(v).lower() for k, v in q["criteria"].items()}
    ent = [k for k, v in o.items() if v.strip() in ("entailment", "entailed")]
    non = [k for k, v in o.items() if v.strip() in ("non-entailment", "not entailed", "not entailment")]
    if len(ent) == 1 and len(non) == 1 and len(o) == 2:
        return peaked(list(o), ent if verdict == "follows" else non, 0.93)
    return None


# ------------------------------------------------------------------------------------- truth-teller puzzles --
TRUE_P = r"(?:tells the truth|is telling the truth|is truthful|is honest|is a truth-teller|is a knight|always tells the truth|speaks the truth)"
FALSE_P = r"(?:lies|is lying|is a liar|is dishonest|is a knave|always lies|tells lies|is not telling the truth|does not tell the truth|doesn't tell the truth)"
NAME = r"([A-Z][a-z]+)"


def _stmt(s):
    """'X lies' / 'X tells the truth' -> (name, truthful)."""
    s = s.strip().rstrip(".")
    m = re.fullmatch(NAME + r"\s+" + TRUE_P, s)
    if m:
        return m.group(1), True
    m = re.fullmatch(NAME + r"\s+" + FALSE_P, s)
    if m:
        return m.group(1), False
    return None


def truth_tellers(state, q):
    text = main_text(state, q)
    if not re.search(r"\b(?:says|claims|states)\b", text) or not re.search(TRUE_P + "|" + FALSE_P, text):
        return None
    m = re.search(r"\b(?:Does|Is)\s+" + NAME + r"\s+(tell the truth|telling the truth|truthful|honest|lie|lying|a liar|a knight|a knave)\s*\?", text)
    if not m:
        return None
    target, asks_truth = m.group(1), m.group(2) in ("tell the truth", "telling the truth", "truthful", "honest", "a knight")
    body = text[: m.start()]
    facts, says = [], []
    for sent in re.split(r"(?<=[.!?])\s+|\n+", body):
        sent = re.sub(r"^(?:question|q|problem|puzzle)\s*:\s*", "", sent.strip(), flags=re.I)
        if not sent:
            continue
        st = _stmt(sent)
        if st:
            facts.append(st)
            continue
        mm = re.fullmatch(NAME + r"\s+(?:says|claims|states)(?: that)?\s+(.+?)\.?", sent)
        if mm:
            inner = _stmt(mm.group(2))
            if not inner:
                return None
            says.append((mm.group(1), inner))
            continue
        if re.search(TRUE_P + "|" + FALSE_P, sent):
            return None  # a statement about truthfulness that we could not read: do not guess
    people = sorted({p for p, _ in facts} | {a for a, _ in says} | {b for _, (b, _) in says})
    if target not in people or len(people) > 12 or not says:
        return None
    idx = {p: i for i, p in enumerate(people)}
    sols = set()
    for v in itertools.product((False, True), repeat=len(people)):
        if all(v[idx[p]] == t for p, t in facts) and all(v[idx[a]] == (v[idx[b]] == t) for a, (b, t) in says):
            sols.add(v[idx[target]])
    if len(sols) != 1:
        return None
    val = sols.pop()
    val = val if asks_truth else not val
    yn = yes_no_keys(q)
    if not yn:
        return None
    return peaked(list(q["criteria"]), [yn[0] if val else yn[1]], 0.95)


SOLVERS = (boolean, truth_tellers, propositional)


def solve(state, questions):
    out = {}
    for k, q in questions.items():
        if q["type"] not in ("choice", "noul"):
            continue
        for fn in SOLVERS:
            if q["type"] == "noul" and fn is not propositional:
                continue
            try:
                r = fn(state, q)
            except Exception:
                r = None
            if r:
                out[k] = r
                break
    return out or None
