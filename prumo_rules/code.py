"""Capability: execute the Python code supplied in the request (sandboxed, see sandbox.py).

- output selection: the request carries Python code that defines a function and an input (argument list), and
  the options are Python literals -> run f(input) and pick the option(s) equal to the value.
- test passing: the request carries code and tests -> run the tests (a `check(candidate)` function, if defined, is
  called on the entry function; bare asserts run as they are) and answer the yes/no question.
Detection is by content (a parseable `def` in a code field or a fenced block), never by benchmark name.
"""
import ast
import re
import warnings

from prumo_rules import sandbox
from prumo_rules.common import bool_answer, crit_text, peaked, yes_no_keys
from prumo_rules.params import P as PRM

CODE_KEYS = ("code", "function", "program", "source", "snippet")
TEST_KEYS = ("tests", "test", "test_code", "unit_tests", "test_cases")
INPUT_KEYS = ("input", "inputs", "args", "arguments", "call_args")
FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def _defs(code):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return []
    return [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _get(state, keys):
    if not isinstance(state, dict):
        return None
    for k in keys:
        v = state.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _code_in_text(text):
    m = FENCE.search(text)
    if m and "def " in m.group(1):
        return m.group(1)
    i = text.find("def ")
    if i < 0 or (i > 0 and text[i - 1] not in "\n"):
        return None
    lines, out = text[i:].split("\n"), []
    for j, ln in enumerate(lines):
        if j and ln.strip() and not ln.startswith((" ", "\t", "def ", "class ", "@", "import ", "from ")):
            break
        out.append(ln)
    return "\n".join(out)


def _call_in_text(text, fname):
    m = re.search(r"\b" + re.escape(fname) + r"\((.*?)\)\s*(?:==|returns?\b|\?|$)", text, re.M)
    return m.group(1) if m else None


def _is_literal(s):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ast.literal_eval(s.strip())
        return True
    except Exception:
        return False


def solve(state, questions):
    code = _get(state, CODE_KEYS)
    texts = [state] if isinstance(state, str) else [v for v in state.values() if isinstance(v, str)] if isinstance(state, dict) else []
    texts += [crit_text(q.get("instructions")) for q in questions.values()]
    if not code:
        for t in texts:
            c = _code_in_text(t) if "def " in t else None
            if c and _defs(c):
                code = c
                break
    if not code or "def " not in code:
        return None
    defs = _defs(code)
    if not defs:
        return None
    out = {}
    tests = _get(state, TEST_KEYS)
    inp = _get(state, INPUT_KEYS)
    fname = "f" if "f" in defs else defs[0]
    if inp is None:
        for t in texts:
            inp = _call_in_text(t.replace(code, ""), fname)
            if inp is not None:
                break
    for key, q in questions.items():
        if tests:
            if q["type"] != "noul" and not yes_no_keys(q):
                continue
            entry = state.get("entry_point") if isinstance(state, dict) and isinstance(state.get("entry_point"), str) else None
            if not entry:
                called = set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", tests))
                cands = [d for d in defs if d in called] or defs
                entry = cands[-1]
            r = sandbox.run({"mode": "tests", "code": code, "tests": tests, "entry": entry})
            if not r.get("ok"):
                continue
            ans = bool_answer(q, bool(r.get("passes")), PRM["code.mass"])
            if ans:
                out[key] = ans
        elif inp is not None and q["type"] == "choice":
            keys = list(q["criteria"])
            cands = [crit_text(q["criteria"][k]) for k in keys]
            if not all(_is_literal(c) for c in cands):
                continue
            r = sandbox.run({"mode": "output", "code": code, "call": f"{fname}({inp})", "candidates": cands})
            if not r.get("ok"):
                continue
            hits = [k for k, m in zip(keys, r["matches"]) if m]
            if hits and len(hits) < len(keys):
                out[key] = peaked(keys, hits, PRM["code.mass"])
    return out or None
