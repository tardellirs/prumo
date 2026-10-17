"""Capability: match a tool/function catalog (JSON schemas) against the user's request.

Triggered by structure: the state holds a list of tool objects (each with a `name` and a description and/or a
parameter schema). Nothing depends on the benchmark's wording.

1. Ranking: BM25 of the user's request against each tool's name, description, parameter names and parameter
   descriptions; IDF is computed over this catalog only.
2. "Should tool X be called?" (a yes/no question that names exactly one catalog tool): yes when X is among the
   best-ranked tools (score >= NEAR x best score, best > 0).
3. "Which tool?" (options are catalog tool names): softmax over the ranking.
4. "Which reply?" (options are a JSON tool call, a question back to the user, a refusal, a plain answer): schema
   validation of the offered call -
     - the call must name a catalog tool and the request must be related to that tool (a non-generic word of
       the tool name, or two words of its description, shared with the request; prefix match);
     - every required parameter of that tool must be present in the call with a value that is stated in the
       request (a value copied from the schema default does not count: defaults are examples, not user input);
     - valid and grounded -> make the call; related but a required value is not stated -> ask for it;
       no related tool in the catalog (or the call is not a catalog tool) -> say it cannot be done.
"""
import json
import re

from prumo_rules.common import crit_text, from_scores, peaked, tokens, uniq
from prumo_rules.textmatch import bm25_scores
from prumo_rules.params import P as PRM

NEAR = PRM["tools.near"]            # per-tool yes/no: "best-ranked" band (chosen on BFCL-style train sets + SS-dev; see README)
LAST_WEIGHT = int(PRM["tools.last_weight"])       # the latest user turn adds 4x its own BM25 (recency; chosen on API-Bank train)
REFUSE = ("sorry", "unable", "can't", "cannot", "can not", "not able", "don't have", "do not have", "no tool", "none of the",
          "not possible", "i'm afraid", "apologies", "not capable", "no way to", "isn't possible", "not available")
TURN_TEXT = ("content", "text", "message", "utterance", "value")
USER_FIELDS = ("question", "query", "request", "user", "user_request", "message", "prompt", "instruction")
RESULT_ROLES = ("api", "tool", "function", "observation", "ipython", "tool_result")


def _is_tool(t):
    return isinstance(t, dict) and isinstance(t.get("name"), str) and any(
        k in t for k in ("description", "parameters", "input_parameters", "input_schema", "args", "arguments"))


def catalog(state):
    """The tool list of the request (any field holding a list of tool objects); [] for an explicitly empty catalog."""
    if not isinstance(state, dict):
        return None
    for k, v in state.items():
        if isinstance(v, list) and v and all(_is_tool(t) for t in v):
            return v
    for k, v in state.items():
        if isinstance(v, list) and not v and re.search(r"tool|function|api", k, re.I):
            return []
    return None


def _turns(state):
    out = []
    for v in state.values():
        if isinstance(v, list):
            flat = []
            for x in v:
                flat.extend(x if isinstance(x, list) else [x])
            if flat and all(isinstance(t, dict) and "role" in t for t in flat):
                out.extend(flat)
    return out


def _turn_text(t):
    for k in TURN_TEXT:
        if isinstance(t.get(k), str):
            return t[k]
    return ""


def user_turns(state, segment=None):
    """User turns in order. segment=True keeps only those after the last tool result (the still-open request)."""
    seg = SEGMENT if segment is None else segment
    tt = _turns(state)
    if seg:
        last = max((i for i, t in enumerate(tt) if str(t.get("role", "")).lower() in RESULT_ROLES), default=-1)
        if any(str(t.get("role", "")).lower() in ("user", "human", "customer") for t in tt[last + 1:]):
            tt = tt[last + 1:]
    turns = [_turn_text(t) for t in tt if str(t.get("role", "")).lower() in ("user", "human", "customer")]
    for k in USER_FIELDS:
        if isinstance(state.get(k), str):
            turns.append(state[k])
    return [t for t in turns if t]


SEGMENT = False


def user_text(state, last_weight=None):
    last_weight = LAST_WEIGHT if last_weight is None else last_weight
    turns = user_turns(state)
    if not turns:
        return ""
    return " ".join(turns[:-1] + [turns[-1]] * last_weight)


def _props(t):
    params = t.get("parameters") or t.get("input_parameters") or t.get("input_schema") or {}
    if isinstance(params, dict) and isinstance(params.get("properties"), dict):
        return params["properties"], params
    return (params if isinstance(params, dict) else {}), {}


def tool_doc(t):
    props, _ = _props(t)
    pn = " ".join(props.keys())
    pd = " ".join(str(v.get("description", "")) for v in props.values() if isinstance(v, dict))
    name = re.sub(r"[._]", " ", str(t.get("name", "")))
    return f"{name} {name} {t.get('description', '')} {pn} {pd}"


def required_params(t):
    props, schema = _props(t)
    req = list(t.get("required") or []) if isinstance(t.get("required"), list) else []
    if isinstance(schema.get("required"), list):
        req += schema["required"]
    req += [k for k, v in props.items() if isinstance(v, dict) and v.get("required") is True]
    return uniq(r for r in req if isinstance(r, str))


def defaults(t):
    props, _ = _props(t)
    return {k: v["default"] for k, v in props.items() if isinstance(v, dict) and "default" in v}


def norm_name(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def rank_tools(state, tools):
    """BM25 of the open request against each tool; the latest user turn adds LAST_WEIGHT x its own BM25 (recency)."""
    turns = user_turns(state)
    docs = [tool_doc(t) for t in tools]
    sc = bm25_scores(" ".join(turns), docs)
    if turns and LAST_WEIGHT and len(turns) > 1:
        last = bm25_scores(turns[-1], docs)
        sc = [a + LAST_WEIGHT * b for a, b in zip(sc, last)]
    out = {}
    for t, s in zip(tools, sc):
        out[t["name"]] = max(out.get(t["name"], 0.0), s)
    return out


def _named_tool(q, tools):
    """The single catalog tool named in the question (longest exact name occurring in the instructions)."""
    ins = crit_text(q.get("instructions"))
    hits = []
    for t in tools:
        n = t["name"]
        if len(n) >= 3 and re.search(r"(?<![\w.])" + re.escape(n) + r"(?![\w])", ins):
            hits.append(n)
    if not hits:
        return None
    hits.sort(key=lambda n: -len(n))
    if len(hits) > 1 and len(hits[0]) == len(hits[1]):
        return None
    return hits[0]


STRONG_REFUSE = ("unable", "can't", "cannot", "can not", "not able", "don't have", "do not have", "no tool", "none of the",
                 "not possible", "not capable", "no way to", "isn't possible", "not available", "beyond my", "outside my")


def reply_kind(text):
    """call (a JSON object naming a function) / ask (a question back) / cannot (a refusal) / direct (anything else,
    including JSON that is data rather than a call)."""
    t = text.strip()
    if t.startswith("{") or t.startswith("["):
        try:
            v = json.loads(t)
        except Exception:
            v = None
        if isinstance(v, list) and len(v) == 1:
            v = v[0]
        if isinstance(v, dict) and any(isinstance(v.get(k), str) for k in ("name", "function", "tool", "api_name")):
            return "call", v
        return "direct", None
    low = t.lower()
    if any(w in low for w in STRONG_REFUSE):
        return "cannot", None
    if "?" in t:
        return "ask", None
    if any(w in low for w in REFUSE):
        return "cannot", None
    return "direct", None


GENERIC = frozenset(tokens("get gets fetch fetches retrieve retrieves return returns given specified specific based using use api "
                           "information info data list lists search searches find finds provide provides details detail query "
                           "request result results value values type types number new current"))


def related(u, tool, mode=None):
    """Does the request share a content word with the tool's name and description?"""
    mode = RELATED if mode is None else mode
    doc = re.sub(r"[._]", " ", tool["name"]) + " " + str(tool.get("description", ""))
    if mode >= 2:
        props, _ = _props(tool)
        doc += " " + " ".join(str(v.get("description", "")) for v in props.values() if isinstance(v, dict))
    ut = set(tokens(u))
    dt = set(tokens(doc))
    if mode >= 3:
        ut -= GENERIC
        dt -= GENERIC
    if mode == 4:
        nt = set(tokens(re.sub(r"[._]", " ", tool["name"]))) - GENERIC
        pre = lambda ws: {w[:5] for w in ws if len(w) >= 5} | {w for w in ws if len(w) < 5}
        if pre(nt) & pre(ut):
            return True
        return len(pre(dt - nt) & pre(ut)) >= 2
    if ut & dt:
        return True
    if mode >= 1:
        P = 5 if mode != 2 else 4
        up = {w[:P] for w in ut if len(w) >= P}
        return any(len(w) >= P and w[:P] in up for w in dt)
    return False


RELATED = int(PRM["tools.related"])
DEFAULT_OK = False


WEEKDAYS = "monday tuesday wednesday thursday friday saturday sunday".split()
RELDATE = ("today", "tomorrow", "yesterday", "tonight", "next week", "this week", "next month", "this weekend") + tuple(WEEKDAYS)


def stated(value, text):
    """Is an argument value stated in the user's text?"""
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        v = int(value) if float(value).is_integer() else value
        return re.search(r"(?<![\d.])" + re.escape(str(v)) + r"(?![\d])", text.replace(",", "")) is not None
    if isinstance(value, list):
        return all(stated(v, text) for v in value) if value else False
    if isinstance(value, dict):
        return all(stated(v, text) for v in value.values()) if value else False
    s = str(value).strip()
    if not s or s.lower() in ("default", "none", "null", "unknown", "n/a", "<unknown>", "?", "tbd", "your_value"):
        return False
    low = text.lower()
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})(?:[ t].*)?", s, re.I)
    if m:
        day = str(int(m.group(3)))
        return re.search(r"(?<!\d)" + day + r"(?:st|nd|rd|th)?(?!\d)", low) is not None or any(w in low for w in RELDATE)
    tv = set(tokens(s, keep_stop=True))
    tt = set(tokens(text, keep_stop=True))
    if not tv:
        return False
    return len(tv & tt) / len(tv) >= PRM["tools.stated_overlap"] or any(len(w) >= 4 and w in tt for w in tv)


def reply_choice(state, q, tools):
    opts = {k: reply_kind(crit_text(v)) for k, v in q["criteria"].items()}
    kinds = {}
    for k, (kind, _) in opts.items():
        kinds.setdefault(kind, []).append(k)
    if "call" not in kinds or ("cannot" not in kinds and "ask" not in kinds):
        return None
    keys = list(q["criteria"])
    cannot = kinds.get("cannot") or kinds.get("direct", [])
    u = " ".join(user_turns(state))
    if not tools:
        return peaked(keys, cannot, PRM["tools.mass_no_catalog"])
    call_key = kinds["call"][0]
    call = opts[call_key][1] if isinstance(opts[call_key][1], dict) else {}
    names = {norm_name(t["name"]): t for t in tools}
    tool = names.get(norm_name(call.get("name", ""))) if call.get("name") else None
    if tool is None or not related(u, tool):
        # no valid call can be made with this catalog for this request
        return peaked(keys, cannot or kinds.get("ask", []), PRM["tools.mass_reply"])
    args = call.get("arguments", call.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    args = args if isinstance(args, dict) else {}
    dflt = defaults(tool)
    missing = [p for p in required_params(tool) if not (stated(args.get(p), u) or (DEFAULT_OK and p in args and p in dflt and args[p] == dflt[p]))]
    if missing and "ask" in kinds:
        return peaked(keys, kinds["ask"], PRM["tools.mass_reply"])
    return peaked(keys, [call_key], PRM["tools.mass_reply"])


YES_CALL = ("yes", "call", "invoke", "use", "true", "required", "needed")
NO_CALL = ("no", "skip", "don't", "do", "not", "false", "none", "unnecessary")


def _call_keys(q):
    """(yes, no) keys of a two-option "call this tool?" question, by key or by the description's first word."""
    if len(q["criteria"]) != 2:
        return None
    yes, no = [], []
    for k, v in q["criteria"].items():
        first = (re.findall(r"[a-z']+", crit_text(v).lower()) or [""])[0]
        kk = k.lower()
        if kk in YES_CALL or first in YES_CALL:
            yes.append(k)
        elif kk in NO_CALL or first in NO_CALL:
            no.append(k)
    return (yes[0], no[0]) if len(yes) == 1 and len(no) == 1 else None


def solve(state, questions):
    tools = catalog(state)
    if tools is None:
        return None
    out = {}
    sc = rank_tools(state, tools) if tools else {}
    by = {norm_name(t["name"]): t["name"] for t in tools}
    per_tool = {}
    for k, q in questions.items():
        if q["type"] not in ("choice", "noul"):
            continue
        if q["type"] == "noul" or len(q["criteria"]) == 2:
            name = _named_tool(q, tools) if tools else None
            if name:
                per_tool[k] = name
                continue
        if q["type"] != "choice":
            continue
        opt_names = {o: norm_name(crit_text(v)) for o, v in q["criteria"].items()}
        if tools and all(n in by for n in opt_names.values()):
            out[k] = from_scores({o: sc[by[n]] for o, n in opt_names.items()}, temp=PRM["tools.temperature"])
            continue
        r = reply_choice(state, q, tools)
        if r:
            out[k] = r
    if per_tool:
        best = max(sc.values()) if sc else 0.0
        for k, name in per_tool.items():
            s = sc.get(name, 0.0)
            call = len(tools) == 1 or (best > 0 and s >= NEAR * best)
            q = questions[k]
            if q["type"] == "noul":
                out[k] = {"noul": PRM["tools.call_high"] if call else PRM["tools.call_low"]}
                continue
            yn = _call_keys(q)
            if yn:
                out[k] = peaked(list(q["criteria"]), [yn[0] if call else yn[1]], PRM["tools.call_high"])
    return out or None
