"""Prumo: a zero-parameter, rule-based decision engine for the Decision Index (no learned weights).

    cd prumo
    python -m decision_index run --engine decision_index_engine:DecisionEngine --out runs/<name>
    (--option solvers=code,chess,...  enables a subset; default: all)

Every question of a request goes through generic capabilities in a fixed order. Each capability reads the
*structure* of the request (Python code with a `def`, a FEN string, a list of tool schemas, candidates to rank,
a boolean expression, dates, an ordering-constraint puzzle, ...) and either returns probabilities for some
questions or passes. A question that no capability claims gets the uniform distribution and a deterministic hash
pick (expected score = chance). There are no answer tables, no benchmark names or item ids, no network access,
and nothing in the engine was fitted to data.
"""
import importlib
import time

from decision_index.engines import Engine

from prumo_rules.common import hpick

ORDER = ["code", "chess", "tools", "retrieval", "grounding", "consensus", "labels", "music", "logic", "puzzles", "dates",
         "arith"]


class DecisionEngine(Engine):
    name = "prumo"
    latency = "In-process wall time per request, including sandboxed code execution and the node-limited chess search."

    def __init__(self, **options):
        super().__init__(**options)
        names = options.get("solvers") or ",".join(ORDER)
        names = names if isinstance(names, str) else ",".join(names)
        self.solvers = [(n, importlib.import_module(f"prumo_rules.{n}")) for n in names.split(",") if n]
        from prumo_rules import sandbox
        self.provenance = {"kind": "rule-based, no weights", "learned_parameters": 0,
                           "capabilities": [n for n, _ in self.solvers], "code_isolation": sandbox.isolation()}

    def synchronize(self):
        pass

    def close(self):
        pass

    def decide(self, state, questions):
        claimed, trace = {}, {}
        for n, mod in self.solvers:
            todo = {k: q for k, q in questions.items() if k not in claimed}
            if not todo:
                break
            t = time.perf_counter()
            try:
                got = mod.solve(state, todo) or {}
            except Exception as e:  # a capability bug must never sink the request: fall through to the next one
                got = {}
                trace[n + "_error"] = f"{type(e).__name__}: {e}"[:200]
            for k, v in got.items():
                if k in todo and v:
                    claimed[k] = v
                    trace[k] = n
            if got:
                trace[n + "_ms"] = round(1000 * (time.perf_counter() - t), 1)
        return claimed, trace

    def __call__(self, state, questions):
        claimed, trace = self.decide(state, questions)
        answers = {}
        for k, q in questions.items():
            v = claimed.get(k)
            if q["type"] == "noul":
                p = v.get("noul", 0.5) if v else 0.5
                answers[k] = {"type": "noul", "noul": float(min(1.0, max(0.0, p)))}
                continue
            keys = list(q["criteria"])
            if v:
                probs = {o: float(max(0.0, v.get(o, 0.0))) for o in keys}
                z = sum(probs.values())
                probs = {o: p / z for o, p in probs.items()} if z > 0 else {o: 1 / len(keys) for o in keys}
            else:
                probs = {o: 1 / len(keys) for o in keys}
            top = max(probs.values())
            best = [o for o in keys if probs[o] >= top - 1e-12]
            choice = best[0] if len(best) == 1 else hpick(best, k + str(len(keys)))
            answers[k] = {"type": "choice", "choice": choice, "probabilities": probs}
        return {"model": "prumo", "answers": answers}, {"solved_by": trace}
