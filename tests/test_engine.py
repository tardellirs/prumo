"""Quick checks for the release (no network, no GPU, ~20 s):

    cd pareto/zero/release && python -m unittest discover -s tests -v

- sandbox: code from a request cannot open sockets, import network modules, spawn processes, write outside its
  temporary directory or read user files; infinite loops stop on the deterministic line budget;
- determinism: identical answers (choices and probabilities, bit for bit) across two runs in separate processes
  with different PYTHONHASHSEED values;
- latency: per-request wall time on a fixed mixed sample stays far below the board's 1,000 ms limit.
"""
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from prumo_rules import sandbox  # noqa: E402


def sample_requests():
    """A fixed, synthetic mix that exercises every capability (no benchmark items)."""
    reqs = []
    reqs.append(({"code": "def f(xs):\n    return sorted(xs)[::-1][:2]", "input": "[3, 9, 1, 4]"},
                 {"a": {"type": "choice", "instructions": "Value returned by f?", "criteria": {"x": "[9, 4]", "y": "[1, 3]", "z": "[9, 3]"}}}))
    reqs.append(({"code": "def add(a, b):\n    return a + b", "tests": "def check(c):\n    assert c(2, 3) == 5"},
                 {"p": {"type": "noul", "instructions": "The code passes the tests."}}))
    reqs.append(({"fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"},
                 {"m": {"type": "choice", "instructions": "Best move?", "criteria": {"f1b5": "Bb5", "d2d4": "d4", "a2a3": "a3", "h2h4": "h4"}}}))
    tools = [{"name": "get_weather", "description": "Current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
             {"name": "convert_currency", "description": "Convert an amount between currencies", "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "to": {"type": "string"}}, "required": ["amount", "to"]}}]
    reqs.append(({"tools": tools, "question": "What's the weather like in Lisbon right now?"},
                 {"r": {"type": "choice", "instructions": "Which reply?", "criteria": {
                     "A": '{"name": "get_weather", "arguments": {"city": "Lisbon"}}', "B": "Which city do you mean?",
                     "C": "Sorry, I cannot do that with these tools.", "D": "It is sunny."}}}))
    reqs.append(({"functions": tools, "conversation": [{"role": "user", "content": "Convert 20 dollars to euros"}]},
                 {"t1": {"type": "choice", "instructions": "Call this function?\nFunction: get_weather", "criteria": {"y": "Yes, call it.", "n": "No, do not call it."}},
                  "t2": {"type": "choice", "instructions": "Call this function?\nFunction: convert_currency", "criteria": {"y": "Yes, call it.", "n": "No, do not call it."}}}))
    reqs.append(("not ( True and False ) or False is", {"b": {"type": "choice", "instructions": "Evaluate.", "criteria": {"t": "True", "f": "False"}}}))
    reqs.append(({"question": "Today is 03/01/2024. What is the date one week ago?"},
                 {"d": {"type": "choice", "instructions": "Pick the date.", "criteria": {"a": "02/23/2024", "b": "02/22/2024", "c": "03/08/2024"}}}))
    reqs.append(({"premises": "If the lamp is on, then the room is bright. The lamp is on.", "claim": "The room is bright."},
                 {"v": {"type": "choice", "instructions": "Status of the claim?", "criteria": {"follows": "Guaranteed.", "contradicted": "Ruled out.", "undetermined": "Either way."}}}))
    reqs.append(("There are three books on a shelf: a red book, a blue book, and a green book. The red book is the leftmost. The green book is to the right of the blue book.",
                 {"o": {"type": "choice", "instructions": "Which is true?", "criteria": {"a": "The blue book is the second from the left.", "b": "The green book is the second from the left.", "c": "The red book is the second from the left."}}}))
    reqs.append(({"question": "Sam has 3 boxes with 12 pencils each. How many pencils does he have?"},
                 {"n": {"type": "choice", "instructions": "Answer.", "criteria": {"a": "36", "b": "40", "c": "50", "d": "7"}}}))
    reqs.append(("Some unrelated free text about the weather in spring.",
                 {"u": {"type": "choice", "instructions": "Which is the best answer?", "criteria": {"a": "alpha", "b": "beta", "c": "gamma"}}}))
    return reqs


class Sandbox(unittest.TestCase):
    def run_code(self, code, call="f()", cands=("1",)):
        return sandbox.run({"mode": "output", "code": code, "call": call, "candidates": list(cands)})

    def test_value(self):
        r = self.run_code("def f(x):\n    return x * 2", "f(21)", ["42", "21"])
        self.assertTrue(r["ok"])
        self.assertEqual(r["matches"], [True, False])

    def test_no_network(self):
        for code in ("import socket\ndef f():\n    return 1",
                     "def f():\n    import urllib.request\n    return urllib.request.urlopen('http://example.com').status",
                     "def f():\n    __import__('_socket')\n    return 1"):
            self.assertFalse(self.run_code(code).get("ok"), code)

    def test_no_write_outside_tmp(self):
        target = Path.home() / f".zr_sandbox_probe_{os.getpid()}"
        r = self.run_code(f"def f():\n    open({str(target)!r}, 'w').write('x')\n    return 1")
        self.assertFalse(r.get("ok"))
        self.assertFalse(target.exists())
        r = self.run_code("def f():\n    open('scratch.txt', 'w').write('ok')\n    return open('scratch.txt').read()", "f()", ["'ok'"])
        self.assertTrue(r["ok"])
        self.assertEqual(r["matches"], [True])

    def test_no_read_user_files(self):
        self.assertFalse(self.run_code("def f():\n    return open('/etc/passwd').read()[:1]").get("ok"))

    def test_no_processes(self):
        for code in ("def f():\n    import os\n    return os.system('true')", "def f():\n    import os\n    return os.fork()",
                     "def f():\n    import subprocess\n    return 1"):
            self.assertFalse(self.run_code(code).get("ok"), code)

    def test_line_budget(self):
        t = time.perf_counter()
        r = self.run_code("def f():\n    while True:\n        pass")
        self.assertFalse(r.get("ok"))
        self.assertEqual(r.get("error"), "budget")
        self.assertLess(time.perf_counter() - t, sandbox.WALL + 0.5)

    def test_wall_timeout(self):
        t = time.perf_counter()
        r = self.run_code("def f():\n    return sum(range(10**11))")
        self.assertFalse(r.get("ok"))
        self.assertLess(time.perf_counter() - t, sandbox.WALL + 0.5)


DUMP = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[1] + "/tests")
from decision_index_engine import DecisionEngine
from test_engine import sample_requests
eng = DecisionEngine()
out = [eng(s, q)[0] for s, q in sample_requests()]
print(json.dumps(out, sort_keys=True))
"""


class Determinism(unittest.TestCase):
    def test_same_answers_across_processes(self):
        outs = []
        for seed in ("1", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            p = subprocess.run([sys.executable, "-c", DUMP, str(HERE)], capture_output=True, text=True, env=env, timeout=120)
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            outs.append(p.stdout)
        self.assertEqual(outs[0], outs[1])

    def test_expected_answers(self):
        from decision_index_engine import DecisionEngine
        eng = DecisionEngine()
        got = [eng(s, q)[0]["answers"] for s, q in sample_requests()]
        self.assertEqual(got[0]["a"]["choice"], "x")
        self.assertGreater(got[1]["p"]["noul"], 0.5)
        self.assertEqual(got[3]["r"]["choice"], "A")
        self.assertEqual((got[4]["t1"]["choice"], got[4]["t2"]["choice"]), ("n", "y"))
        self.assertEqual(got[5]["b"]["choice"], "t")
        self.assertEqual(got[6]["d"]["choice"], "a")
        self.assertEqual(got[7]["v"]["choice"], "follows")
        self.assertEqual(got[8]["o"]["choice"], "a")
        self.assertEqual(got[9]["n"]["choice"], "a")
        self.assertAlmostEqual(got[10]["u"]["probabilities"]["a"], 1 / 3)


class Latency(unittest.TestCase):
    def test_latency(self):
        from decision_index_engine import DecisionEngine
        eng = DecisionEngine()
        reqs = sample_requests() * 3
        ms = []
        for s, q in reqs:
            t = time.perf_counter()
            eng(s, q)
            ms.append(1000 * (time.perf_counter() - t))
        ms.sort()
        med, mean, p80 = statistics.median(ms), statistics.mean(ms), ms[int(0.8 * len(ms))]
        print(f"\nlatency ms: median {med:.1f} mean {mean:.1f} p80 {p80:.1f} max {ms[-1]:.1f}")
        self.assertLess(med, 200)
        self.assertLess(mean, 200)
        self.assertLess(p80, 300)
        self.assertLess(ms[-1], 1000)


if __name__ == "__main__":
    unittest.main()
