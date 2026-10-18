# Prumo

**A decision engine with zero learned parameters. It computes what can be computed and abstains, by construction, from everything else.**

[![Public index](https://img.shields.io/badge/Decision%20Index%200.3-public%2020.14-2a78d6)](#results)
[![Size](https://img.shields.io/badge/size-527%20hand--set%20parameters%20%C2%B7%200%20learned-eb6834)](#what-it-is)
[![Hardware](https://img.shields.io/badge/hardware-1%20CPU%20thread-555)](#latency)
[![Deterministic](https://img.shields.io/badge/deterministic-bit--identical%20across%20OS-555)](#determinism)
[![License](https://img.shields.io/badge/license-Apache--2.0-555)](#license)

*Prumo* is Portuguese for **plumb line**: a tool that is exact without learning anything.

Prumo reads a *state* and a set of typed *questions*, the [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)
/ `systemone` wire format, and returns a probability for every option. It contains **no model, no weights, no
embeddings and no statistics fitted to a corpus**. It is a fixed pipeline of generic capabilities, each of which
recognises a *structure* in the request and solves it exactly:

- code to execute;
- a chess position;
- a catalogue of tools;
- candidates to rank;
- a logical expression;
- a date;
- an arithmetic problem.

When nothing applies, it says so with a uniform distribution.

Size is part of the result. Prumo sits at the zero-parameter end of the size/quality trade-off and puts a floor under
it: this much of the public Decision Index is reachable **without learning anything**.

## Highlights

- **Public index 20.14** on the Decision Index 0.3, from a complete run of the official kit (140,178 requests). For
  reference, the other rule-based entrant scores 1.55, and learned models of 17M and 68M parameters score 7.44 and
  13.01.
- **Exact where it applies:**
  - executes the code in CRUXEval (skill 0.997);
  - validates tool calls against their schemas (When2Call 0.70, BFCL 0.67);
  - searches chess positions;
  - solves algorithmic BIG-Bench Hard tasks.
- **Honest elsewhere.** Knowledge, commonsense and NLI benchmarks stay at chance by design. The engine never guesses
  beyond what it can compute.
- **Fast and tiny:**
  - median **0.2 ms** per request on one CPU thread, mean 5.1 ms;
  - no GPU, no download beyond ~2.7k lines of Python and one pure-Python dependency.
- **Deterministic:**
  - every budget is counted in operations, not seconds;
  - the macOS and Linux runs give bit-identical answers.

## What it is

| | |
|---|---|
| **Kind** | rule-based, symbolic; **0 learned parameters** |
| **Size** | **527** hand-set numeric parameters, all in [`model.safetensors`](model.safetensors) (the size the Hub reports); no weights, no embeddings |
| **Input / output** | Decision Index wire format: `state` + `questions` (`choice`, `noul`) → a probability for every option |
| **Runtime** | CPython ≥ 3.10, one CPU thread; dependency: [`chess==1.11.2`](https://pypi.org/project/chess/) (move generation only) |
| **Code** | ~2.7k lines (`prumo_rules/`), Apache-2.0 |
| **Language** | English (lexical parts); the exact solvers are language-agnostic |

### How it decides

Every question goes through the same capabilities, in the same order, for every request:

`code → chess → tools → retrieval → grounding → consensus → labels → music → logic → puzzles → dates → arith`

Each capability either **claims** a question and returns probabilities, or passes it on. A question that no
capability claims gets the **uniform distribution**, with a deterministic hash pick for the reported choice, so its
expected score is exactly chance. The engine never looks at benchmark names, item ids or answer tables, and it uses
one response format for every request.

| Capability | Triggered by | What it computes |
|---|---|---|
| `code` | Python with a parseable `def` plus a call, or plus tests | runs it in a sandbox (below); picks the option equal to the value, or answers pass/fail |
| `chess` | a FEN string and legal-move options | alpha-beta with quiescence; material plus textbook piece-square tables; **7,000 nodes** per question |
| `tools` | tool objects (name + description/schema) | BM25 relevance; for "which reply?" it **validates the call against the schema** (the tool exists, it is related to the request, every required value is user-stated) → call / ask / cannot |
| `retrieval` | a query plus candidates | BM25 with IDF over the request's own candidates; graded relevance from term coverage |
| `grounding` | a claim plus evidence, a response plus its source | lexical support: numbers and names must occur, and ≥ 70% of content words must be covered |
| `consensus` | participant opinions plus candidate statements | lexical centroid (mean normalised BM25 to every opinion) |
| `labels` | ≥ 8 short label options | IDF-weighted overlap between the message and each label; out-of-scope wins when nothing overlaps |
| `music` | timed notes plus pitch-class options | chord-template matching (chord tones, non-chord tones, bass) |
| `logic` | Boolean expressions; premises plus a claim; liar/truth-teller puzzles | exact evaluation; truth tables (follows / contradicted / undetermined); SAT by enumeration |
| `puzzles` | ordering constraints, swaps, moves, schedules, SVG paths, brackets, word sorting | exhaustive search or simulation; claimed only when every sentence parses and the solution is unique |
| `dates` | an explicit anchor date plus date options | calendar arithmetic (days, weeks, months with end-of-month clamp, years, business days) |
| `arith` | an expression, or a word problem with numeric options | exact fractions; which option is reachable from the stated numbers (bounded dynamic programming) |

## Results

Decision Index **0.3 public suite**, run **once** with this exact code through the official kit (`pipeline --edition
0.3`; `complete: true`, all 140,178 requests answered). The public suite is 20% of the board's Full score. The
remaining 80% (private same-skill tests and private new-domain tasks) is run by the maintainers.

**Public index 20.14** (raw 38.96). Skill per area, chance-corrected:

| Area | Skill |
|---|---:|
| Tools & Automation | 0.453 |
| Retrieval & Classification | 0.266 |
| Knowledge & Reasoning | 0.170 |
| Arts & Human Taste | 0.134 |
| Language Understanding | 0.031 |

The table below has every benchmark with skill above 0.03. Skill is chance-corrected: 0 is chance and 1 is perfect.

| Benchmark | Skill | Capability |
|---|---:|---|
| CRUXEval | **0.997** | code |
| When2Call | 0.701 | tools |
| BFCL | 0.674 | tools |
| POP909 | 0.520 | music |
| BIG-Bench Hard | 0.472 | logic / puzzles / dates / arith |
| API-Bank | 0.452 | tools |
| CLINC150 | 0.447 | labels |
| BANKING77 | 0.431 | labels |
| ToolRet | 0.394 | retrieval |
| HoVer | 0.393 | grounding |
| ChessBench | 0.333 | chess |
| Habermas | 0.246 | consensus |
| RAGTruth | 0.244 | grounding |
| BRIGHT | 0.204 | retrieval |
| GSM8K | 0.114 | arith |
| Amazon ESCI | 0.064 | retrieval |

Everything else (MMLU, MMLU-Pro, GPQA, ARC, HellaSwag, WinoGrande, ANLI, NLI4CT, VAST, CLadder, SATA-Bench,
HLE, …) is at chance, as intended.

**Public index of other submissions**, from the board, for scale:

| Submission | Learned parameters | Public index |
|---|---:|---:|
| Zev (rule-based) | 0 | 1.55 |
| **Prumo** | **0** | **20.14** |
| Bekko System One v0 17M | 17M | 7.44 |
| Bekko System One v0 68M | 68M | 13.01 |
| Bekko System One v0 400M | 400M | 22.45 |

The Full score will be the maintainers' measurement. Below the learned models, Prumo's private parts depend on how
often its structures recur. The same-skill tests are expected to keep most of the exact capabilities. The new-domain
tasks are expected to stay mostly at chance, because the engine claims almost nothing it does not recognise.

### Latency

One CPU thread, per request, over the whole public suite (Apple M2):

| Median | Mean | p80 | p99 | Max |
|---:|---:|---:|---:|---:|
| 0.17 ms | 5.1 ms | 0.59 ms | 132 ms | 523 ms |

The slowest paths are the chess search (median 118 ms, bounded by nodes) and the code sandbox (median 41 ms per
start). The board's admission limit is 1,000 ms (median, mean or p80).

### Determinism

Every budget is counted in operations, never in time:

- chess: 7,000 nodes;
- code: 10⁶ executed lines and calls;
- arithmetic: 60k operations.

The answers therefore do not depend on the machine. Two checks:

- **Same machine, different processes:** 1,100 requests re-run in a fresh process with a different `PYTHONHASHSEED`
  are bit-identical.
- **Across operating systems:** 3,370 requests (17,258 questions; 80 from each of the suite's 43 datasets) were run
  on Linux. That host was Ubuntu 24.04, x86-64, CPython 3.12.3, with no OS sandbox available. The answers are
  identical to the macOS run (Apple M2, CPython 3.12.13, `sandbox-exec`): agreement 1.0, maximum probability
  difference 0.0.

## Usage

The Decision Index kit is not on PyPI; install it from its repository.

```sh
git clone https://github.com/apolinario/decision-index && pip install -e decision-index
git clone https://huggingface.co/tardellirs/prumo && cd prumo      # or github.com/tardellirs/prumo
pip install -r requirements.txt                                     # chess==1.11.2

# the public suite, exactly as submitted
python -m decision_index pipeline --engine decision_index_engine:DecisionEngine --edition 0.3 \
    --suite-dir <decision-index>/suite-0.3 --out runs/prumo

python -m unittest discover -s tests -v     # sandbox, determinism and latency checks (~5 s)
```

In Python:

```python
from decision_index_engine import DecisionEngine

engine = DecisionEngine()

state = {"code": "def f(xs):\n    return [x * 2 for x in xs if x % 2]", "call": "f([1, 2, 3, 4, 5])"}
questions = {"output": {"type": "choice", "instructions": "What does the call return?",
                        "criteria": {"A": "[2, 6, 10]", "B": "[2, 4, 6, 8, 10]", "C": "[1, 3, 5]", "D": "[]"}}}
response, trace = engine(state, questions)
response["answers"]["output"]["choice"]   # "A"
trace                                     # {"solved_by": {"output": "code", "code_ms": 52.9}}
```

A question that no capability recognises comes back with uniform probabilities, and `trace["solved_by"]` stays empty
for it. That is the signal to send the question to a learned model. Prumo works well as a **zero-cost first stage**
in front of one.

To run without executing any code from the request, leave out the `code` capability:

```sh
--option solvers=chess,tools,retrieval,grounding,consensus,labels,music,logic,puzzles,dates,arith
```

## Safety: executing code from requests

Code from a request never runs in the engine's own process. It runs in a child `python -I -S -B`, in a fresh empty
temporary directory, with a minimal environment. The protections come in layers:

1. **OS sandbox, when the host offers one:**
   - macOS: `sandbox-exec` (no network, no writes outside the temp dir);
   - Linux: `bwrap` (all namespaces unshared, read-only root), or else `unshare -rn`.

   On hosts where unprivileged namespaces are blocked, the engine detects it and relies on layers 2–4. Ubuntu 24.04
   with default AppArmor is one such host.
2. **Resource limits:** 1 s of CPU, 2 GiB of address space, 1 MiB per file, no new processes.
3. **A CPython audit hook**, installed before the code runs. It refuses:
   - sockets and network modules;
   - `subprocess`, fork, exec and spawn;
   - `ctypes`;
   - writes outside the temp dir;
   - reads outside the Python installation and the temp dir.
4. **A deterministic budget** of 10⁶ executed lines and calls, plus a 1 s wall-clock safety net.

A run that hits a limit counts as "cannot tell", and the question falls through. The tests in `tests/test_engine.py`
check every layer on each host: no network, no writes, no reads of user files, no processes, budget and timeout.
They pass on macOS and on Linux.

## Transparency

- **No learned values.** Every number that shapes a decision is one of the **527 hand-set parameters** in
  [`model.safetensors`](model.safetensors). The engine loads them at start-up (`prumo_rules/params.py`, standard
  library only, F64, so the values are bit-exact). [`tools/make_constants.py`](tools/make_constants.py) is their
  human-readable source, with a comment on each. The file holds:
  - 454 chess values: piece values plus seven textbook piece-square tables;
  - search constants;
  - the code, arithmetic and sandbox budgets;
  - BM25 `k1 = 1.2`, `b = 0.75`;
  - every threshold, weight, temperature and confidence mass.

  The file is named `model.safetensors` so that the Hub counts it, but it is not a trained model: it is the engine's
  configuration, kept where the size can be counted. Word lists (stop
  words, phrase lists, month and number words) and structural constants (board geometry, the calendar, the 12 pitch
  classes, text-length limits) stay in code. The thresholds that could not be fixed a priori were chosen on **training/dev splits** of the source datasets and
  on our own panels, never on the public test.
- **Removed before release.** An earlier prototype scored 24.81 on the public index, but it had parsers keyed to the
  fixed wording of particular benchmarks, and a few rules inspired by reading errors on the public test. All of them
  were removed:
  - the CLadder template parser;
  - the synthetic-phishing URL heuristics;
  - BIG-Bench Hard phrase anchors and word tables;
  - the dialogue boosts of API-Bank and When2Call.

  The tool rules were rebuilt from generic principles (schema validation, relatedness, recency) and validated on
  training data and on generated paraphrases. The public suite was then run **once**, for measurement only.
- **Remaining caveat.** A few constants of the prototype were first tried on public subsets before they were frozen:
  the chess evaluation, the chord templates, the arithmetic reachability budget and the consensus scoring. They are
  generic and kept, and they are declared here.

## Limitations

- **No world knowledge, commonsense or language understanding.** MMLU-style, NLI, sentiment and stance tasks are at
  chance by design.
- **Recognition is structural.** A request whose structure the parsers do not recognise falls back to chance.
  Examples: code without a `def`, tools described only in prose, dates without an anchor. Prumo does not degrade
  gracefully toward partial understanding the way a learned model does.
- **The lexical capabilities are heuristics** (labels, grounding, retrieval, consensus). They help, but they are far
  from neural quality.
- **Executing dataset code is a risk you should evaluate.** Read the safety section, or disable `code`.

## License

Apache-2.0 for this repository. The `chess` dependency (python-chess) is GPL-3.0 and is installed separately, not
bundled.

## Citation

```bibtex
@software{prumo2026,
  title  = {Prumo: a zero-parameter rule-based decision engine},
  author = {Tardelli Stekel},
  year   = {2026},
  url    = {https://huggingface.co/tardellirs/prumo}
}
```
