"""The engine's hand-set numeric parameters, loaded from `constants.safetensors` at the repository root.

Every number that shapes a decision lives in that file, and nothing else:
- the chess evaluation and search constants;
- the code, arithmetic and sandbox budgets;
- the BM25 parameters;
- every threshold, weight, temperature and confidence mass.

Word lists and structural constants (board geometry, calendar, the 12 pitch classes, text-length limits) stay in
code. The file is plain safetensors with F64 tensors. It is read here with the standard library only (no numpy, no
safetensors package), so the values are bit-identical to the ones the engine was measured with. The file is built
by `tools/make_constants.py`.
"""
import json
import struct
from pathlib import Path

FILE = Path(__file__).resolve().parent.parent / "constants.safetensors"


def _load(path=FILE):
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + n])
    base = 8 + n
    out = {}
    for name, t in header.items():
        if name == "__metadata__":
            continue
        if t["dtype"] != "F64":
            raise ValueError(f"{name}: expected F64, got {t['dtype']}")
        a, b = t["data_offsets"]
        vals = list(struct.unpack(f"<{(b - a) // 8}d", raw[base + a:base + b]))
        out[name] = vals[0] if t["shape"] in ([], [1]) else vals
    return out


P = _load()


def count():
    """Number of scalar parameters in the file (what the Hub reports as the model size)."""
    return sum(len(v) if isinstance(v, list) else 1 for v in P.values())
