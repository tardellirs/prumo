"""Build `constants.safetensors`: every hand-set numeric parameter of the engine (F64, standard library only).

    python tools/make_constants.py          # rewrites ../constants.safetensors and prints the parameter count

This script is the human-readable source of the numbers; the engine itself only reads the .safetensors file
(`prumo_rules/params.py`). None of the values was fitted by gradient or estimated from corpus statistics.
Where a value was chosen on data, the comment says which (training/dev splits, never the public test).
"""
import json
import struct
from pathlib import Path

PARAMS = {
    # --- chess: material (pawn, knight, bishop, rook, queen, king) and piece-square tables (rank 8 first) ---
    "chess.piece_values": [100, 320, 330, 500, 900, 0],
    "chess.pst.pawn": [
        0, 0, 0, 0, 0, 0, 0, 0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5, 5, 10, 25, 25, 10, 5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, -5, -10, 0, 0, -10, -5, 5,
        5, 10, 10, -20, -20, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0,
    ],
    "chess.pst.knight": [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50,
    ],
    "chess.pst.bishop": [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -20, -10, -10, -10, -10, -10, -10, -20,
    ],
    "chess.pst.rook": [
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, 10, 10, 10, 10, 5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        0, 0, 0, 5, 5, 0, 0, 0,
    ],
    "chess.pst.queen": [
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20,
    ],
    "chess.pst.king_mid": [
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        20, 20, 0, 0, 0, 0, 20, 20,
        20, 30, 10, 0, 0, 10, 30, 20,
    ],
    "chess.pst.king_end": [
        -50, -40, -30, -20, -20, -30, -40, -50,
        -30, -20, -10, 0, 0, -10, -20, -30,
        -30, -10, 20, 30, 30, 20, -10, -30,
        -30, -10, 30, 40, 40, 30, -10, -30,
        -30, -10, 30, 40, 40, 30, -10, -30,
        -30, -10, 20, 30, 30, 20, -10, -30,
        -30, -30, 0, 0, 0, 0, -30, -30,
        -50, -30, -30, -30, -30, -30, -30, -50,
    ],
    "chess.mate": 100000,               # mate score
    "chess.endgame_material": 1300,     # non-pawn material at or below which the endgame king table is used
    "chess.capture_victim_weight": 10,  # move ordering: 10 x victim - attacker (MVV-LVA)
    "chess.capture_bonus": 10000,       # move ordering: captures first
    "chess.promotion_bonus": 9000,      # move ordering: promotions next
    "chess.quiescence_depth": 6,        # maximum quiescence depth
    "chess.nodes": 7000,                # search nodes per question (deterministic budget)
    "chess.score_scale": 100.0,         # centipawns -> softmax units
    "chess.temperature": 1.0,
    # --- sandbox for request code (deterministic line budget; the rest are safety nets) ---
    "sandbox.lines": 1000000,           # executed lines + calls per run
    "sandbox.wall_seconds": 1.0,
    "sandbox.cpu_seconds": 1,
    "sandbox.file_bytes": 1048576,      # RLIMIT_FSIZE
    "sandbox.address_bytes": 2147483648,  # RLIMIT_AS
    "code.mass": 0.97,                  # confidence on the option equal to the executed value / test verdict
    # --- arithmetic ---
    "arith.pool": 6,                    # at most this many stated numbers are combined
    "arith.max_terms": 4,               # at most this many numbers in one derivation
    "arith.small_pool": 5,              # pools this small may use one more term
    "arith.const_pool": 4,              # pools this small may also use one unit constant
    "arith.budget": 60000,              # binary operations per question (deterministic)
    "arith.min_terms": 2,               # partial derivations count from this many terms
    "arith.score_full": 2.0,            # option reachable from all stated numbers (and not itself stated)
    "arith.score_stated": 0.3,          # option equal to a stated number
    "arith.score_base": 0.5,            # partial derivation: base + per_term x terms
    "arith.score_per_term": 0.2,
    "arith.temperature": 0.25,
    "arith.mass": 0.95,                 # exact expression value
    # --- BM25 (standard values) ---
    "bm25.k1": 1.2,
    "bm25.b": 0.75,
    "bm25.idf_smoothing": 0.5,
    # --- retrieval ---
    "retrieval.coverage_exact": 0.5,    # graded relevance: query-term coverage for "exact"
    "retrieval.mass_exact": 0.6,
    "retrieval.mass_irrelevant": 0.6,
    "retrieval.mass_partial": 0.45,
    "retrieval.decay": 8.0,             # yes/no relevance: p = 1 - exp(-bm25 / decay)
    "retrieval.p_floor": 0.02,
    "retrieval.p_span": 0.96,
    "retrieval.p_ceiling": 0.98,
    # --- grounding (chosen on HoVer and RAGTruth train splits) ---
    "grounding.coverage": 0.7,
    "grounding.noul_low": 0.3,
    "grounding.noul_high": 0.7,
    "grounding.mass": 0.65,
    # --- consensus ---
    "consensus.temperature": 0.1,
    # --- labels (out-of-scope chosen on CLINC150 train) ---
    "labels.oos_score": 0.15,
    "labels.hit_bonus": 0.1,
    "labels.temperature": 0.15,
    "labels.temperature_oos": 0.3,
    # --- music ---
    "music.window_beats": 1.0,
    "music.bass_weight": 1.5,
    "music.outside_penalty": 0.5,
    "music.missing_penalty": 0.5,
    "music.other_score": -1.0,
    "music.temperature": 0.05,
    "music.temperature_none": 0.3,
    # --- tools (band on BFCL-style train sets + SS-dev; recency on API-Bank train; rules on When2Call train) ---
    "tools.near": 0.5,                  # per-tool yes/no: call tools scoring at least near x best
    "tools.last_weight": 4,             # the latest user turn adds this x its own BM25
    "tools.related": 4,                 # relatedness rule variant (0-4; chosen on When2Call train): 4 = shared name
                                        # prefix, or >= 2 shared description prefixes, generic tool verbs ignored
    "tools.stated_overlap": 0.5,        # a value counts as user-stated at this token overlap
    "tools.temperature": 0.5,
    "tools.mass_no_catalog": 0.85,
    "tools.mass_reply": 0.8,
    "tools.call_high": 0.85,
    "tools.call_low": 0.15,
    # --- logic, dates, puzzles ---
    "logic.mass_exact": 0.95,
    "logic.mass_inference": 0.93,
    "logic.noul_high": 0.93,
    "logic.noul_low": 0.07,
    "dates.mass": 0.93,
    "puzzles.mass": 0.95,
    "puzzles.right_angle_tolerance": 0.01,
    "puzzles.side_tolerance": 0.02,
    # --- shared defaults ---
    "common.peaked_mass": 0.9,
    "common.bool_mass": 0.95,
}


def write(path):
    header, blobs, off = {}, [], 0
    for name in sorted(PARAMS):
        v = PARAMS[name]
        vals = [float(x) for x in (v if isinstance(v, list) else [v])]
        b = struct.pack(f"<{len(vals)}d", *vals)
        header[name] = {"dtype": "F64", "shape": [len(vals)], "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    header["__metadata__"] = {"format": "prumo-constants", "note": "hand-set parameters; nothing learned"}
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    Path(path).write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))
    return sum(header[n]["shape"][0] for n in header if n != "__metadata__")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "constants.safetensors"
    print(write(out), "parameters ->", out)
