"""Capability: chess move choice from a FEN position, by a small hand-written alpha-beta search (python-chess is
used for move generation only).

Evaluation: material + piece-square tables written by hand (the textbook 'simplified evaluation function': centre
and advancement bonuses), mate/stalemate detection, quiescence on captures. No learned or tuned weights, no engine
binary, no opening book or tablebase. Iterative deepening under a fixed NODE budget per question: the result is a
pure function of the position and the options (it does not depend on the machine's speed).
"""
import re

try:
    import chess
except ImportError:  # the engine still runs without the library; chess questions then fall through
    chess = None

from prumo_rules.common import from_scores
from prumo_rules.params import P as PRM

VAL = {i + 1: int(v) for i, v in enumerate(PRM["chess.piece_values"])}
# hand-written tables, white's view, a8..h8 first row (rank 8) down to rank 1
PAWN = [int(v) for v in PRM["chess.pst.pawn"]]
KNIGHT = [int(v) for v in PRM["chess.pst.knight"]]
BISHOP = [int(v) for v in PRM["chess.pst.bishop"]]
ROOK = [int(v) for v in PRM["chess.pst.rook"]]
QUEEN = [int(v) for v in PRM["chess.pst.queen"]]
KING_MID = [int(v) for v in PRM["chess.pst.king_mid"]]
KING_END = [int(v) for v in PRM["chess.pst.king_end"]]
PST = {1: PAWN, 2: KNIGHT, 3: BISHOP, 4: ROOK, 5: QUEEN}
MATE = int(PRM["chess.mate"])
ENDGAME_MATERIAL = int(PRM["chess.endgame_material"])
CAPTURE_VICTIM_WEIGHT = int(PRM["chess.capture_victim_weight"])
CAPTURE_BONUS = int(PRM["chess.capture_bonus"])
PROMOTION_BONUS = int(PRM["chess.promotion_bonus"])
QUIESCENCE_DEPTH = int(PRM["chess.quiescence_depth"])


def _sq_index(sq, white):
    # tables are listed rank 8 first; python-chess squares are a1=0
    r, f = divmod(sq, 8)
    return (7 - r) * 8 + f if white else r * 8 + f


def evaluate(b):
    """Static score from the side to move's point of view (centipawns)."""
    s = 0
    occ_w, occ_b = b.occupied_co[True], b.occupied_co[False]
    major = 0
    for pt, bb in ((2, b.knights), (3, b.bishops), (4, b.rooks), (5, b.queens)):
        major += VAL[pt] * chess.popcount(bb)
    endgame = major <= ENDGAME_MATERIAL
    for pt, bb in ((1, b.pawns), (2, b.knights), (3, b.bishops), (4, b.rooks), (5, b.queens), (6, b.kings)):
        table = (KING_END if endgame else KING_MID) if pt == 6 else PST[pt]
        v = VAL[pt]
        for sq in chess.scan_forward(bb & occ_w):
            r, f = divmod(sq, 8)
            s += v + table[(7 - r) * 8 + f]
        for sq in chess.scan_forward(bb & occ_b):
            s -= v + table[sq]
    return s if b.turn else -s


def _order(b, moves):
    def key(m):
        k = 0
        if b.is_capture(m):
            victim = b.piece_type_at(m.to_square) or 1
            attacker = b.piece_type_at(m.from_square) or 1
            k += CAPTURE_VICTIM_WEIGHT * VAL[victim] - VAL[attacker] + CAPTURE_BONUS
        if m.promotion:
            k += PROMOTION_BONUS
        return -k
    return sorted(moves, key=key)


NODES = int(PRM["chess.nodes"])  # search nodes per question (negamax + quiescence), fixed


class Search:
    def __init__(self, budget):
        self.budget = budget
        self.nodes = 0

    def out_of_time(self):
        return self.nodes > self.budget

    def quiesce(self, b, alpha, beta, depth=0):
        if self.out_of_time():
            raise TimeoutError
        self.nodes += 1
        stand = evaluate(b)
        if stand >= beta:
            return beta
        alpha = max(alpha, stand)
        if depth > QUIESCENCE_DEPTH:
            return alpha
        caps = list(b.generate_legal_captures())
        caps += [m for m in b.generate_legal_moves(to_mask=chess.BB_BACKRANKS) if m.promotion and not b.is_capture(m)]
        for m in _order(b, caps):
            b.push(m)
            v = -self.quiesce(b, -beta, -alpha, depth + 1)
            b.pop()
            if v >= beta:
                return beta
            alpha = max(alpha, v)
        return alpha

    def negamax(self, b, depth, alpha, beta, ply):
        if self.out_of_time():
            raise TimeoutError
        self.nodes += 1
        if b.is_checkmate():
            return -MATE + ply
        if b.is_stalemate() or b.is_insufficient_material() or b.can_claim_fifty_moves() or b.is_repetition(2):
            return 0
        if depth == 0:
            return self.quiesce(b, alpha, beta)
        best = -MATE * 2
        for m in _order(b, list(b.legal_moves)):
            b.push(m)
            try:
                v = -self.negamax(b, depth - 1, -beta, -alpha, ply + 1)
            finally:
                b.pop()
            if v > best:
                best = v
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        return best


def root_scores(b, moves, budget=NODES):
    s = Search(budget)
    scores = {m: 0.0 for m in moves}
    done = None
    order = _order(b, moves)
    for depth in range(1, 8):
        cur = {}
        try:
            for m in order:
                b.push(m)
                try:
                    cur[m] = -s.negamax(b, depth - 1, -MATE * 2, MATE * 2, 1)
                finally:
                    b.pop()
        except TimeoutError:
            break
        done = cur
        order = sorted(order, key=lambda m: -cur[m])
        if max(abs(v) for v in cur.values()) > MATE // 2:
            break
    return done or scores


FEN_RE = re.compile(r"([rnbqkpRNBQKP1-8]{1,8}(?:/[rnbqkpRNBQKP1-8]{1,8}){7}) ([wb]) (-|[KQkq]{1,4}) (-|[a-h][36])(?: (\d+) (\d+))?")


def _find_fen(state, questions):
    if isinstance(state, dict) and isinstance(state.get("fen"), str):
        return state["fen"]
    texts = [state] if isinstance(state, str) else [str(v) for v in state.values()] if isinstance(state, dict) else []
    texts += [str(q.get("instructions")) for q in questions.values()]
    for t in texts:
        m = FEN_RE.search(t)
        if m:
            return m.group(0) if m.group(5) else m.group(0) + " 0 1"
    return None


def _move(b, key, desc):
    for cand in (key, desc.get("uci") if isinstance(desc, dict) else None, desc.get("san") if isinstance(desc, dict) else desc):
        if not isinstance(cand, str):
            continue
        try:
            m = chess.Move.from_uci(cand.strip())
            if m in b.legal_moves:
                return m
        except ValueError:
            pass
        try:
            return b.parse_san(cand.strip())
        except ValueError:
            pass
    return None


def solve(state, questions):
    if chess is None:
        return None
    fen = _find_fen(state, questions)
    if not fen:
        return None
    try:
        b = chess.Board(fen)
    except ValueError:
        return None
    out = {}
    for k, q in questions.items():
        if q["type"] != "choice":
            continue
        moves = {}
        for key in q["criteria"]:
            m = _move(b, key, q["criteria"][key])
            if m is None:
                moves = None
                break
            moves[key] = m
        if not moves:
            continue
        sc = root_scores(b, list(moves.values()))
        out[k] = from_scores({key: sc[m] / PRM["chess.score_scale"] for key, m in moves.items()}, temp=PRM["chess.temperature"])
    return out or None
