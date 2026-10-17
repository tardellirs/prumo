"""Capability: name the chord sounding at a target time from symbolic notes, by pitch-class template matching
(textbook rule: the chord whose tones carry most of the sounding pitch-class weight, penalising sounding
non-chord tones and absent chord tones; the lowest sounding note counts extra as the likely root/bass).
Triggered by structure: a list of notes with start/end times and options that carry pitch-class sets."""
from prumo_rules.common import crit_text, from_scores
from prumo_rules.params import P as PRM

NOTE_KEYS = ("context_notes", "notes")
TIME_KEYS = ("target_time_beats", "target_beat", "time_beats")


def _get(state, keys):
    return next((state[k] for k in keys if k in state), None)


def solve(state, questions):
    if not isinstance(state, dict):
        return None
    notes, t = _get(state, NOTE_KEYS), _get(state, TIME_KEYS)
    if not isinstance(notes, list) or t is None:
        return None
    t = float(t)
    out = {}
    for k, q in questions.items():
        if q["type"] != "choice":
            continue
        chords = {o: v["pitch_classes"] for o, v in q["criteria"].items() if isinstance(v, dict) and isinstance(v.get("pitch_classes"), list)}
        if len(chords) < 2:
            continue
        other = [o for o in q["criteria"] if o not in chords]
        none_opt = [o for o in other if "no chord" in crit_text(q["criteria"][o]).lower()]
        w = [0.0] * 12
        sounding = []
        for n in notes:
            try:
                a, b, p = float(n["start_beat"]), float(n["end_beat"]), int(n["pitch"])
            except (KeyError, TypeError, ValueError):
                continue
            ov = max(0.0, min(b, t + PRM["music.window_beats"]) - max(a, t))
            if ov > 0:
                sounding.append((p, ov))
        if not sounding:
            if none_opt:
                out[k] = from_scores({o: (1.0 if o in none_opt else 0.0) for o in q["criteria"]}, temp=PRM["music.temperature_none"])
            continue
        low = min(p for p, _ in sounding)
        for p, ov in sounding:
            w[p % 12] += ov * (PRM["music.bass_weight"] if p == low else 1.0)
        tot = sum(w)
        mean_w = tot / max(1, sum(1 for x in w if x > 0))
        sc = {}
        for o, pcs in chords.items():
            s = sorted(set(int(p) % 12 for p in pcs))
            inside = sum(w[p] for p in s)
            missing = sum(1 for p in s if w[p] == 0)
            sc[o] = (inside - PRM["music.outside_penalty"] * (tot - inside) - PRM["music.missing_penalty"] * missing * mean_w) / tot
        for o in other:
            sc[o] = PRM["music.other_score"]
        out[k] = from_scores(sc, temp=PRM["music.temperature"])
    return out or None
