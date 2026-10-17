"""Capability: calendar arithmetic with explicitly stated dates.

Reads one anchor date from the text (formats: 2025-10-07, 10/07/2025 (US order), 7 October 2025, October 7, 2025,
Oct 7 2025), optionally re-anchored by "yesterday was <date>" / "tomorrow is <date>" / "<date> was N days ago",
and one offset (today / tomorrow / yesterday, N days|weeks|months|years ago|later|from today|before|after, N
calendar or business days counted from the day after, end-of-month clamping for months). The option equal to
the computed date wins; claimed only when exactly one option matches. Nothing relies on world knowledge
(holidays and "Christmas Eve" are not resolved).
"""
import calendar
import re
from datetime import date, timedelta

from prumo_rules.common import crit_text, main_text, opt_texts, peaked

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
MONTHS["sept"] = 9
MON = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)\.?"
NUMW = {"one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30}
DATE_PATTERNS = [
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), lambda m: (int(m[1]), int(m[2]), int(m[3]))),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"), lambda m: (int(m[3]), int(m[1]), int(m[2]))),
    (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?(?: of)? " + MON + r",? (\d{4})\b", re.I), lambda m: (int(m[3]), MONTHS[m[2].lower()], int(m[1]))),
    (re.compile(r"\b" + MON + r" (\d{1,2})(?:st|nd|rd|th)?,? (\d{4})\b", re.I), lambda m: (int(m[3]), MONTHS[m[1].lower()], int(m[2]))),
]


def find_dates(text):
    """[(start, end, date)] of every explicit date, non-overlapping, in order."""
    found = []
    for rx, conv in DATE_PATTERNS:
        for m in rx.finditer(text):
            try:
                d = date(*conv(m))
            except (ValueError, KeyError):
                continue
            if not any(s < m.end() and m.start() < e for s, e, _ in found):
                found.append((m.start(), m.end(), d))
    return sorted(found)


def parse_option(s):
    ds = find_dates(s)
    if len(ds) == 1 and len(re.sub(r"[\W_]", "", s[: ds[0][0]] + s[ds[0][1]:])) <= 12:
        return ds[0][2]
    return None


def add_months(d, n):
    mo = d.month - 1 + n
    y, mo = d.year + mo // 12, mo % 12 + 1
    if not 1 <= y <= 9999:
        raise ValueError("year")
    return date(y, mo, min(d.day, calendar.monthrange(y, mo)[1]))


def shift(d, n, unit):
    unit = unit.lower()
    if unit.startswith(("business", "working", "weekday")):
        step = 1 if n >= 0 else -1
        k = 0
        while k < abs(n):
            d += timedelta(days=step)
            if d.weekday() < 5:
                k += 1
        return d
    if unit.startswith(("calendar", "day")):
        return d + timedelta(days=n)
    if unit.startswith("week"):
        return d + timedelta(days=7 * n)
    if unit.startswith("fortnight"):
        return d + timedelta(days=14 * n)
    if unit.startswith("month"):
        return add_months(d, n)
    if unit.startswith(("year", "decade")):
        return add_months(d, 12 * n * (10 if unit.startswith("decade") else 1))
    if unit.startswith("hour"):
        return d + timedelta(days=n // 24) if n % 24 == 0 else None
    raise ValueError(unit)


NUM = r"(\d+|" + "|".join(NUMW) + r")"
UNIT = r"(calendar days?|business days?|working days?|weekdays?|days?|weeks?|fortnights?|months?|years?|decades?|hours?)(?:\(s\))?"


def _n(s):
    return int(s) if s.isdigit() else NUMW[s.lower()]


def target_date(text):
    ds = find_dates(text)
    if not ds:
        return None
    sents = re.split(r"(?<=[.?!])\s+|\n+", text.strip())
    question = next((s for s in reversed(sents) if "?" in s), sents[-1])
    qlow = question.lower()
    ctx = text[: text.rfind(question)] if question in text else text
    cds = find_dates(ctx)
    if len({d for _, _, d in cds}) != 1:
        return None
    s0, e0, anchor = cds[0]
    before = ctx[max(0, s0 - 60): s0].lower()
    after = ctx[e0: e0 + 60].lower()
    today = anchor
    if re.search(r"\byesterday (?:was|is)\b[^.]*$", before):
        today = anchor + timedelta(days=1)
    elif re.search(r"\btomorrow (?:is|will be)\b[^.]*$", before):
        today = anchor - timedelta(days=1)
    else:
        m = re.match(r"\W*(?:was|is)\s+" + NUM + r"\s+" + UNIT + r"\s+ago\b", after)
        if m:
            today = shift(anchor, _n(m.group(1)), m.group(2))
            if today is None:
                return None
        else:
            m = re.match(r"\W*(?:is|will be)\s+(?:in\s+)?" + NUM + r"\s+" + UNIT + r"\s+(?:from now|from today|later)\b", after)
            if m:
                today = shift(anchor, -_n(m.group(1)), m.group(2))
                if today is None:
                    return None
    rel = re.search(r"\bdate\s+(?:of\s+)?(today|tomorrow|yesterday)\b|\b(today|tomorrow|yesterday)'s date\b|\bwhat (?:date|day) (?:is|was|will be) (?:it )?(today|tomorrow|yesterday)\b", qlow)
    if rel:
        w = rel.group(1) or rel.group(2) or rel.group(3)
        return today + timedelta(days={"today": 0, "tomorrow": 1, "yesterday": -1}[w])
    m = re.search(NUM + r"\s+" + UNIT + r"\s+(ago|before today|from today|from now|later|after today|hence|before|after)\b", qlow)
    if m:
        n = _n(m.group(1))
        sign = -1 if m.group(3) in ("ago", "before today", "before") else 1
        return shift(today, sign * n, m.group(2))
    # a period counted from a stated date (in the context or the question): "N days after <date>", "N business days
    # from <date>", "a period of N days counted from the day after <date>"
    m = re.search(NUM + r"\s+" + UNIT, text, re.I)
    if m and len(re.findall(NUM + r"\s+" + UNIT, text, re.I)) == 1:
        n = _n(m.group(1))
        sign = -1 if re.search(r"\b(?:before|prior to|earlier)\b", text[m.end(): m.end() + 40], re.I) else 1
        return shift(anchor, sign * n, m.group(2))
    return None


def solve(state, questions):
    out = {}
    for k, q in questions.items():
        if q["type"] != "choice":
            continue
        opts = {o: parse_option(v) for o, v in opt_texts(q).items()}
        if any(v is None for v in opts.values()) or len(opts) < 2:
            continue
        text = main_text(state, None) or ""
        ins = crit_text(q.get("instructions"))
        try:
            tgt = target_date(text + ("\n" + ins if ins and find_dates(ins) or "?" in ins and "?" not in text else ""))
        except (ValueError, OverflowError):
            tgt = None
        if tgt is None:
            continue
        hits = [o for o, d in opts.items() if d == tgt]
        if len(hits) == 1:
            out[k] = peaked(list(opts), hits, 0.93)
    return out or None
