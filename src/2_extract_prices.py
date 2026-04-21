#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Stage 2: grain price extractor and linker — v12 (classify-then-extract)

Architecture change from v11b:
  - Classification gate: each grain mention's context is classified before pairing.
    Negative contexts (tax, admin, ration, loan, transport, seed) are rejected.
  - Signal-first scoring: transactional evidence (τιμή, sale verbs, rate constructions)
    is required, not just a bonus. No signal → score capped at 20.
  - Grain-centered extraction: for each grain mention, find the nearest price/qty
    instead of generating all (G×P×Q) combinations.

CLI arguments
-------------
Required:
  --candidates FILE      Stage 1 candidates CSV
  --ddb-dir DIR          Path to DDB EpiDoc XML directory

Output:
  --out FILE             Linked CSV output (default: data/extracted_price_mentions.csv)
  --out-mentions FILE    Optional block-level debug CSV (default: disabled)
  --out-rejected FILE    Optional rejected-mentions CSV for auditing (default: disabled)

Extraction tuning:
  --scan-commentary      Also scan <div type='commentary'> blocks
  --require-unit         Require a unit term for each grain hit
  --window-tokens N      Token window around each grain hit (default: 80)
  --min-score FLOAT      Minimum confidence score to include in output (default: 30.0)
  --topk N               Output top-K candidates per grain hit (default: 1)

Assignment strategy:
  --global-assign        Enable global greedy assignment to prevent price reuse
  --emit-only-primary    Emit only the top-ranked pairing per grain hit
  --emit-topk-and-primary-only
                         Emit top-K plus the primary if it falls outside top-K

Processing:
  --workers N            Number of worker processes; 0 = cpu_count (default: 0)
  --chunksize N          ProcessPoolExecutor map chunk size (default: 10)
  --max-docs N           Process only first N documents; 0 = all (default: 0)
  --encoding ENC         CSV encoding (default: utf-8-sig)
  --debug                Enable verbose (DEBUG-level) logging

Usage examples
--------------
Standard extraction:
  python src/2_extract_prices.py ^
    --candidates data/candidate_documents.csv ^
    --ddb-dir "C:\research\DDB_EpiDoc_XML" ^
    --out data/extracted_price_mentions.csv ^
    --global-assign --workers 8

With rejected-mentions audit log:
  python src/2_extract_prices.py ^
    --candidates data/candidate_documents.csv ^
    --ddb-dir "C:\research\DDB_EpiDoc_XML" ^
    --out data/extracted_price_mentions.csv ^
    --out-rejected data/rejected_mentions.csv ^
    --global-assign

Notes
-----
- 𐅵 is treated as a fraction (1/2), not as currency evidence.
- Line crossings incur a mild penalty but do not block matching.
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import csv
import logging
import math
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from lxml import etree

from pipeline_shared import (
    NS,
    GRAIN_RE,
    UNIT_RE,
    MONEY_RE,
    PRICEWORD_RE,
    GRAIN_FALSE_STEMS,
    TAX_RE,
    ADMIN_RE,
    TRANSPORT_FEE_RE,
    RATION_SALARY_RE,
    LOAN_RE,
    SEED_RE,
    ACCOUNTING_RE,
    PRICEWORD_EXTENDED_RE,
    RECEIPT_PRICE_RE,
    SALE_VERB_RE,
    RATE_CONSTRUCTION_RE,
    PER_AROURA_RE,
    NEGATIVE_PATTERNS,
    POSITIVE_PATTERNS,
    classify_grain_context,
    normalize_for_search,
)


def safe_string(node) -> str:
    if node is None:
        return ""
    try:
        return node.xpath("string()").strip()
    except Exception:
        return ""


# -----------------------------
# XML traversal
# -----------------------------


def iter_blocks(div_node: etree._Element) -> Iterable[Tuple[str, etree._Element]]:
    blocks = div_node.xpath(".//tei:ab | .//tei:p | .//tei:seg | .//tei:l", namespaces=NS)
    for b in blocks:
        yield etree.QName(b).localname, b


def has_num_tag(block: etree._Element) -> Tuple[bool, List[str], List[str]]:
    """
    STRICT requirement: at least one <num value="..."> inside the block.
    Returns: (has_any, values, texts)
    """
    vals: List[str] = []
    txts: List[str] = []
    nodes = block.xpath(".//tei:num[@value]", namespaces=NS)
    if not nodes:
        return False, vals, txts

    for n in nodes:
        v = (n.get("value") or "").strip()
        if v:
            vals.append(v)
        t = safe_string(n)
        if t:
            txts.append(t)

    return (len(vals) > 0), vals, txts


LB_MARK = "__LB__"
CB_MARK = "__CB__"
PB_MARK = "__PB__"


def block_text_with_num_values(elem: etree._Element) -> str:
    """
    Convert <num value="25">κε</num> -> "25" in the output string.
    Also inject line/page/column markers as tokens:
      <lb/> -> "__LB__"
      <cb/> -> "__CB__"
      <pb/> -> "__PB__"

    Markers enable a mild cross-line penalty during scoring (no hard filtering).
    """
    pieces: List[str] = []

    def walk(node: etree._Element):
        if node.text:
            pieces.append(node.text)

        for ch in node:
            ln = etree.QName(ch).localname

            if ln == "num":
                v = (ch.get("value") or "").strip()
                if v:
                    pieces.append(v)
                else:
                    t = safe_string(ch)
                    if t:
                        pieces.append(t)

            elif ln == "lb":
                pieces.append(f" {LB_MARK} ")

            elif ln == "cb":
                pieces.append(f" {CB_MARK} ")

            elif ln == "pb":
                pieces.append(f" {PB_MARK} ")

            else:
                walk(ch)

            if ch.tail:
                pieces.append(ch.tail)

    walk(elem)
    return " ".join(" ".join(pieces).split())


# -----------------------------
# Fractions
# -----------------------------


def fmt_fraction(fr: Fraction) -> str:
    if fr.denominator == 1:
        return str(fr.numerator)
    v = float(fr)
    return f"{v:.6f}".rstrip("0").rstrip(".")


FRAC_TOKEN_RE = re.compile(r"(?P<n>\d+)\s*/\s*(?P<d>\d+)")


def parse_int_and_fracs(int_str: Optional[str], fracs_str: Optional[str]) -> str:
    total = Fraction(0, 1)
    if int_str is not None and int_str != "":
        try:
            total += Fraction(int(int_str), 1)
        except Exception:
            pass

    if fracs_str:
        for m in FRAC_TOKEN_RE.finditer(fracs_str):
            n = int(m.group("n"))
            d = int(m.group("d"))
            if d != 0:
                total += Fraction(n, d)

    return fmt_fraction(total)


# -----------------------------
# Patterns
# -----------------------------


def compile_patterns() -> Dict[str, re.Pattern]:  # noqa: C901
    """
    Core grain/unit/money/priceword patterns come from pipeline_shared.
    Extraction-specific patterns (qty, price) are local.
    Classification patterns are imported from pipeline_shared.
    """
    grain = GRAIN_RE
    unit = UNIT_RE
    money = MONEY_RE
    priceword = PRICEWORD_RE

    unit_core = r"(?:αρταβ\w*|μεδιμν\w*|χοιν\w*|μετρ\w*|κοτυλ\w*|χοινιξ\w*)"
    cur_core = r"(?:δραχμ\w*|οβολ\w*|δηναρ\w*|μνα\w*|ταλαντ\w*|σεστ\w*|ἀσσ\w*|χαλκ\w*)"

    int_plus_fracs = r"(?P<int>\d+)(?P<fracs>(?:\W+(?:\d+/\d+))*)"
    frac_only = r"(?P<fracs_only>\d+/\d+)"

    qty_num_unit = re.compile(rf"{int_plus_fracs}\W+(?P<unit>{unit_core})", re.UNICODE)
    qty_unit_num = re.compile(rf"(?P<unit>{unit_core})\W+{int_plus_fracs}", re.UNICODE)
    qty_frac_unit = re.compile(rf"{frac_only}\W+(?P<unit>{unit_core})", re.UNICODE)
    qty_unit_frac = re.compile(rf"(?P<unit>{unit_core})\W+{frac_only}", re.UNICODE)

    price_num_cur = re.compile(rf"{int_plus_fracs}\W+(?P<cur>{cur_core})", re.UNICODE)
    price_cur_num = re.compile(rf"(?P<cur>{cur_core})\W+{int_plus_fracs}", re.UNICODE)
    price_frac_cur = re.compile(rf"{frac_only}\W+(?P<cur>{cur_core})", re.UNICODE)
    price_cur_frac = re.compile(rf"(?P<cur>{cur_core})\W+{frac_only}", re.UNICODE)

    # Bug 3a: admin compound words whose grain stem is embedded inside the compound.
    grain_admin_compound = re.compile(
        r"\b(σιτολογ|σιτομετρ|σιτοσπορ|σιτοποι)\w*", re.UNICODE
    )
    # Bug 3b: standalone admin/fee words
    nearby_admin_word = re.compile(
        r"\b(φυλακιτικ|παραναυλ|αλεστρ|χωματικ|πρακτορ)\w*", re.UNICODE
    )

    return {
        "grain": grain,
        "unit": unit,
        "money": money,
        "priceword": priceword,
        "qty_num_unit": qty_num_unit,
        "qty_unit_num": qty_unit_num,
        "qty_frac_unit": qty_frac_unit,
        "qty_unit_frac": qty_unit_frac,
        "price_num_cur": price_num_cur,
        "price_cur_num": price_cur_num,
        "price_frac_cur": price_frac_cur,
        "price_cur_frac": price_cur_frac,
        "grain_admin_compound": grain_admin_compound,
        "nearby_admin_word": nearby_admin_word,
    }


# Normalized (accent-stripped, casefolded) forms of the κρίνω aorist passive stem.
# These share the κριθ- prefix with κριθή (barley) but are not grain words.
KRINO_BLOCKLIST = re.compile(
    r"^κριθ(εν|εντ\w*|εις|εισ\w+|ηναι|ητω|ωσ\w*|ησαν)$"
)


# -----------------------------
# Tokenization / positioning
# -----------------------------


@dataclass(frozen=True)
class Tokens:
    text: str
    tokens: List[str]
    starts: List[int]
    line_ids: List[int]

    def tok_at_char(self, charpos: int) -> int:
        lo, hi = 0, len(self.starts)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.starts[mid] <= charpos:
                lo = mid + 1
            else:
                hi = mid
        idx = lo - 1
        if idx < 0:
            return 0
        if idx >= len(self.tokens):
            return len(self.tokens) - 1
        return idx


def tokenize_with_positions(s: str) -> Tokens:
    toks: List[str] = []
    starts: List[int] = []
    line_ids: List[int] = []
    line = 0
    for m in re.finditer(r"\S+", s):
        tok = m.group(0)
        toks.append(tok)
        starts.append(m.start())
        if tok == LB_MARK:
            line += 1
        line_ids.append(line)
    return Tokens(text=s, tokens=toks, starts=starts, line_ids=line_ids)


# -----------------------------
# Events
# -----------------------------


@dataclass(frozen=True)
class GrainEvent:
    tok: int
    form: str
    char_start: int = 0
    char_end: int = 0


@dataclass(frozen=True)
class QtyEvent:
    tok: int
    value: str
    unit: str


@dataclass(frozen=True)
class PriceEvent:
    tok: int
    value: str
    cur: str


@dataclass(frozen=True)
class PricewordEvent:
    tok: int
    form: str


# -----------------------------
# Scoring v2 (signal-first)
# -----------------------------


def prox(d: float, s: float) -> float:
    if d >= 1e9:
        return 0.0
    return math.exp(-d / s)


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def score_candidate_v2(
    signal_strength: float,
    d_gp: int,
    d_gq: int,
    span: int,
    has_unit: bool,
    lines_crossed: int,
    has_boundary: bool,
    n_competing: int,
) -> float:
    """Signal-first scoring formula.

    Components (additive):
      proximity_base (0-40): distance-based score for grain↔price and grain↔qty
      signal_bonus (0-30): bonus proportional to transactional signal strength
      quality (0.5-1.0): multiplicative modifier for unit, lines, boundaries
      ambiguity (0.4-1.0): multiplicative penalty for competing prices

    No-signal cap: signal_strength < 0.1 → score capped at 40.
    """
    # Proximity base: 0–40 points (grain↔price proximity + grain↔qty proximity)
    prox_gp = prox(d_gp, 12)
    prox_gq = prox(d_gq, 20) if d_gq < 1e9 else 0.3
    proximity_base = 30.0 * prox_gp + 10.0 * prox_gq

    # Signal bonus: 0–30 points
    signal_bonus = 30.0 * signal_strength

    raw = proximity_base + signal_bonus

    # Quality factor: 0.5–1.0
    quality = 1.0
    if not has_unit:
        quality -= 0.15
    if lines_crossed > 0:
        quality -= 0.05 * min(lines_crossed, 5)
    if has_boundary:
        quality -= 0.2
    quality = max(0.5, quality)

    # Ambiguity factor: 0.7–1.0 (mild penalty for competing prices)
    ambiguity = max(0.7, 1.0 - 0.05 * max(0, n_competing - 1))

    score = raw * quality * ambiguity

    # No-signal cap: without transactional evidence, limit max score
    if signal_strength < 0.1:
        score = min(score, 40.0)

    return clamp(score)


# -----------------------------
# Output rows
# -----------------------------


@dataclass
class MentionRow:
    DDB_ID: str
    Mention_ID: int
    Block_Tag: str

    Title: str
    Place: str
    Date_Text: str
    Date_When: str
    Date_NotBefore: str
    Date_NotAfter: str

    Grain_Hits: str
    Unit_Hits: str
    Money_Hits: str
    Priceword_Hit: str

    Number_Hit: str
    Number_Type: str
    Num_Tag_Values: str
    Num_Tag_Texts: str

    Quantities: str
    Prices: str

    Raw_Block: str
    Value_Block: str


@dataclass
class LinkedRow:
    DDB_ID: str
    Mention_ID: int
    Grain_Index: int
    Candidate_Rank: int
    Is_Primary: str
    Block_Tag: str

    Title: str
    Place: str
    Date_Text: str
    Date_When: str
    Date_NotBefore: str
    Date_NotAfter: str

    Grain_Form: str
    Qty_Value: str
    Qty_Unit: str
    Price_Value: str
    Price_Cur: str

    Score: float
    Dist_GP: int
    Dist_GQ: int
    Span_Toks: int
    Lines_Crossed: int
    Priceword_Near: str
    Pos_Signal_Near: str

    Ambiguous: str
    AltScore: str
    Alt_Qty: str
    Alt_Price: str

    Context_Window: str

    # New v12 columns
    Context_Type: str
    Signal_Type: str
    Signal_Strength: float
    Neg_Signals: str
    Rejection_Reason: str


@dataclass
class RejectedRow:
    """Row emitted for rejected grain mentions (for auditing)."""
    DDB_ID: str
    Mention_ID: int
    Grain_Form: str
    Context_Type: str
    Signal_Strength: float
    Neg_Signals: str
    Rejection_Reason: str
    Context_Window: str


# -----------------------------
# Candidate model + ranking
# -----------------------------


@dataclass(frozen=True)
class Candidate:
    grain: GrainEvent
    price: PriceEvent
    qty: Optional[QtyEvent]
    score: float
    d_gp: int
    d_gq: int
    span: int
    lines_crossed: int
    has_priceword_near: bool
    has_pos_signal_near: bool
    has_boundary: bool
    signal_strength: float
    context_type: str
    signal_type: str
    neg_signals: str
    rejection_reason: str


def nearest_tok_distance(tok: int, events: List[int]) -> int:
    if not events:
        return int(1e9)
    return min(abs(tok - t) for t in events)


def make_context_window(value_tokens: List[str], lo_tok: int, hi_tok: int) -> str:
    out: List[str] = []
    for t in value_tokens[lo_tok:hi_tok]:
        if t == LB_MARK:
            out.append("\n")
        elif t in (CB_MARK, PB_MARK):
            out.append("\n")
        else:
            out.append(t)
    s = " ".join(out)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


# -----------------------------
# Event building
# -----------------------------


def build_events(
    pats: Dict[str, re.Pattern],
    value_norm: str,
    tok_norm: Tokens,
) -> Tuple[List[GrainEvent], List[QtyEvent], List[PriceEvent], List[PricewordEvent]]:
    grains: List[GrainEvent] = []
    qtys: List[QtyEvent] = []
    prices: List[PriceEvent] = []
    pws: List[PricewordEvent] = []

    # Bug 3a: character spans of admin compound words containing a grain stem.
    admin_compound_spans = [
        (m.start(), m.end())
        for m in pats["grain_admin_compound"].finditer(value_norm)
    ]

    # Bug 3b: token positions of standalone admin/fee words.
    nearby_admin_toks = {
        tok_norm.tok_at_char(m.start())
        for m in pats["nearby_admin_word"].finditer(value_norm)
    }

    # Additional false-stem filtering from pipeline_shared
    false_stem_spans = [
        (m.start(), m.end())
        for m in GRAIN_FALSE_STEMS.finditer(value_norm)
    ]

    for m in pats["grain"].finditer(value_norm):
        form = m.group(0)

        # Bug 1: skip κρίνω aorist passive forms
        if KRINO_BLOCKLIST.match(form):
            continue

        t = tok_norm.tok_at_char(m.start())

        # Bug 3a: skip grain hits inside an admin compound word
        if any(s <= m.start() < e for s, e in admin_compound_spans):
            continue

        # Skip grain hits inside a false-stem match
        if any(s <= m.start() < e for s, e in false_stem_spans):
            continue

        # Bug 3b: skip grain hits with an admin/fee word within 10 tokens
        if any(abs(t - at) <= 10 for at in nearby_admin_toks):
            continue

        grains.append(GrainEvent(tok=t, form=form,
                                  char_start=m.start(), char_end=m.end()))

    for m in pats["priceword"].finditer(value_norm):
        t = tok_norm.tok_at_char(m.start())
        pws.append(PricewordEvent(tok=t, form=m.group(0)))

    def add_qty(tok: int, n: str, u: str):
        qtys.append(QtyEvent(tok=tok, value=n, unit=u))

    for m in pats["qty_num_unit"].finditer(value_norm):
        t = tok_norm.tok_at_char(m.start())
        n = parse_int_and_fracs(m.groupdict().get("int"), m.groupdict().get("fracs"))
        add_qty(t, n, m.group("unit"))
    for m in pats["qty_unit_num"].finditer(value_norm):
        t = tok_norm.tok_at_char(m.start())
        n = parse_int_and_fracs(m.groupdict().get("int"), m.groupdict().get("fracs"))
        add_qty(t, n, m.group("unit"))
    for m in pats["qty_frac_unit"].finditer(value_norm):
        t = tok_norm.tok_at_char(m.start())
        n = parse_int_and_fracs(None, m.group("fracs_only"))
        add_qty(t, n, m.group("unit"))
    for m in pats["qty_unit_frac"].finditer(value_norm):
        t = tok_norm.tok_at_char(m.start())
        n = parse_int_and_fracs(None, m.group("fracs_only"))
        add_qty(t, n, m.group("unit"))

    def add_price(tok: int, n: str, c: str):
        prices.append(PriceEvent(tok=tok, value=n, cur=c))

    for m in pats["price_num_cur"].finditer(value_norm):
        t = tok_norm.tok_at_char(m.start())
        n = parse_int_and_fracs(m.groupdict().get("int"), m.groupdict().get("fracs"))
        add_price(t, n, m.group("cur"))
    for m in pats["price_cur_num"].finditer(value_norm):
        t = tok_norm.tok_at_char(m.start())
        n = parse_int_and_fracs(m.groupdict().get("int"), m.groupdict().get("fracs"))
        add_price(t, n, m.group("cur"))
    for m in pats["price_frac_cur"].finditer(value_norm):
        t = tok_norm.tok_at_char(m.start())
        n = parse_int_and_fracs(None, m.group("fracs_only"))
        add_price(t, n, m.group("cur"))
    for m in pats["price_cur_frac"].finditer(value_norm):
        t = tok_norm.tok_at_char(m.start())
        n = parse_int_and_fracs(None, m.group("fracs_only"))
        add_price(t, n, m.group("cur"))

    # Bug 2: remove price events whose extracted value coincides with a qty event
    qty_val_toks = {(q.value, q.tok) for q in qtys}
    prices = [
        p for p in prices
        if not any(p.value == qv and abs(p.tok - qt) <= 3 for qv, qt in qty_val_toks)
    ]

    return grains, qtys, prices, pws


# -----------------------------
# Classification gate
# -----------------------------


def classify_mention_context(
    value_norm: str,
    grain: GrainEvent,
    tok_norm: Tokens,
    window_tokens: int = 20,
) -> Tuple[str, List[str], List[str], float, str]:
    """Classify a grain mention's local context using pipeline_shared.classify_grain_context.

    Uses a character window derived from ±window_tokens around the grain event.
    """
    # Convert token window to approximate character window
    n_toks = len(tok_norm.tokens)
    lo_tok = max(0, grain.tok - window_tokens)
    hi_tok = min(n_toks - 1, grain.tok + window_tokens)

    if n_toks == 0:
        return ("OTHER", [], [], 0.0, "")

    lo_char = tok_norm.starts[lo_tok]
    hi_char = tok_norm.starts[hi_tok] + len(tok_norm.tokens[hi_tok]) if hi_tok < n_toks else len(value_norm)

    return classify_grain_context(
        norm_text=value_norm,
        grain_start=grain.char_start,
        grain_end=grain.char_end,
        window_chars=max(hi_char - grain.char_start, grain.char_start - lo_char, 100),
    )


# -----------------------------
# Grain-centered extraction
# -----------------------------

# Boundary tokens that signal accounting list breaks
BOUNDARY_RE = re.compile(r"\b(γινονται|λοιπ|ομοιως)\w*", re.UNICODE)


def has_intervening_boundary(tok_norm: Tokens, tok_a: int, tok_b: int) -> bool:
    """Check if any accounting boundary token lies between tok_a and tok_b."""
    lo = min(tok_a, tok_b)
    hi = max(tok_a, tok_b)
    for i in range(lo + 1, min(hi, len(tok_norm.tokens))):
        if BOUNDARY_RE.match(tok_norm.tokens[i]):
            return True
    return False


def find_nearest_price_for_grain(
    grain: GrainEvent,
    prices: List[PriceEvent],
    qtys: List[QtyEvent],
    tok_norm: Tokens,
    window_tokens: int,
) -> Optional[Tuple[PriceEvent, Optional[QtyEvent], bool]]:
    """Find the nearest price and quantity for a grain mention.

    Returns (best_price, best_qty, has_boundary) or None if no price within window.
    """
    local_prices = [p for p in prices if abs(p.tok - grain.tok) <= window_tokens]
    if not local_prices:
        return None

    # Sort by distance to grain, take nearest
    local_prices.sort(key=lambda p: abs(p.tok - grain.tok))
    best_price = local_prices[0]

    # Check for boundary between grain and price
    boundary = has_intervening_boundary(tok_norm, grain.tok, best_price.tok)

    # Find nearest quantity
    local_qtys = [q for q in qtys if abs(q.tok - grain.tok) <= window_tokens]
    best_qty: Optional[QtyEvent] = None
    if local_qtys:
        local_qtys.sort(key=lambda q: abs(q.tok - grain.tok))
        best_qty = local_qtys[0]
        # Check boundary between grain and qty too
        if has_intervening_boundary(tok_norm, grain.tok, best_qty.tok):
            best_qty = None

    return (best_price, best_qty, boundary)


# -----------------------------
# Candidate ranking helpers
# -----------------------------


def candidate_sort_key(c: Candidate) -> Tuple:
    has_qty = 1 if c.qty is not None else 0
    return (
        c.score,
        has_qty,
        -c.span,
        -c.d_gp,
        -c.price.tok,
    )


def pick_primary_global(
    ranked: Dict[int, List[Candidate]],
    reuse_score: float = 50.0,
    reuse_max_dgp: int = 12,
    reuse_max_span: int = 18,
    fallback_score: float = 40.0,
) -> Dict[int, Candidate]:
    """Global greedy assignment to reduce reuse of a single money amount across many grains."""
    all_cands: List[Tuple[int, Candidate]] = []
    for gi, lst in ranked.items():
        for c in lst:
            all_cands.append((gi, c))

    all_cands.sort(key=lambda x: candidate_sort_key(x[1]), reverse=True)

    primary: Dict[int, Candidate] = {}
    used_price_toks: set[int] = set()

    for gi, c in all_cands:
        if gi in primary:
            continue
        if c.price.tok not in used_price_toks:
            primary[gi] = c
            used_price_toks.add(c.price.tok)
        else:
            if c.score >= reuse_score and c.d_gp <= reuse_max_dgp and c.span <= reuse_max_span:
                primary[gi] = c

    # Fallback for unassigned grains
    for gi, lst in ranked.items():
        if gi in primary:
            continue
        if not lst:
            continue
        if lst[0].score >= fallback_score:
            primary[gi] = lst[0]

    return primary


# -----------------------------
# Core extraction per doc
# -----------------------------


def extract_for_doc(
    xml_path: Path,
    ddb_id: str,
    meta: Dict[str, str],
    scan_commentary: bool,
    require_unit: bool,
    window_tokens: int,
    min_score: float,
    topk: int,
    global_assign: bool,
    emit_only_primary: bool,
    emit_topk_and_primary_only: bool,
    keep_block_chars: int = 2000,
    ambiguity_delta: float = 8.0,
) -> Tuple[List[LinkedRow], List[MentionRow], List[RejectedRow]]:
    pats = WORKER_PATS or compile_patterns()

    linked: List[LinkedRow] = []
    mentions: List[MentionRow] = []
    rejected: List[RejectedRow] = []

    try:
        root = etree.parse(str(xml_path)).getroot()
    except Exception:
        return linked, mentions, rejected

    div_xpath = "//tei:div[@type='edition']"
    if scan_commentary:
        div_xpath += " | //tei:div[@type='commentary']"
    divs = root.xpath(div_xpath, namespaces=NS)

    mention_id = 0

    for div in divs:
        for tag, block in iter_blocks(div):
            raw = safe_string(block)
            if not raw:
                continue
            raw = " ".join(raw.split())
            norm = normalize_for_search(raw)

            # Gating
            if not pats["grain"].search(norm):
                continue

            has_num, num_vals, num_txts = has_num_tag(block)
            if not has_num:
                continue

            has_unit_sig = bool(pats["unit"].search(norm))
            if require_unit and not has_unit_sig:
                continue

            has_money_sig = bool(pats["money"].search(norm))
            has_priceword_sig = bool(pats["priceword"].search(norm))
            if not (has_money_sig or has_priceword_sig):
                continue

            grain_hits = sorted(set(m.group(0) for m in pats["grain"].finditer(norm)))
            unit_hits = sorted(set(m.group(0) for m in pats["unit"].finditer(norm)))
            money_hits = sorted(set(m.group(0) for m in pats["money"].finditer(norm)))
            priceword_hit = "yes" if has_priceword_sig else "no"

            # Value-rendered text + markers
            value_text = block_text_with_num_values(block)
            value_text = value_text.replace("𐅵", " 1/2 ")
            value_text = " ".join(value_text.split())
            value_norm = normalize_for_search(value_text)

            tok_norm = tokenize_with_positions(value_norm)
            tok_val = value_text.split()

            # Debug lists
            quantities_dbg: List[str] = []
            prices_dbg: List[str] = []
            seenq = set()
            seenp = set()

            def add_qty_dbg(num_s: str, unit_s: str):
                item = f"{num_s} {unit_s}"
                if item not in seenq:
                    seenq.add(item)
                    quantities_dbg.append(item)

            def add_price_dbg(num_s: str, cur_s: str):
                item = f"{num_s} {cur_s}"
                if item not in seenp:
                    seenp.add(item)
                    prices_dbg.append(item)

            for m in pats["qty_num_unit"].finditer(value_norm):
                n = parse_int_and_fracs(m.groupdict().get("int"), m.groupdict().get("fracs"))
                add_qty_dbg(n, m.group("unit"))
            for m in pats["qty_unit_num"].finditer(value_norm):
                n = parse_int_and_fracs(m.groupdict().get("int"), m.groupdict().get("fracs"))
                add_qty_dbg(n, m.group("unit"))
            for m in pats["qty_frac_unit"].finditer(value_norm):
                n = parse_int_and_fracs(None, m.group("fracs_only"))
                add_qty_dbg(n, m.group("unit"))
            for m in pats["qty_unit_frac"].finditer(value_norm):
                n = parse_int_and_fracs(None, m.group("fracs_only"))
                add_qty_dbg(n, m.group("unit"))

            for m in pats["price_num_cur"].finditer(value_norm):
                n = parse_int_and_fracs(m.groupdict().get("int"), m.groupdict().get("fracs"))
                add_price_dbg(n, m.group("cur"))
            for m in pats["price_cur_num"].finditer(value_norm):
                n = parse_int_and_fracs(m.groupdict().get("int"), m.groupdict().get("fracs"))
                add_price_dbg(n, m.group("cur"))
            for m in pats["price_frac_cur"].finditer(value_norm):
                n = parse_int_and_fracs(None, m.group("fracs_only"))
                add_price_dbg(n, m.group("cur"))
            for m in pats["price_cur_frac"].finditer(value_norm):
                n = parse_int_and_fracs(None, m.group("fracs_only"))
                add_price_dbg(n, m.group("cur"))

            grains, qtys, prices_ev, pws = build_events(pats, value_norm, tok_norm)

            mention_id += 1
            mentions.append(
                MentionRow(
                    DDB_ID=ddb_id,
                    Mention_ID=mention_id,
                    Block_Tag=tag,
                    Title=meta.get("Title", ""),
                    Place=meta.get("Place", ""),
                    Date_Text=meta.get("Date_Text", ""),
                    Date_When=meta.get("Date_When", ""),
                    Date_NotBefore=meta.get("Date_NotBefore", ""),
                    Date_NotAfter=meta.get("Date_NotAfter", ""),
                    Grain_Hits=";".join(grain_hits),
                    Unit_Hits=";".join(unit_hits),
                    Money_Hits=";".join(money_hits),
                    Priceword_Hit=priceword_hit,
                    Number_Hit=f"num:{num_vals[0]}" if num_vals else "",
                    Number_Type="num_tag",
                    Num_Tag_Values=";".join([v for v in num_vals if v]),
                    Num_Tag_Texts=";".join([t for t in num_txts if t]),
                    Quantities=";".join(quantities_dbg),
                    Prices=";".join(prices_dbg),
                    Raw_Block=raw[:keep_block_chars],
                    Value_Block=value_text[:keep_block_chars],
                )
            )

            if not grains or not prices_ev:
                continue

            # Priceword token positions (for Priceword_Near column)
            priceword_toks = [pw.tok for pw in pws]

            # --- Grain-centered extraction with classification gate ---
            ranked: Dict[int, List[Candidate]] = {}

            for gi, grain in enumerate(grains, start=1):
                # Classification gate
                ctx_type, pos_sigs, neg_sigs, sig_strength, rej_reason = \
                    classify_mention_context(value_norm, grain, tok_norm, window_tokens=20)

                # Reject only when negative signals are present and no
                # strong positive override.  OTHER (no signal) passes through
                # with reduced score (signal_strength=0 → score capped at 20
                # by score_candidate_v2).
                if ctx_type != "PRICE" and ctx_type != "OTHER" and sig_strength < 0.3:
                    # Hard reject: negative context with no/weak positive
                    lo_c = max(0, grain.tok - 8)
                    hi_c = min(len(tok_val), grain.tok + 9)
                    ctx_text = make_context_window(tok_val, lo_c, hi_c)
                    rejected.append(RejectedRow(
                        DDB_ID=ddb_id,
                        Mention_ID=mention_id,
                        Grain_Form=grain.form,
                        Context_Type=ctx_type,
                        Signal_Strength=round(sig_strength, 2),
                        Neg_Signals=",".join(neg_sigs),
                        Rejection_Reason=rej_reason if rej_reason else f"context={ctx_type}",
                        Context_Window=ctx_text[:500],
                    ))
                    continue

                # Find nearest price/qty for this grain
                result = find_nearest_price_for_grain(
                    grain, prices_ev, qtys, tok_norm, window_tokens,
                )
                if result is None:
                    continue

                best_price, best_qty, has_boundary = result

                d_gp = abs(grain.tok - best_price.tok)
                d_gq = abs(grain.tok - best_qty.tok) if best_qty else int(1e9)
                span = max(grain.tok, best_price.tok,
                           best_qty.tok if best_qty else grain.tok) - \
                       min(grain.tok, best_price.tok,
                           best_qty.tok if best_qty else grain.tok)
                lines_crossed = abs(
                    tok_norm.line_ids[grain.tok] - tok_norm.line_ids[best_price.tok]
                ) if tok_norm.line_ids else 0

                # Count competing prices near grain (within 20 tokens)
                n_competing = len([p for p in prices_ev
                                   if abs(p.tok - grain.tok) <= 20])

                score = score_candidate_v2(
                    signal_strength=sig_strength,
                    d_gp=d_gp,
                    d_gq=d_gq,
                    span=span,
                    has_unit=has_unit_sig or (best_qty is not None),
                    lines_crossed=lines_crossed,
                    has_boundary=has_boundary,
                    n_competing=n_competing,
                )

                d_pw = nearest_tok_distance(best_price.tok, priceword_toks)
                has_pw_near = (d_pw <= 8)
                has_ps_near = sig_strength > 0.0

                cand = Candidate(
                    grain=grain,
                    price=best_price,
                    qty=best_qty,
                    score=score,
                    d_gp=d_gp,
                    d_gq=d_gq if d_gq < 1e9 else -1,
                    span=span,
                    lines_crossed=lines_crossed,
                    has_priceword_near=has_pw_near,
                    has_pos_signal_near=has_ps_near,
                    has_boundary=has_boundary,
                    signal_strength=sig_strength,
                    context_type=ctx_type,
                    signal_type=",".join(pos_sigs),
                    neg_signals=",".join(neg_sigs),
                    rejection_reason=rej_reason,
                )

                if score >= min_score:
                    ranked[gi] = [cand]

            if not ranked:
                continue

            primary: Dict[int, Candidate] = {}
            if global_assign:
                primary = pick_primary_global(ranked=ranked)

            for gi, cand_list in ranked.items():
                if not cand_list:
                    continue

                prim = primary.get(gi, cand_list[0])

                # Build emission list
                if emit_only_primary:
                    try:
                        prim_rank = cand_list.index(prim) + 1
                    except ValueError:
                        prim_rank = 1
                    cand_iter = [(prim_rank, prim)]
                else:
                    K = max(1, topk)
                    cand_iter = list(enumerate(cand_list[:K], start=1))

                    if emit_topk_and_primary_only:
                        if prim not in [c for _, c in cand_iter]:
                            try:
                                prim_rank = cand_list.index(prim) + 1
                            except ValueError:
                                prim_rank = 1
                            cand_iter.append((prim_rank, prim))

                for rank, c in cand_iter:
                    is_primary = "yes" if c == prim else "no"

                    ambiguous = "no"
                    alt_score = ""
                    alt_qty = ""
                    alt_price = ""
                    if 1 <= rank < len(cand_list):
                        nxt = cand_list[rank]
                        if (c.score - nxt.score) < ambiguity_delta:
                            ambiguous = "yes"
                            alt_score = f"{nxt.score:.2f}"
                            alt_qty = f"{nxt.qty.value} {nxt.qty.unit}" if nxt.qty else ""
                            alt_price = f"{nxt.price.value} {nxt.price.cur}"

                    # Context window
                    lo = min(c.grain.tok, c.price.tok, c.qty.tok if c.qty else c.grain.tok)
                    hi = max(c.grain.tok, c.price.tok, c.qty.tok if c.qty else c.grain.tok)
                    pad = 8
                    lo2 = max(0, lo - pad)
                    hi2 = min(len(tok_val), hi + pad + 1)
                    ctx = make_context_window(tok_val, lo2, hi2)

                    linked.append(
                        LinkedRow(
                            DDB_ID=ddb_id,
                            Mention_ID=mention_id,
                            Grain_Index=gi,
                            Candidate_Rank=rank,
                            Is_Primary=is_primary,
                            Block_Tag=tag,
                            Title=meta.get("Title", ""),
                            Place=meta.get("Place", ""),
                            Date_Text=meta.get("Date_Text", ""),
                            Date_When=meta.get("Date_When", ""),
                            Date_NotBefore=meta.get("Date_NotBefore", ""),
                            Date_NotAfter=meta.get("Date_NotAfter", ""),
                            Grain_Form=c.grain.form,
                            Qty_Value=c.qty.value if c.qty else "",
                            Qty_Unit=c.qty.unit if c.qty else "",
                            Price_Value=c.price.value,
                            Price_Cur=c.price.cur,
                            Score=round(c.score, 2),
                            Dist_GP=c.d_gp,
                            Dist_GQ=c.d_gq,
                            Span_Toks=c.span,
                            Lines_Crossed=c.lines_crossed,
                            Priceword_Near="yes" if c.has_priceword_near else "no",
                            Pos_Signal_Near="yes" if c.has_pos_signal_near else "no",
                            Ambiguous=ambiguous,
                            AltScore=alt_score,
                            Alt_Qty=alt_qty,
                            Alt_Price=alt_price,
                            Context_Window=ctx[:keep_block_chars],
                            Context_Type=c.context_type,
                            Signal_Type=c.signal_type,
                            Signal_Strength=round(c.signal_strength, 2),
                            Neg_Signals=c.neg_signals,
                            Rejection_Reason=c.rejection_reason,
                        )
                    )

    return linked, mentions, rejected


# -----------------------------
# Multiprocessing worker
# -----------------------------

WORKER_PATS: Optional[Dict[str, re.Pattern]] = None
WORKER_DDB_DIR: Optional[Path] = None
WORKER_SCAN_COMMENTARY: bool = False
WORKER_REQUIRE_UNIT: bool = False
WORKER_WINDOW_TOKENS: int = 80
WORKER_MIN_SCORE: float = 25.0
WORKER_TOPK: int = 1
WORKER_GLOBAL_ASSIGN: bool = False
WORKER_EMIT_ONLY_PRIMARY: bool = False
WORKER_EMIT_TOPK_AND_PRIMARY_ONLY: bool = False


def _init_worker(
    ddb_dir: str,
    scan_commentary: bool,
    require_unit: bool,
    window_tokens: int,
    min_score: float,
    topk: int,
    global_assign: bool,
    emit_only_primary: bool,
    emit_topk_and_primary_only: bool,
):
    global WORKER_PATS, WORKER_DDB_DIR
    global WORKER_SCAN_COMMENTARY, WORKER_REQUIRE_UNIT
    global WORKER_WINDOW_TOKENS, WORKER_MIN_SCORE, WORKER_TOPK
    global WORKER_GLOBAL_ASSIGN, WORKER_EMIT_ONLY_PRIMARY, WORKER_EMIT_TOPK_AND_PRIMARY_ONLY

    WORKER_PATS = compile_patterns()
    WORKER_DDB_DIR = Path(ddb_dir)
    WORKER_SCAN_COMMENTARY = scan_commentary
    WORKER_REQUIRE_UNIT = require_unit
    WORKER_WINDOW_TOKENS = window_tokens
    WORKER_MIN_SCORE = min_score
    WORKER_TOPK = topk
    WORKER_GLOBAL_ASSIGN = global_assign
    WORKER_EMIT_ONLY_PRIMARY = emit_only_primary
    WORKER_EMIT_TOPK_AND_PRIMARY_ONLY = emit_topk_and_primary_only


def _process_one_candidate(job: Tuple[str, str, Dict[str, str]]) -> Tuple[bool, List[dict], List[dict], List[dict]]:
    """
    Worker entry point. Returns:
      (ok, linked_rows_as_dicts, mention_rows_as_dicts, rejected_rows_as_dicts)
    """
    ddb_id, rel_path, meta = job
    try:
        assert WORKER_DDB_DIR is not None
        xml_path = WORKER_DDB_DIR / rel_path
        if not xml_path.exists():
            return False, [], [], []

        linked_rows, mention_rows, rejected_rows = extract_for_doc(
            xml_path=xml_path,
            ddb_id=ddb_id,
            meta=meta,
            scan_commentary=WORKER_SCAN_COMMENTARY,
            require_unit=WORKER_REQUIRE_UNIT,
            window_tokens=WORKER_WINDOW_TOKENS,
            min_score=WORKER_MIN_SCORE,
            topk=WORKER_TOPK,
            global_assign=WORKER_GLOBAL_ASSIGN,
            emit_only_primary=WORKER_EMIT_ONLY_PRIMARY,
            emit_topk_and_primary_only=WORKER_EMIT_TOPK_AND_PRIMARY_ONLY,
        )
        return (True,
                [asdict(r) for r in linked_rows],
                [asdict(r) for r in mention_rows],
                [asdict(r) for r in rejected_rows])
    except Exception:
        return False, [], [], []


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2 v12: grain price extractor (classify-then-extract)."
    )
    parser.add_argument("--candidates", required=True, help="Stage 1 candidates CSV")
    parser.add_argument("--ddb-dir", required=True, help="Path to DDB EpiDoc XML directory")

    parser.add_argument(
        "--out",
        default="data/extracted_price_mentions.csv",
        help="Output LINKED (grain<->qty<->price) CSV",
    )
    parser.add_argument(
        "--out-mentions",
        default="",
        help="Optional output block-level debug CSV (old-style mention rows)",
    )
    parser.add_argument(
        "--out-rejected",
        default="",
        help="Optional output CSV for rejected grain mentions (audit log)",
    )

    parser.add_argument("--scan-commentary", action="store_true", help="Also scan <div type='commentary'>")
    parser.add_argument("--require-unit", action="store_true", help="Require unit term (may miss some)")
    parser.add_argument("--window-tokens", type=int, default=80, help="Token window around each grain hit")
    parser.add_argument("--min-score", type=float, default=25.0, help="Minimum confidence score to output")
    parser.add_argument("--topk", type=int, default=1, help="Output top-K candidates per grain-hit (>=1)")
    parser.add_argument("--global-assign", action="store_true", help="Global greedy assignment to reduce price reuse")
    parser.add_argument(
        "--emit-only-primary",
        action="store_true",
        help="Emit only the primary linked row per grain-hit (ignores --topk for output)",
    )
    parser.add_argument(
        "--emit-topk-and-primary-only",
        action="store_true",
        help="Emit top-K candidates per grain-hit, and also include the primary if outside top-K",
    )

    parser.add_argument("--workers", type=int, default=0, help="Number of worker processes (0=cpu_count)")
    parser.add_argument("--chunksize", type=int, default=10, help="ProcessPoolExecutor map chunksize")
    parser.add_argument("--max-docs", type=int, default=0, help="Process only N docs (0 = all)")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV encoding (default utf-8-sig for Excel)")
    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("2_extract_prices")

    cand_path = Path(args.candidates)
    ddb_dir = Path(args.ddb_dir)
    out_linked = Path(args.out)
    out_mentions = Path(args.out_mentions) if args.out_mentions else None
    out_rejected = Path(args.out_rejected) if args.out_rejected else None

    if not cand_path.exists():
        raise SystemExit(f"Candidates CSV not found: {cand_path}")
    if not ddb_dir.exists():
        raise SystemExit(f"DDB dir not found: {ddb_dir}")

    df = pd.read_csv(cand_path).fillna("")
    if not {"DDB_ID", "XML_RelPath"}.issubset(df.columns):
        raise SystemExit("Candidates CSV must contain columns: DDB_ID, XML_RelPath")

    # Build jobs
    jobs: List[Tuple[str, str, Dict[str, str]]] = []
    for _, r in df.iterrows():
        ddb_id = str(r.get("DDB_ID", "")).strip()
        rel = str(r.get("XML_RelPath", "")).strip().replace("\\", "/")
        if not ddb_id or not rel:
            continue

        meta = {
            "Title": str(r.get("Title", "")),
            "Place": str(r.get("Place", "")),
            "Date_Text": str(r.get("Date_Text", "")),
            "Date_When": str(r.get("Date_When", "")),
            "Date_NotBefore": str(r.get("Date_NotBefore", "")),
            "Date_NotAfter": str(r.get("Date_NotAfter", "")),
        }
        jobs.append((ddb_id, rel, meta))

    if args.max_docs and args.max_docs > 0:
        jobs = jobs[: args.max_docs]

    total = len(jobs)
    if total == 0:
        logger.warning("No jobs found in candidates CSV.")
        return

    # Prepare CSV writers (streaming)
    linked_fields = [f.name for f in fields(LinkedRow)]
    mention_fields = [f.name for f in fields(MentionRow)]
    rejected_fields = [f.name for f in fields(RejectedRow)]

    out_linked.parent.mkdir(parents=True, exist_ok=True)
    f_linked = out_linked.open("w", newline="", encoding=args.encoding)
    linked_writer = csv.DictWriter(f_linked, fieldnames=linked_fields)
    linked_writer.writeheader()

    f_mentions = None
    mention_writer = None
    if out_mentions is not None:
        out_mentions.parent.mkdir(parents=True, exist_ok=True)
        f_mentions = out_mentions.open("w", newline="", encoding=args.encoding)
        mention_writer = csv.DictWriter(f_mentions, fieldnames=mention_fields)
        mention_writer.writeheader()

    f_rejected = None
    rejected_writer = None
    if out_rejected is not None:
        out_rejected.parent.mkdir(parents=True, exist_ok=True)
        f_rejected = out_rejected.open("w", newline="", encoding=args.encoding)
        rejected_writer = csv.DictWriter(f_rejected, fieldnames=rejected_fields)
        rejected_writer.writeheader()

    workers = args.workers if args.workers and args.workers > 0 else None
    logger.info("Stage 2 v12 (classify-then-extract) processing %d candidate docs with workers=%s ...", total, workers or "cpu_count")

    processed = 0
    ok_docs = 0
    linked_rows_written = 0
    mention_rows_written = 0
    rejected_rows_written = 0

    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(
                str(ddb_dir),
                args.scan_commentary,
                args.require_unit,
                args.window_tokens,
                args.min_score,
                max(1, args.topk),
                args.global_assign,
                args.emit_only_primary,
                args.emit_topk_and_primary_only,
            ),
        ) as ex:
            for ok, linked_dicts, mention_dicts, rejected_dicts in ex.map(_process_one_candidate, jobs, chunksize=args.chunksize):
                processed += 1
                if ok:
                    ok_docs += 1

                for d in linked_dicts:
                    linked_writer.writerow(d)
                linked_rows_written += len(linked_dicts)

                if mention_writer is not None:
                    for d in mention_dicts:
                        mention_writer.writerow(d)
                    mention_rows_written += len(mention_dicts)

                if rejected_writer is not None:
                    for d in rejected_dicts:
                        rejected_writer.writerow(d)
                    rejected_rows_written += len(rejected_dicts)

                if processed % 200 == 0:
                    logger.info(
                        "Progress: %d/%d docs; ok=%d; linked=%d; rejected=%d",
                        processed,
                        total,
                        ok_docs,
                        linked_rows_written,
                        rejected_rows_written,
                    )
    finally:
        f_linked.close()
        if f_mentions is not None:
            f_mentions.close()
        if f_rejected is not None:
            f_rejected.close()

    logger.info(
        "Done. Docs processed: %d/%d (ok=%d). Linked rows: %d. Rejected: %d. Mentions: %d.",
        processed,
        total,
        ok_docs,
        linked_rows_written,
        rejected_rows_written,
        mention_rows_written,
    )
    logger.info("Wrote: %s", out_linked.resolve())
    if out_mentions is not None:
        logger.info("Wrote mentions: %s", out_mentions.resolve())
    if out_rejected is not None:
        logger.info("Wrote rejected: %s", out_rejected.resolve())


if __name__ == "__main__":
    main()
