# -*- coding: utf-8 -*-
"""
Shared constants, conversion tables, regex patterns, normalization functions,
and canonicalization logic used across the grain-price extraction pipeline.

Imported by Stages 1–4. Changes here propagate to every script automatically.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd


# ============================================================
# Section 1 — XML namespace
# ============================================================

NS = {"tei": "http://www.tei-c.org/ns/1.0"}


# ============================================================
# Section 2 — Conversion tables
# ============================================================

CUR_TO_DRACHMA = {
    "drachma": 1.0,
    "obol": 1.0 / 6.0,
    "chalkous": 1.0 / 48.0,
    "denarius": 4.0,
    "sestertius": 1.0,
    "mina": 100.0,
    "talent": 6000.0,
}

UNIT_TO_LITER = {
    "artaba": 38.8,      # approximate placeholder
    "choenix": 1.08,     # approximate placeholder
    "medimnos": 52.5,    # approximate placeholder
    "kotyle": 0.27,      # approximate placeholder
    "metretes": 39.0,    # approximate placeholder
}


# ============================================================
# Section 3 — Text normalization
# ============================================================

def strip_accents(s: str) -> str:
    """Remove combining diacritics. Handles None and non-str input."""
    if s is None:
        return ""
    s = str(s)
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def normalize_for_search(s: str) -> str:
    """NFKC → strip accents → casefold → collapse whitespace."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = strip_accents(s).casefold()
    s = " ".join(s.split())
    return s


# Backward-compatible alias used in Stage 3
norm_greek = normalize_for_search


# ============================================================
# Section 4 — Numeric helpers
# ============================================================

def to_float(x) -> float:
    """Fraction-aware scalar parser (e.g. '3/4' → 0.75)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    if re.fullmatch(r"\d+/\d+", s):
        n, d = s.split("/")
        d = int(d)
        return float(int(n) / d) if d else np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def to_num(s: pd.Series) -> pd.Series:
    """Vectorized pd.to_numeric wrapper."""
    return pd.to_numeric(s, errors="coerce")


# ============================================================
# Section 5 — Regex vocabulary (compiled patterns)
# ============================================================

GRAIN_RE = re.compile(
    r"\b(σιτ|πυρ|κριθ|ζεα|ολυρ|σταχυ|αλευρ)\w*", re.UNICODE
)

UNIT_RE = re.compile(
    r"\b(αρταβ|μεδιμν|χοινι|μετρ|κοτυλ|χοινιξ|γομφ|καλαθ)\w*", re.UNICODE
)

MONEY_RE = re.compile(
    r"\b(δραχμ|οβολ|δηναρ|μνα|ταλαντ|σεστ|ἀσσ|χαλκ)\w*", re.UNICODE
)

PRICEWORD_RE = re.compile(
    r"(τιμη|τιμ(?=[\.\)\(]))", re.UNICODE
)


# ============================================================
# Section 5b — Classification patterns (negative & positive evidence)
# ============================================================

# --- Negative evidence patterns ---

# False grain stems: compound words whose grain stem is not a commodity reference.
GRAIN_FALSE_STEMS = re.compile(
    r"\b(σιτολογ|σιτομετρ|σιταρχ|σιτοφορ|σιτοποι|σιτικ|σιτηγ|σιτηρεσ|σιτωνι"
    r"|πυρρ|πυριν)\w*",
    re.UNICODE,
)

# Tax / fiscal vocabulary
TAX_RE = re.compile(
    r"\b(φυλακιτικ|χωματικ|δημοσι|εκφορ|φορολογ|διοικησ|προσοδ|λαογραφ"
    r"|μερισμ|επιβολ)\w*",
    re.UNICODE,
)

# Administrative titles
ADMIN_RE = re.compile(
    r"\b(πρακτορ|στρατηγ|βασιλικ\w*γραμματ|κωμογραμματ|θησαυροφυλακ|γεωμετρ)\w*",
    re.UNICODE,
)

# Transport / processing fees
TRANSPORT_FEE_RE = re.compile(
    r"\b(ναυλ|φορετρ|αλεστρ|παραναυλ|κοσκινευτ|κατεργ)\w*",
    re.UNICODE,
)

# Ration / salary / wages (narrow: μισθ/τροφ removed — too common in contracts/leases)
RATION_SALARY_RE = re.compile(
    r"\b(οψων|σιτηρεσ|διατροφ|επισιτισμ)\w*",
    re.UNICODE,
)

# Loan vocabulary
LOAN_RE = re.compile(
    r"\b(ατοκ|δανε|εντοκ|τοκος)\w*",
    re.UNICODE,
)

# Seed / sowing (σπορ narrowed to σπορα/σπορο/σπορι to avoid false matches)
SEED_RE = re.compile(
    r"\b(σπερμ|σπειρ|σπορα|σπορο|σπορι)\w*",
    re.UNICODE,
)

# Accounting / list markers (only strong accounting terms; λοιπ/ομοιως are too common)
ACCOUNTING_RE = re.compile(
    r"\b(γινονται|αναλωμ)\w*",
    re.UNICODE,
)

# --- Extended positive evidence patterns ---

# Extended τιμή pattern (all case forms)
PRICEWORD_EXTENDED_RE = re.compile(
    r"\b(τιμη|τιμης|τιμην|τιμας|τιμαι|τιμων)\w*",
    re.UNICODE,
)

# Receipt-price constructions: ἀπὸ τιμ-, εἰς τιμ-, ὑπὲρ τιμ-, ἐκ τιμ-
RECEIPT_PRICE_RE = re.compile(
    r"(απο\s+τιμ|εις\s+τιμ|υπερ\s+τιμ|εκ\s+τιμ)\w*",
    re.UNICODE,
)

# Sale / purchase verb stems
SALE_VERB_RE = re.compile(
    r"\b(αγοραζ|ηγορακ|επριατ|ωνε|πωλ|πιπρασκ|πεπρακ|πρια|ονησ)\w*",
    re.UNICODE,
)

# Per-unit rate constructions (strong positive signal)
RATE_CONSTRUCTION_RE = re.compile(
    r"(ως\s+της\s+αρταβης|ανα\s+δραχμ\w*|εκ\s+δραχμ\w*|ανα\s+αρταβ\w*"
    r"|εκαστης\s+\w*\s*δραχμ\w*)",
    re.UNICODE,
)

# Per-aroura rate (land rent, NOT grain price) — weakens positive signal
PER_AROURA_RE = re.compile(
    r"\b(αρουρ)\w*",
    re.UNICODE,
)

# All negative pattern keys and their context labels.
# ACCOUNTING_RE is excluded: γίνονται/ἀνάλωμα are too common as general vocabulary;
# accounting boundaries are handled by has_intervening_boundary() in Stage 2.
NEGATIVE_PATTERNS = {
    "TAX": TAX_RE,
    "ADMIN": ADMIN_RE,
    "TRANSPORT": TRANSPORT_FEE_RE,
    "RATION": RATION_SALARY_RE,
    "LOAN": LOAN_RE,
    "SEED": SEED_RE,
}

# All positive pattern keys, their compiled regex, and base strength
POSITIVE_PATTERNS = {
    "receipt_price": (RECEIPT_PRICE_RE, 0.7),
    "rate_construction": (RATE_CONSTRUCTION_RE, 0.8),
    "priceword_ext": (PRICEWORD_EXTENDED_RE, 0.6),
    "sale_verb": (SALE_VERB_RE, 0.5),
}


def classify_grain_context(
    norm_text: str,
    grain_start: int = 0,
    grain_end: int | None = None,
    window_chars: int = 300,
) -> tuple[str, list[str], list[str], float, str]:
    """Classify the context around a grain mention as PRICE or a negative type.

    Parameters
    ----------
    norm_text : str
        Full normalized (accent-stripped, casefolded) text of the block.
    grain_start, grain_end : int
        Character span of the grain mention in *norm_text*.
    window_chars : int
        How many characters around the grain mention to inspect.

    Returns
    -------
    (context_type, positive_signals, negative_signals, signal_strength, rejection_reason)

    context_type: "PRICE" | "TAX" | "ADMIN" | "TRANSPORT" | "RATION" | "LOAN" | "SEED" | "OTHER"
    positive_signals: list of matched positive pattern keys
    negative_signals: list of matched negative pattern keys
    signal_strength: 0.0–1.0 (max of matched positive strengths)
    rejection_reason: human-readable reason when rejected, else ""
    """
    if grain_end is None:
        grain_end = grain_start + 10

    lo = max(0, grain_start - window_chars)
    hi = min(len(norm_text), grain_end + window_chars)
    window = norm_text[lo:hi]

    # Collect negative signals
    neg_signals: list[str] = []
    for label, pat in NEGATIVE_PATTERNS.items():
        if pat.search(window):
            neg_signals.append(label)

    # Collect positive signals
    pos_signals: list[str] = []
    max_strength = 0.0
    for key, (pat, strength) in POSITIVE_PATTERNS.items():
        if pat.search(window):
            pos_signals.append(key)
            if strength > max_strength:
                max_strength = strength

    # Per-aroura override: weaken positive signal if ἄρουρα is nearby
    if PER_AROURA_RE.search(window) and max_strength > 0:
        max_strength = max(0.1, max_strength - 0.4)

    # Classification logic
    has_pos = len(pos_signals) > 0
    has_neg = len(neg_signals) > 0

    if has_pos and not has_neg:
        return ("PRICE", pos_signals, neg_signals, max_strength, "")

    if has_pos and has_neg:
        if max_strength >= 0.6:
            # Strong positive overrides negative
            return ("PRICE", pos_signals, neg_signals, max_strength, "")
        else:
            # Weak positive + negative → reject using first negative label
            reason = f"weak_positive({max_strength:.1f})+negative({','.join(neg_signals)})"
            return (neg_signals[0], pos_signals, neg_signals, max_strength, reason)

    if has_neg and not has_pos:
        reason = f"no_positive+negative({','.join(neg_signals)})"
        return (neg_signals[0], pos_signals, neg_signals, 0.0, reason)

    # Neither positive nor negative
    return ("OTHER", pos_signals, neg_signals, 0.0, "")


# ============================================================
# Section 6 — Canonicalization
# ============================================================

PLACE_PATTERNS = [
    ("Oxyrhynchite", re.compile(r"oxyrh|οξυρ", re.I)),
    ("Arsinoite", re.compile(r"arsin|αρσιν|φαγιουμ|fayum", re.I)),
    ("Hermopolite", re.compile(r"hermop|ερμοπ", re.I)),
    ("Herakleopolite", re.compile(r"herakleop|ηρακλεοπ", re.I)),
    ("Memphite", re.compile(r"memph|μεμφ", re.I)),
    ("Panopolite", re.compile(r"panop|πανοπ|akhmim", re.I)),
    ("Thebaid", re.compile(r"theb|θηβ|λυκοπ|lycop", re.I)),
    ("Alexandria", re.compile(r"alex|αλεξανδρ", re.I)),
    ("Unknown", re.compile(r".*", re.I)),
]


def canon_grain(form: str) -> str:
    g = norm_greek(form)
    if not g:
        return ""
    if g.startswith("πυρ"):
        return "wheat (pyros)"
    if g.startswith("σιτ"):
        return "grain (sitos)"
    if g.startswith("κριθ"):
        return "barley (krithē)"
    if g.startswith("ζε"):
        return "spelt/zea"
    if g.startswith("ολυρ"):
        return "emmer (olyra)"
    if g.startswith("αλευρ"):
        return "flour (aleuron)"
    if g.startswith("σταχυ"):
        return "ear/spike (stachys)"
    return g


def canon_unit(unit: str) -> str:
    u = norm_greek(unit)
    if not u:
        return ""
    if u.startswith("αρταβ"):
        return "artaba"
    if u.startswith("μεδιμν"):
        return "medimnos"
    if u.startswith("χοιν"):
        return "choenix"
    if u.startswith("κοτυλ"):
        return "kotyle"
    if u.startswith("μετρ"):
        return "metretes"
    return u


def canon_currency(cur: str) -> str:
    c = norm_greek(cur)
    if not c:
        return ""
    if c.startswith("δραχ"):
        return "drachma"
    if c.startswith("οβολ"):
        return "obol"
    if c.startswith("δηναρ"):
        return "denarius"
    if c.startswith("μνα"):
        return "mina"
    if c.startswith("ταλαν"):
        return "talent"
    if c.startswith("σεστ"):
        return "sestertius"
    if c.startswith("χαλκ"):
        return "chalkous"
    if c.startswith("ασσ"):
        return "as"
    return c


def canon_place(place: str) -> str:
    p = norm_greek(place)
    if not p:
        return "Unknown"
    for label, pat in PLACE_PATTERNS:
        if pat.search(p):
            return label
    return "Unknown"
