#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Stage 7: Recall comparison against Harper (2016) wheat price dataset.

Converts Harper's papyrological source references to DDB_ID format and
checks which entries our pipeline (rule-based and/or LLM) recovered.

CLI arguments
-------------
  --harper FILE         Harper 2016 Excel file
  --candidates FILE     Stage 1 candidates CSV
  --rulebased FILE      Stage 2 extraction CSV
  --llm FILE            Stage 2B LLM extraction CSV (optional)
  --out FILE            Output recall report CSV
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Normalize papyrological references to DDB_ID
# ──────────────────────────────────────────────

# Roman numeral → integer
ROMAN = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6,
    "vii": 7, "viii": 8, "ix": 9, "x": 10, "xi": 11, "xii": 12,
    "xiii": 13, "xiv": 14, "xv": 15, "xvi": 16, "xvii": 17,
    "xviii": 18, "xix": 19, "xx": 20, "xxi": 21, "xxii": 22,
}


def roman_to_int(s: str) -> str:
    """Convert a Roman numeral string to integer string, or return as-is."""
    return str(ROMAN.get(s.lower(), s))


def normalize_source_to_ddb_id(ref: str) -> list[str]:
    """
    Convert a Harper-style papyrological reference to possible DDB_ID(s).

    Examples:
        "P. Mich. 2.127"      → ["p.mich;2;127"]
        "BGU 1.14"            → ["bgu;1;14"]
        "SB 20.14576.34"      → ["sb;20;14576"]
        "BGU VII 1717"        → ["bgu;7;1717"]
        "P. Oxy. 49.3513, 3516, 3518, 3519" → ["p.oxy;49;3513", ...]
        "PSI 4.281"           → ["psi;4;281"]
        "P. Cair. Isid. 11"  → ["p.cair.isid;;11"]
    """
    if pd.isna(ref) or not ref.strip():
        return []

    ref = ref.strip()

    # Handle multiple references separated by " = " (take first)
    if " = " in ref:
        ref = ref.split(" = ")[0].strip()

    # Handle multiple document numbers: "P. Oxy. 49.3513, 3516, 3518, 3519"
    # Split on comma and generate multiple IDs
    results = []

    # Remove parenthetical notes
    ref = re.sub(r"\([^)]*\)", "", ref).strip()

    # Check for comma-separated document numbers at the end
    # Pattern: base reference, then optional ", number, number, ..."
    comma_match = re.match(r"^(.+?\.\d+)(?:,\s*(\d[\d,\s]*))?$", ref)
    if comma_match:
        base_ref = comma_match.group(1)
        extra_nums = comma_match.group(2)
        results.append(_single_ref_to_ddb(base_ref))

        if extra_nums:
            # Extract the collection+volume prefix
            prefix_match = re.match(r"^(.+\.)(\d+)$", base_ref)
            if prefix_match:
                prefix = prefix_match.group(1)
                for num in re.findall(r"\d+", extra_nums):
                    results.append(_single_ref_to_ddb(prefix + num))
    else:
        results.append(_single_ref_to_ddb(ref))

    return [r for r in results if r]


def _single_ref_to_ddb(ref: str) -> str:
    """Convert a single papyrological reference to DDB_ID format."""
    ref = ref.strip()
    if not ref:
        return ""

    # Remove trailing line/column references (e.g., ".34", ".10", ".18-19")
    # But only if there's already a volume.number structure
    # This is tricky — some refs like "SB 20.14576.34" have the .34 as a sub-document

    # Lowercase everything
    s = ref.lower()

    # Remove extra spaces
    s = re.sub(r"\s+", " ", s).strip()

    # Collections WITHOUT volume numbers (use ;; in DDB_ID)
    NO_VOLUME = {
        "p.sarap", "p.berl.zill", "p.berl.frisk", "p.erl", "o.ashm",
        "p.cair.goodsp", "p.cair.isid", "p.abinn", "p.bingen", "p.worp",
        "p.fay",
    }

    # Known collection mappings (order matters — longer first)
    COLLECTION_MAP = [
        # Multi-word P. collections
        (r"^p\.\s*berl\.\s*leigh\.", "p.berl.leihg"),  # DDB spelling
        (r"^p\.\s*berl\.\s*zill\.", "p.berl.zill"),
        (r"^p\.\s*berl\.\s*frisk\b", "p.berl.frisk"),
        (r"^p\.\s*cair\.\s*goodsp\.", "p.cair.goodsp"),
        (r"^p\.\s*cair\.\s*isid\.\s*", "p.cair.isid"),
        (r"^p\.\s*cair\.\s*zen\.", "p.cair.zen"),
        (r"^p\.\s*col\.\s*", "p.col"),
        (r"^p\.\s*erl\.\s*", "p.erl"),
        (r"^p\.\s*flor\.\s*", "p.flor"),
        (r"^p\.\s*gen\.\s*2\s*;?\s*", "p.gen.2"),
        (r"^p\.\s*grenf\.\s*", "p.grenf"),
        (r"^p\.\s*iand\.\s*", "p.iand"),
        (r"^p\.\s*kellis\s*", "p.kell"),
        (r"^p\.\s*laur\.\s*", "p.laur"),
        (r"^p\.\s*lond\.\s*", "p.lond"),
        (r"^p\.\s*louvre\s*", "p.louvre"),
        (r"^p\.\s*lund\s*", "p.lund"),
        (r"^p\.\s*mich\.\s*", "p.mich"),
        (r"^p\.\s*mil\.\s*vogl\.\s*", "p.mil.vogl"),
        (r"^p\.\s*nyu\s*", "p.nyu"),
        (r"^p\.\s*oxy\.\s*", "p.oxy"),
        (r"^p\.\s*prag\.\s*varcl\.\s*", "p.prag.varcl"),
        (r"^p\.\s*prag\.\s*", "p.prag"),
        (r"^p\.\s*princ\.\s*", "p.princ"),
        (r"^p\.\s*prince\.\s*", "p.princ"),
        (r"^p\.\s*ross\.\s*goerg\.\s*", "p.ross.georg"),
        (r"^p\.\s*ryl\.\s*", "p.ryl"),
        (r"^p\.\s*sarap\.\s*", "p.sarap"),
        (r"^p\.\s*sorb\.\s*", "p.sorb"),
        (r"^p\.\s*stras\.\s*", "p.stras"),
        (r"^p\.\s*worp\s*", "p.worp"),
        (r"^p\.\s*abinn\.\s*", "p.abinn"),
        (r"^p\.\s*bingen\s*", "p.bingen"),
        (r"^p\.\s*fay\.\s*", "p.fay"),
        # Simple collections
        (r"^bgu\s+", "bgu"),
        (r"^sb\s+", "sb"),
        (r"^psi\s+", "psi"),
        (r"^spp\s+", "stud.pal"),   # SPP = Studien zur Palaeographie
        (r"^cpr\s+", "cpr"),
        (r"^o\.\s*ashm\.\s*", "o.ashm"),
    ]

    collection = None
    for pattern, replacement in COLLECTION_MAP:
        m = re.match(pattern, s)
        if m:
            rest = s[m.end():].strip()
            collection = replacement
            break

    if collection is None:
        return s  # unknown collection

    no_vol = collection in NO_VOLUME

    # Remove trailing recto/verso annotations
    rest = re.sub(r"\s+(recto|verso|r|v)\b.*$", "", rest)

    # Handle Roman numeral volume: "VII 1717"
    roman_match = re.match(
        r"^([ivxlcdm]+)\s+(\d+)", rest, re.IGNORECASE
    )
    if roman_match:
        vol = roman_to_int(roman_match.group(1))
        num = roman_match.group(2)
        return f"{collection};{vol};{num}"

    # Handle "volume.number" or just "number"
    dot_parts = rest.split(".")
    if no_vol:
        # No-volume collection: first number is document, rest are line refs
        num = dot_parts[0].strip()
        num = re.sub(r"\s.*$", "", num)
        return f"{collection};;{num}"
    elif len(dot_parts) >= 2:
        vol = dot_parts[0].strip()
        num = dot_parts[1].strip()
        # Remove trailing sub-references (line numbers etc.)
        num = re.sub(r"\s.*$", "", num)
        return f"{collection};{vol};{num}"
    elif len(dot_parts) == 1:
        # Just a number (no volume)
        num = dot_parts[0].strip()
        num = re.sub(r"\s.*$", "", num)
        return f"{collection};;{num}"

    return s


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Recall comparison against Harper (2016) wheat prices",
    )
    p.add_argument(
        "--harper",
        default="data/harper_2016_wheat_prices.xlsx",
        help="Harper 2016 Excel file (default: %(default)s)",
    )
    p.add_argument(
        "--candidates",
        default="data/candidate_documents.csv",
        help="Stage 1 candidates CSV (default: %(default)s)",
    )
    p.add_argument(
        "--rulebased",
        default="data/extracted_price_mentions.csv",
        help="Stage 2 extraction CSV (default: %(default)s)",
    )
    p.add_argument(
        "--llm",
        default=None,
        help="Stage 2B LLM extraction CSV (optional)",
    )
    p.add_argument(
        "--out",
        default="data/harper_recall_comparison.csv",
        help="Output recall report CSV (default: %(default)s)",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    # Load Harper data (sheets 2 and 3 = wheat prices)
    df2 = pd.read_excel(args.harper, sheet_name="2", dtype=str)
    df3 = pd.read_excel(args.harper, sheet_name="3", dtype=str)

    # Standardize column names
    df2 = df2.rename(columns={"Primary_Source_Reference": "Source"})
    df2["Period"] = "1-3C"
    df3["Period"] = "4C"

    # Keep only rows with source references
    df2 = df2[df2["Source"].notna()].copy()
    df3 = df3[df3["Source"].notna()].copy()

    # Combine
    keep_cols_2 = ["Source", "Date_Earliest", "Date_Latest",
                   "Drachmas_per_artabas_literal", "Nome", "Notes", "Period"]
    keep_cols_3 = ["Source", "Date_earliest", "Date_latest",
                   "Denarii_per_artabas", "Nome", "Notes", "Period"]

    df2_sub = df2[keep_cols_2].copy()
    df2_sub.columns = ["Source", "Date_Earliest", "Date_Latest",
                       "Price_Literal", "Nome", "Notes", "Period"]

    df3_sub = df3[keep_cols_3].copy()
    df3_sub.columns = ["Source", "Date_Earliest", "Date_Latest",
                       "Price_Literal", "Nome", "Notes", "Period"]

    harper = pd.concat([df2_sub, df3_sub], ignore_index=True)
    log.info("Harper wheat entries: %d (%d 1-3C + %d 4C)",
             len(harper), len(df2_sub), len(df3_sub))

    # Load our pipeline data
    cand = pd.read_csv(args.candidates, dtype=str)
    cand_ids = set(cand["DDB_ID"].unique())

    rb = pd.read_csv(args.rulebased, dtype=str)
    rb_ids = set(rb["DDB_ID"].unique())

    llm_ids = set()
    if args.llm and Path(args.llm).exists():
        llm = pd.read_csv(args.llm, dtype=str)
        # Only count windows where LLM judged is_price=True
        llm_price = llm[llm["Is_Price"].astype(str).str.lower().isin(["true", "1"])]
        llm_ids = set(llm_price["DDB_ID"].unique())
        log.info("LLM extractions (is_price=True): %d from %d docs",
                 len(llm_price), len(llm_ids))

    # Build prefix index for suffix-tolerant matching
    # e.g., "p.laur;1;11" matches "p.laur;1;11r"
    def fuzzy_match(ddb_id: str, id_set: set) -> bool:
        if ddb_id in id_set:
            return True
        # Try prefix match: ddb_id might match ddb_id + suffix (r/v/bis/etc.)
        return any(c.startswith(ddb_id) and len(c) - len(ddb_id) <= 3
                    for c in id_set)

    # Convert Harper references to DDB_IDs and check recall
    results = []
    for _, row in harper.iterrows():
        source = row["Source"]
        ddb_ids = normalize_source_to_ddb_id(source)

        in_candidates = any(fuzzy_match(d, cand_ids) for d in ddb_ids)
        in_rulebased = any(fuzzy_match(d, rb_ids) for d in ddb_ids)
        in_llm = any(fuzzy_match(d, llm_ids) for d in ddb_ids) if llm_ids else None

        results.append({
            "Harper_Source": source,
            "Period": row["Period"],
            "Date_Earliest": row["Date_Earliest"],
            "Date_Latest": row["Date_Latest"],
            "Price_Literal": row["Price_Literal"],
            "Nome": row["Nome"],
            "DDB_IDs_Matched": "; ".join(ddb_ids) if ddb_ids else "",
            "In_Candidates": in_candidates,
            "In_RuleBased": in_rulebased,
            "In_LLM": in_llm if in_llm is not None else "",
            "Notes": row.get("Notes", ""),
        })

    df_out = pd.DataFrame(results)

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info("Wrote %d rows to %s", len(df_out), out_path)

    # Summary statistics
    n_total = len(df_out)
    n_matched = (df_out["DDB_IDs_Matched"] != "").sum()
    n_cand = df_out["In_Candidates"].sum()
    n_rb = df_out["In_RuleBased"].sum()

    log.info("")
    log.info("=== RECALL SUMMARY ===")
    log.info("Harper wheat entries:        %d", n_total)
    log.info("DDB_ID resolved:             %d / %d (%.0f%%)",
             n_matched, n_total, 100 * n_matched / n_total)
    log.info("In Stage 1 candidates:       %d / %d (%.0f%%)",
             n_cand, n_total, 100 * n_cand / n_total)
    log.info("In Stage 2 rule-based:       %d / %d (%.0f%%)",
             n_rb, n_total, 100 * n_rb / n_total)

    if llm_ids:
        n_llm = df_out["In_LLM"].apply(
            lambda x: str(x).lower() in ("true", "1")
        ).sum()
        log.info("In Stage 2B LLM:             %d / %d (%.0f%%)",
                 n_llm, n_total, 100 * n_llm / n_total)

    # Show unmatched entries
    unmatched = df_out[df_out["DDB_IDs_Matched"] == ""]
    if len(unmatched) > 0:
        log.info("")
        log.info("Unmatched Harper entries (%d):", len(unmatched))
        for _, row in unmatched.iterrows():
            log.info("  %s", row["Harper_Source"])

    # Show matched but not in candidates
    missed = df_out[
        (df_out["DDB_IDs_Matched"] != "") & (~df_out["In_Candidates"])
    ]
    if len(missed) > 0:
        log.info("")
        log.info("Matched DDB_ID but NOT in candidates (%d):", len(missed))
        for _, row in missed.iterrows():
            log.info("  %s → %s", row["Harper_Source"], row["DDB_IDs_Matched"])


if __name__ == "__main__":
    main()
