#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Stage 6: Build gold-standard annotation set for CH141 comparison.

Samples text windows from Stage 2B (LLM extraction dry-run output) and
pre-fills rule-based results from Stage 2 for the same documents.
The resulting CSV has three column groups:
  - Document metadata + text window (shared)
  - Human judgment columns (to be filled by annotator)
  - Rule-based extraction results (pre-filled from Stage 2)
  - LLM extraction results (to be filled after LLM run)

Sampling is stratified by:
  - Whether the rule-based method found a price in this document
  - Score bin (for documents with rule-based results)
  - Random selection within each stratum

CLI arguments
-------------
  --windows FILE        Stage 2B dry-run CSV (default: data/extracted_price_mentions_llm.csv)
  --rulebased FILE      Stage 2 extraction CSV (default: data/extracted_price_mentions.csv)
  --old-assessment FILE Previous assessment with labels (optional, for pre-filling)
  --out FILE            Output gold-standard CSV
  --n-sample N          Total sample size (default: 150)
  --seed N              Random seed (default: 42)
"""
from __future__ import annotations

import argparse
import logging
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


# Labels from the old assessment document (sample_assessment.md)
# Maps (DDB_ID, Grain_Form_prefix) -> (label, brief reason)
OLD_LABELS = {
    # Bin 00-10
    ("p.thmouis;1;1", "κριθεν", 48): "FP",
    ("sb;16;13003", "σιτικησ", None): "FP",
    ("p.cair.isid;;41", "σιτολογοι", None): "FP",
    ("p.herakl.bank;;2", "σιτολο", None): "FP",
    ("p.tebt;3.2;847", "κριθησ", None): "FP",
    ("p.hamb;1;86", "σιτον", None): "FP",
    ("p.berl.salmen;;9", "σιτοσ", None): "FP",
    ("bgu;15;2520", "σιτικ", None): "FP",
    ("p.graux;3;30col3", "πυρου", None): "TP",
    ("p.oxy;49;3518", "πυρ", None): "ME",
    # Bin 11-20
    ("p.thmouis;1;1", "κριθεν", 23): "FP",
    ("p.tebt;1;61b", "πυρου", None): "FP",
    ("p.lond;2;171A", "σιτι", None): "BORDER",
    ("sb;12;10947", "πυρου", 1): "FP",
    ("p.fay;;11", "πυρων", None): "TP",
    ("p.cair.zen;3;59326", "σιτου", None): "ME",
    ("p.gen.2;1;28", "πυρου", None): "TP",
    ("p.oxy;51;3628", "σιτου", None): "ME",
    ("p.frankf;;1", "σιτικον", None): "TP",
    ("p.mil.vogl;1;28", "κριθ", None): "TP",
    # Bin 21-30
    ("p.oxy;8;1158", "σιτι", None): "FP",
    ("p.oxy;19;2241", "πυρο", None): "FP",
    ("p.tebt;1;79", "πυρωι", None): "FP",
    ("p.stras;6;559", "σιτου", None): "TP",
    ("p.berl.frisk;;1", "κριθ", 4): "ME",
    ("p.dryton;1;40", "πυρου", None): "BORDER",
    ("psi;4;388", "σιτου", None): "FP",
    ("p.oxy;57;3906", "πυρου", None): "FP",
    ("p.lond;3;1212r", "πυρου", None): "ME",
    ("sb;26;16634", "σιτου", None): "BORDER",
    # Bin 31-40
    ("p.princ;2;54", "πυρου", None): "FP",
    ("p.petr;3;109", "πυρου", None): "FP",
    ("p.vind.pher;;1", "πυρου", None): "FP",
    ("p.oxy;49;3516", "πυρου", None): "ME",
    ("sb;26;16460", "πυρου", None): "FP",
    ("o.narm;;58", "κριθη", None): "BORDER",
    ("bgu;6;1228", "σιτον", None): "TP",
    ("p.wisc;2;80", "πυρο", None): "ME",
    ("p.abinn;;73", "κριθησ", None): "BORDER",
    ("bgu;1;269", "σιτικ", None): "FP",
    # Bin 41-50
    ("p.oxy;16;1912", "σιτ", None): "FP",
    ("o.heid;;37", "πυρου", None): "TP",
    ("p.cair.zen;4;59647", "πυρων", None): "FP",
    ("p.stras;8;767", "σιτομετρια", None): "FP",
    ("p.oxy;85;5516", "πυρου", None): "ME",
    ("o.wilck;;694", "πυρου", None): "TP",
    ("chr.wilck;;370", "κριθ", None): "ME",
    ("o.ashm;;9", "σιτου", None): "TP",
    ("p.oxy;4;736", "πυρου", None): "FP",
    ("p.thmouis;1;1", "πυρου", 52): "FP",
    # Bin 51-60
    ("p.cair.zen;1;59004", "σιτοποιωι", None): "BORDER",
    ("p.lond;7;2074", "σιτοσ", None): "FP",
    ("p.graux;3;30col8", "πυρου", None): "TP",
    ("o.fay;;46", "πυρου", None): "FP",
    ("p.oxy;24;2421", "κριθ", None): "ME",
    ("sb;12;10947", "πυρου", 12): "FP",
    ("sb;20;14197", "πυρου", None): "ME",
    ("p.cair.zen;2;59269", "πυρων", None): "BORDER",
    ("p.sarap;;74", "πυρου", None): "ME",
    ("p.berl.frisk;;1", "πυρου", 9): "ME",
    # Bin 61-70
    ("bgu;16;2577r", "πυρων", None): "FP",
    ("p.tebt;1;120", "πυρου", 3): "BORDER",
    ("p.grenf;1;51", "πυρου", None): "ME",
    ("p.mich;5;238r", "πυρου", None): "ME",
    ("p.cair.goodsp;;30", "πυρου", None): "TP",
    ("p.oxy;14;1650", "πυρου", None): "ME",
    ("p.oslo;3;194", "πυρου", None): "ME",
    ("p.petr;3;91", "πυρων", None): "TP",
    ("p.tebt;1;120", "πυρου", 1): "ME",
    ("p.heid;6;383", "πυρ", None): "TP",
    # Bin 71-80
    ("o.cair;;139", "πυρου", None): "BORDER",
    ("p.hib;1;99", "πυρων", None): "TP",
    ("sb;18;13118", "πυρου", None): "ME",
    ("p.lond;3;1203", "πυρου", None): "FP",
    ("bgu;16;2611", "πυρου", None): "TP",
    ("o.trim;1;36", "πυρου", None): "ME",
    ("p.bour;;18", "πυρου", None): "ME",
    ("o.edfou;1;26", "πυρου", None): "TP",
    ("bgu;3;800", "πυρου", None): "TP",
    ("p.oslo;3;131", "πυρου", None): "FP",
}


def lookup_old_label(ddb_id: str, grain_form: str) -> str:
    """Look up a label from the old assessment by DDB_ID + grain form prefix."""
    # Try exact match first
    for (did, gf, _mid), label in OLD_LABELS.items():
        if did == ddb_id and grain_form.startswith(gf[:4]):
            return label
    return ""


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build gold-standard annotation CSV for CH141 comparison",
    )
    p.add_argument(
        "--windows",
        default="data/extracted_price_mentions_llm.csv",
        help="Stage 2B windows CSV (default: %(default)s)",
    )
    p.add_argument(
        "--rulebased",
        default="data/extracted_price_mentions.csv",
        help="Stage 2 rule-based CSV (default: %(default)s)",
    )
    p.add_argument(
        "--out",
        default="data/gold_standard_annotation.csv",
        help="Output gold-standard CSV (default: %(default)s)",
    )
    p.add_argument(
        "--n-sample", type=int, default=150,
        help="Total sample size (default: %(default)s)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: %(default)s)",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    # Load Stage 2B windows (from dry-run)
    win_path = Path(args.windows)
    if not win_path.exists():
        log.error("Windows CSV not found: %s", win_path)
        sys.exit(1)
    df_win = pd.read_csv(win_path, dtype=str)
    log.info("Loaded %d windows from %s", len(df_win), win_path)

    # Load Stage 2 rule-based results
    rb_path = Path(args.rulebased)
    if not rb_path.exists():
        log.error("Rule-based CSV not found: %s", rb_path)
        sys.exit(1)
    df_rb = pd.read_csv(rb_path, dtype=str)
    log.info("Loaded %d rule-based extractions from %s", len(df_rb), rb_path)

    # Build a lookup of rule-based results by DDB_ID
    rb_docs = set(df_rb["DDB_ID"].unique())

    # Stratify windows:
    # Stratum A: windows from documents WHERE rule-based found a price (overlap)
    # Stratum B: windows from documents WHERE rule-based found nothing (LLM-only)
    df_win["In_RB"] = df_win["DDB_ID"].isin(rb_docs)
    df_win["Has_Money"] = df_win["Has_Money_Cue"].astype(str).str.lower() == "true"

    stratum_a = df_win[df_win["In_RB"]].copy()
    stratum_b = df_win[~df_win["In_RB"]].copy()

    log.info(
        "Stratum A (overlap with rule-based): %d windows from %d docs",
        len(stratum_a), stratum_a["DDB_ID"].nunique(),
    )
    log.info(
        "Stratum B (LLM-only candidates): %d windows from %d docs",
        len(stratum_b), stratum_b["DDB_ID"].nunique(),
    )

    # Sampling allocation:
    # - 60% from stratum A (overlap — enables direct comparison)
    # - 40% from stratum B (LLM-only — tests LLM's independent detection)
    # Within each stratum, prioritize windows with money cues
    n_a = int(args.n_sample * 0.6)
    n_b = args.n_sample - n_a

    n_a = min(n_a, len(stratum_a))
    n_b = min(n_b, len(stratum_b))

    # Within stratum A, sub-stratify: 70% with money cue, 30% without
    a_money = stratum_a[stratum_a["Has_Money"]]
    a_nomoney = stratum_a[~stratum_a["Has_Money"]]
    n_a_money = min(int(n_a * 0.7), len(a_money))
    n_a_nomoney = min(n_a - n_a_money, len(a_nomoney))

    # Within stratum B, sub-stratify similarly
    b_money = stratum_b[stratum_b["Has_Money"]]
    b_nomoney = stratum_b[~stratum_b["Has_Money"]]
    n_b_money = min(int(n_b * 0.7), len(b_money))
    n_b_nomoney = min(n_b - n_b_money, len(b_nomoney))

    # Sample
    rng = args.seed
    samples = []

    for subset, n, label in [
        (a_money, n_a_money, "A_money"),
        (a_nomoney, n_a_nomoney, "A_nomoney"),
        (b_money, n_b_money, "B_money"),
        (b_nomoney, n_b_nomoney, "B_nomoney"),
    ]:
        if n > 0 and len(subset) > 0:
            s = subset.sample(n=min(n, len(subset)), random_state=rng)
            s = s.copy()
            s["Sample_Stratum"] = label
            samples.append(s)
            log.info("  Sampled %d from %s (%d available)", len(s), label, len(subset))

    df_sample = pd.concat(samples, ignore_index=True)
    log.info("Total sampled: %d windows", len(df_sample))

    # Pre-fill rule-based results for stratum A rows
    # Match by DDB_ID (there may be multiple RB rows per doc — take the primary)
    rb_primary = df_rb[df_rb["Is_Primary"].astype(str).str.lower() == "yes"].copy()
    rb_lookup = {}
    for _, row in rb_primary.iterrows():
        did = row["DDB_ID"]
        if did not in rb_lookup:
            rb_lookup[did] = row

    rb_cols = []
    for _, row in df_sample.iterrows():
        did = row["DDB_ID"]
        rb_row = rb_lookup.get(did)
        if rb_row is not None:
            rb_cols.append({
                "RB_Grain_Form": rb_row.get("Grain_Form", ""),
                "RB_Qty_Value": rb_row.get("Qty_Value", ""),
                "RB_Qty_Unit": rb_row.get("Qty_Unit", ""),
                "RB_Price_Value": rb_row.get("Price_Value", ""),
                "RB_Price_Cur": rb_row.get("Price_Cur", ""),
                "RB_Score": rb_row.get("Score", ""),
                "RB_Priceword_Near": rb_row.get("Priceword_Near", ""),
                "RB_Context_Type": rb_row.get("Context_Type", ""),
                "RB_Signal_Type": rb_row.get("Signal_Type", ""),
                "RB_Signal_Strength": rb_row.get("Signal_Strength", ""),
                "RB_Neg_Signals": rb_row.get("Neg_Signals", ""),
            })
        else:
            rb_cols.append({
                "RB_Grain_Form": "",
                "RB_Qty_Value": "",
                "RB_Qty_Unit": "",
                "RB_Price_Value": "",
                "RB_Price_Cur": "",
                "RB_Score": "",
                "RB_Priceword_Near": "",
                "RB_Context_Type": "",
                "RB_Signal_Type": "",
                "RB_Signal_Strength": "",
                "RB_Neg_Signals": "",
            })

    df_rb_extra = pd.DataFrame(rb_cols)

    # Pre-fill old labels where possible
    old_labels = []
    for _, row in df_sample.iterrows():
        label = lookup_old_label(row["DDB_ID"], str(row.get("Grain_Form", "")))
        old_labels.append(label)

    # Build output
    df_out = pd.DataFrame()

    # Section 1: Document metadata + window
    df_out["Sample_ID"] = range(1, len(df_sample) + 1)
    df_out["Sample_Stratum"] = df_sample["Sample_Stratum"].values
    df_out["DDB_ID"] = df_sample["DDB_ID"].values
    df_out["Title"] = df_sample["Title"].values
    df_out["Place"] = df_sample["Place"].values
    df_out["Date_NotBefore"] = df_sample["Date_NotBefore"].values
    df_out["Date_NotAfter"] = df_sample["Date_NotAfter"].values
    df_out["Grain_Form"] = df_sample["Grain_Form"].values
    df_out["Context_Window"] = df_sample["Context_Window"].values
    df_out["Has_Money_Cue"] = df_sample["Has_Money_Cue"].values
    df_out["Has_Unit_Cue"] = df_sample["Has_Unit_Cue"].values
    df_out["Has_Number"] = df_sample["Has_Number"].values

    # Section 2: Human judgment (to be filled by annotator)
    df_out["Human_Label"] = old_labels  # pre-filled from old assessment where possible
    df_out["Human_Commodity"] = ""
    df_out["Human_Qty"] = ""
    df_out["Human_Unit"] = ""
    df_out["Human_Price"] = ""
    df_out["Human_Currency"] = ""
    df_out["Human_Notes"] = ""

    # Section 3: Rule-based results (pre-filled)
    for col in df_rb_extra.columns:
        df_out[col] = df_rb_extra[col].values

    # Section 4: LLM results (to be filled after LLM run)
    df_out["LLM_Is_Price"] = ""
    df_out["LLM_Confidence"] = ""
    df_out["LLM_Commodity"] = ""
    df_out["LLM_Qty"] = ""
    df_out["LLM_Unit"] = ""
    df_out["LLM_Price"] = ""
    df_out["LLM_Currency"] = ""
    df_out["LLM_Reasoning"] = ""

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info("Wrote %d rows to %s", len(df_out), out_path)

    # Summary
    n_prefilled = sum(1 for l in old_labels if l)
    n_overlap = (df_out["RB_Score"] != "").sum()
    log.info("Summary:")
    log.info("  Total samples: %d", len(df_out))
    log.info("  With rule-based results: %d", n_overlap)
    log.info("  With pre-filled labels (from old assessment): %d", n_prefilled)
    log.info("  Remaining to annotate: %d", len(df_out) - n_prefilled)


if __name__ == "__main__":
    main()
