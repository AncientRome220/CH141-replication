# -*- coding: utf-8 -*-
"""
Stage 3: cleaning + analysis pipeline

Takes the linked extraction CSV from Stage 2 and performs canonicalization,
unit/currency conversion, date parsing, quality tiering, robust outlier
flagging, trend computation, price index construction, and generates a
manual-review queue.

CLI arguments
-------------
Input / output:
  --input, -i FILE       Input CSV from Stage 2 (default: data/extracted_price_mentions.csv)
  --outdir, -o DIR       Output directory (default: outputs/cleaned_data_latest)

Filtering:
  --min-score FLOAT      Minimum score for the main filtered dataset (default: 60)
  --drop-ambiguous       Drop ambiguous rows from the main filtered dataset

Binning and trends:
  --year-bin N           Year bin width in years (default: 25)
  --smooth-window N      Rolling median smoothing window in bins (default: 3)
  --index-baseline S E   Price-index baseline period, inclusive (default: 50 99)

Outlier detection:
  --robust-z FLOAT       MAD z-score threshold for group outlier flagging (default: 4.0)

Review queue:
  --review-top-n N       Maximum rows in the manual review queue (default: 600)

Plotting:
  --top-places N         Number of top places to plot (default: 6)
  --min-n-bin N          Minimum observations per bin for trend plots (default: 5)
  --no-plots             Skip generating matplotlib plots

Outputs (into --outdir):
  PRIMARY_cleaned.csv                         All primary rows with canonicalized fields
  MAIN_filtered.csv                           Filtered by --min-score, with outlier flags
  manual_review_queue.csv                     Prioritized rows for human inspection
  subset_wheat_artaba_drachma.csv             Wheat/artaba/drachma subset
  trend_wheat_artaba_drachma_all_places.csv   Binned trend statistics
  trend_wheat_artaba_drachma_top_places.csv   Trends by place
  price_index_wheat_artaba_drachma.csv        Price index (baseline = 100)

Usage examples
--------------
Default run:
  python src/3_clean_and_analyze.py

Custom input/output and stricter filtering:
  python src/3_clean_and_analyze.py ^
    --input data/extracted_price_mentions.csv ^
    --outdir outputs/cleaned_data_latest ^
    --min-score 65 --drop-ambiguous

Batch mode (no plots):
  python src/3_clean_and_analyze.py --input data/extracted_price_mentions.csv --no-plots
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pipeline_shared import (
    CUR_TO_DRACHMA,
    UNIT_TO_LITER,
    PLACE_PATTERNS,
    strip_accents,
    norm_greek,
    to_float,
    canon_grain,
    canon_unit,
    canon_currency,
    canon_place,
)


# ============================================================
# CLI
# ============================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 3: clean, canonicalize, and analyze linked price extractions.",
    )
    p.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/extracted_price_mentions.csv"),
        help="Input CSV from Stage 2 (default: data/extracted_price_mentions.csv)",
    )
    p.add_argument(
        "--outdir", "-o",
        type=Path,
        default=Path("outputs/cleaned_data_latest"),
        help="Output directory (default: outputs/cleaned_data_latest)",
    )
    p.add_argument(
        "--min-score",
        type=float,
        default=25,
        help="Minimum score for the main filtered dataset (default: 25)",
    )
    p.add_argument(
        "--drop-ambiguous",
        action="store_true",
        default=False,
        help="Drop ambiguous rows from the main filtered dataset",
    )
    p.add_argument(
        "--year-bin",
        type=int,
        default=25,
        help="Year bin width in years (default: 25)",
    )
    p.add_argument(
        "--review-top-n",
        type=int,
        default=600,
        help="Maximum rows in the manual review queue (default: 600)",
    )
    p.add_argument(
        "--robust-z",
        type=float,
        default=4.0,
        help="MAD-based z-score threshold for group outlier flagging (default: 4.0)",
    )
    p.add_argument(
        "--smooth-window",
        type=int,
        default=3,
        help="Rolling median smoothing window in bins (default: 3)",
    )
    p.add_argument(
        "--index-baseline",
        type=int,
        nargs=2,
        default=[50, 99],
        metavar=("START", "END"),
        help="Price-index baseline period, inclusive (default: 50 99)",
    )
    p.add_argument(
        "--top-places",
        type=int,
        default=6,
        help="Number of top places to plot (default: 6)",
    )
    p.add_argument(
        "--min-n-bin",
        type=int,
        default=5,
        help="Minimum observations per bin for trend plots (default: 5)",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Skip generating matplotlib plots",
    )
    return p.parse_args(argv)



# (Conversion tables CUR_TO_DRACHMA, UNIT_TO_LITER and normalization helpers
#  strip_accents, norm_greek are now imported from pipeline_shared.)


def clean_context(s: str, max_len: int = 500) -> str:
    if s is None:
        return ""
    s = str(s).replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s).strip()
    return s[:max_len]



# (Canonicalization functions canon_grain, canon_unit, canon_currency, canon_place
#  and PLACE_PATTERNS are now imported from pipeline_shared.)


# ============================================================
# DATE PARSING
# ============================================================

def parse_year_from_iso(s: str) -> float:
    if s is None:
        return np.nan
    s = str(s).strip()
    if not s:
        return np.nan
    m = re.match(r"^([+-]?)\s*(\d{1,4})", s)
    if not m:
        return np.nan
    sign = -1 if m.group(1) == "-" else 1
    return float(sign * int(m.group(2)))


def infer_year(row: pd.Series) -> float:
    y = parse_year_from_iso(row.get("Date_When", ""))
    if np.isfinite(y):
        return y
    nb = parse_year_from_iso(row.get("Date_NotBefore", ""))
    na = parse_year_from_iso(row.get("Date_NotAfter", ""))
    if np.isfinite(nb) and np.isfinite(na):
        return (nb + na) / 2.0
    if np.isfinite(nb):
        return nb
    if np.isfinite(na):
        return na
    return np.nan



# (to_float is now imported from pipeline_shared.)


# ============================================================
# PRIMARY SELECTION
# ============================================================

def select_primary_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "Is_Primary" in df.columns:
        prim = df[df["Is_Primary"].astype(str).str.lower().eq("yes")]
        if len(prim) > 0:
            return prim.copy()

    if "Candidate_Rank" in df.columns:
        rank1 = df[df["Candidate_Rank"].astype(str) == "1"]
        if len(rank1) > 0:
            return rank1.copy()

    key_cols = [c for c in ["DDB_ID", "Mention_ID", "Grain_Index"] if c in df.columns]
    if key_cols and "Score" in df.columns:
        tmp = df.copy()
        tmp["ScoreNum"] = pd.to_numeric(tmp["Score"], errors="coerce")
        tmp = tmp.sort_values("ScoreNum", ascending=False)
        return tmp.drop_duplicates(key_cols, keep="first").drop(columns=["ScoreNum"])

    return df.copy()


# ============================================================
# CONVERSIONS
# ============================================================

def convert_currency_to_drachma(amount: float, cur_canon: str) -> float:
    if not np.isfinite(amount):
        return np.nan
    factor = CUR_TO_DRACHMA.get(cur_canon, None)
    if factor is None:
        return np.nan
    return amount * factor


def convert_qty_to_liters(qty: float, unit_canon: str) -> float:
    if not np.isfinite(qty):
        return np.nan
    factor = UNIT_TO_LITER.get(unit_canon, None)
    if factor is None:
        return np.nan
    return qty * factor


def add_standardized_prices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Price_DrachmaEq"] = out.apply(lambda r: convert_currency_to_drachma(r["PriceNum"], r["Cur_Canon"]), axis=1)
    out["Qty_Liters"] = out.apply(lambda r: convert_qty_to_liters(r["QtyNum"], r["Unit_Canon"]), axis=1)

    out["Price_per_Liter_DrachmaEq"] = np.where(
        out["Qty_Liters"] > 0,
        out["Price_DrachmaEq"] / out["Qty_Liters"],
        np.nan,
    )

    artaba_L = UNIT_TO_LITER.get("artaba", None)
    if artaba_L and artaba_L > 0:
        out["Price_per_Artaba_DrachmaEq"] = out["Price_per_Liter_DrachmaEq"] * artaba_L
    else:
        out["Price_per_Artaba_DrachmaEq"] = np.nan

    out["log_Price_per_Liter"] = np.log10(out["Price_per_Liter_DrachmaEq"])
    out.loc[~np.isfinite(out["log_Price_per_Liter"]), "log_Price_per_Liter"] = np.nan

    return out


# ============================================================
# QUALITY TIERS & FILTERS
# ============================================================

def add_quality_tiers(df: pd.DataFrame, min_score: float = 25) -> pd.DataFrame:
    out = df.copy()
    score = pd.to_numeric(out.get("Score", np.nan), errors="coerce")
    ambiguous = out.get("Ambiguous", "").astype(str).str.lower().eq("yes")

    out["Tier"] = np.select(
        [
            (score >= min_score + 5) & (~ambiguous),
            (score >= min_score),
        ],
        ["A", "B"],
        default="C",
    )
    return out


def filter_main_dataset(df: pd.DataFrame, min_score: float = 60,
                        drop_ambiguous: bool = False,
                        require_price_context: bool = True) -> pd.DataFrame:
    out = df.copy()
    score = pd.to_numeric(out.get("Score", np.nan), errors="coerce")
    out = out[score >= min_score]

    if drop_ambiguous and "Ambiguous" in out.columns:
        out = out[out["Ambiguous"].astype(str).str.lower().ne("yes")]

    # Only keep rows classified as PRICE context (v12+)
    if require_price_context and "Context_Type" in out.columns:
        out = out[out["Context_Type"].astype(str).str.upper().eq("PRICE")]

    return out


# ============================================================
# ROBUST OUTLIERS (within groups)
# ============================================================

def robust_zscore(x: pd.Series) -> pd.Series:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or not np.isfinite(mad):
        return pd.Series([np.nan] * len(x), index=x.index)
    return 0.6745 * (x - med) / mad


def add_group_outlier_flags(df: pd.DataFrame, value_col: str, out_col_prefix: str,
                           z_thresh: float = 4.0) -> pd.DataFrame:
    """
    Adds:
      {prefix}_GroupZ
      {prefix}_IsOutlier
    within groups (Grain_Canon, Unit_Canon, Cur_Canon)
    """
    out = df.copy()
    gcols = ["Grain_Canon", "Unit_Canon", "Cur_Canon"]

    zcol = f"{out_col_prefix}_GroupZ"
    fcol = f"{out_col_prefix}_IsOutlier"
    out[zcol] = np.nan

    for _, grp in out.groupby(gcols):
        z = robust_zscore(pd.to_numeric(grp[value_col], errors="coerce"))
        out.loc[grp.index, zcol] = z

    out[fcol] = out[zcol].abs() >= z_thresh
    return out


# ============================================================
# TRENDS + SMOOTHING + INDEX
# ============================================================

def rolling_median(s: pd.Series, window: int = 3) -> pd.Series:
    return s.rolling(window=window, center=True, min_periods=1).median()


def make_trend_table(df: pd.DataFrame, value_col: str, group_cols: list[str],
                     smooth_window: int = 3) -> pd.DataFrame:
    """
    group_cols should include:
      - YearBin
      - optional Place_Canon, Grain_Canon, Unit_Canon, Cur_Canon
    """
    sub = df.dropna(subset=["YearBin", value_col]).copy()
    if len(sub) == 0:
        return pd.DataFrame()

    trend = sub.groupby(group_cols).agg(
        n=(value_col, "size"),
        median=(value_col, "median"),
        q25=(value_col, lambda x: np.nanpercentile(x, 25)),
        q75=(value_col, lambda x: np.nanpercentile(x, 75)),
    ).reset_index()

    trend = trend.sort_values(group_cols)
    # smoothed median only if YearBin is in group_cols
    if "YearBin" in group_cols:
        # apply smoothing per other groups
        other = [c for c in group_cols if c != "YearBin"]
        if other:
            trend["median_smooth"] = trend.groupby(other)["median"].transform(lambda s: rolling_median(s, smooth_window))
        else:
            trend["median_smooth"] = rolling_median(trend["median"], smooth_window)

    return trend


def compute_price_index(trend: pd.DataFrame, baseline_start: int = 50,
                        baseline_end: int = 99) -> pd.DataFrame:
    """
    Compute index=100 for baseline period median level.
    Requires columns: YearBin, median (or median_smooth)
    """
    if len(trend) == 0:
        return trend

    out = trend.copy()
    y = out["YearBin"].astype(float)

    base = out[(y >= baseline_start) & (y <= baseline_end)]
    if len(base) == 0:
        out["price_index"] = np.nan
        return out

    base_level = np.nanmedian(base["median"])
    if not np.isfinite(base_level) or base_level == 0:
        out["price_index"] = np.nan
        return out

    out["price_index"] = 100.0 * out["median"] / base_level
    if "median_smooth" in out.columns and out["median_smooth"].notna().any():
        out["price_index_smooth"] = 100.0 * out["median_smooth"] / base_level
    else:
        out["price_index_smooth"] = np.nan

    return out


def plot_trend_lines(trend: pd.DataFrame, title: str, label_col: str | None = None,
                     ycol: str = "median_smooth", min_n: int = 5,
                     year_bin: int = 25, top_places: int = 6):
    """
    Plot smoothed median trend lines.
    If label_col is provided, plot one line per label value (top labels by total n).
    """
    if len(trend) == 0:
        print("No trend data:", title)
        return

    # filter bins by n
    tr = trend[trend["n"] >= min_n].copy()
    if len(tr) == 0:
        print(f"No bins with n >= {min_n}:", title)
        return

    plt.figure()

    if label_col is None:
        x = tr["YearBin"].astype(float)
        plt.plot(x, tr[ycol].astype(float), linewidth=2)
    else:
        # choose top places/labels by total n
        label_totals = tr.groupby(label_col)["n"].sum().sort_values(ascending=False)
        top_labels = list(label_totals.head(top_places).index)

        for lab in top_labels:
            t2 = tr[tr[label_col] == lab].sort_values("YearBin")
            x = t2["YearBin"].astype(float)
            plt.plot(x, t2[ycol].astype(float), linewidth=2, label=str(lab))

        plt.legend()

    plt.xlabel(f"Year bin ({year_bin}y)")
    plt.ylabel(ycol)
    plt.title(title)
    plt.show()


# ============================================================
# MANUAL REVIEW QUEUE
# ============================================================

def build_review_queue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prioritize rows for manual inspection.

    Rules (reasons):
      - ambiguous extraction
      - low score
      - missing qty (can't compute unit price)
      - very large span or distance
      - group outlier in UnitPrice_raw
      - group outlier in Price_per_Liter_DrachmaEq
      - extreme unit price (top 1% within subset)
    """
    out = df.copy()

    score = pd.to_numeric(out.get("Score", np.nan), errors="coerce")
    ambiguous = out.get("Ambiguous", "").astype(str).str.lower().eq("yes")

    out["ReviewReason"] = ""

    def add_reason(mask, reason):
        out.loc[mask, "ReviewReason"] = out.loc[mask, "ReviewReason"].apply(
            lambda s: (s + "; " + reason).strip("; ").strip()
        )

    add_reason(ambiguous, "ambiguous")
    add_reason(score < 25, "low_score(<25)")

    add_reason(out["QtyNum"].isna() | (out["QtyNum"] <= 0), "missing_or_bad_qty")
    add_reason(out.get("Span_Toks", 0).astype(float) >= 60, "wide_span(>=60)")
    add_reason(out.get("Dist_GP", 0).astype(float) >= 35, "far_grain_price(>=35)")
    add_reason(out.get("Priceword_Near", "").astype(str).str.lower().eq("no"), "no_priceword_near")

    # v12 signal-based review reasons
    if "Signal_Strength" in out.columns:
        sig = pd.to_numeric(out["Signal_Strength"], errors="coerce")
        add_reason(sig < 0.5, "weak_signal(<0.5)")
    if "Neg_Signals" in out.columns:
        has_neg = out["Neg_Signals"].astype(str).str.len() > 0
        add_reason(has_neg, "has_negative_signals")

    # outlier flags if present
    if "RAW_Group_IsOutlier" in out.columns:
        add_reason(out["RAW_Group_IsOutlier"].fillna(False), "group_outlier_raw_unitprice")
    if "STD_Group_IsOutlier" in out.columns:
        add_reason(out["STD_Group_IsOutlier"].fillna(False), "group_outlier_std_price_per_liter")

    # extreme unit price heuristic (global top 1%)
    if "UnitPrice_raw" in out.columns:
        up = pd.to_numeric(out["UnitPrice_raw"], errors="coerce")
        cutoff = up.quantile(0.99)
        add_reason(up >= cutoff, "unitprice_top1pct")

    # final priority score (bigger = review earlier)
    priority = np.zeros(len(out), dtype=float)
    priority += ambiguous.astype(float) * 3.0
    priority += (score < 25).astype(float) * 2.5
    priority += (out["QtyNum"].isna() | (out["QtyNum"] <= 0)).astype(float) * 2.0
    priority += (out.get("Span_Toks", 0).astype(float) >= 60).astype(float) * 1.5
    priority += (out.get("Dist_GP", 0).astype(float) >= 35).astype(float) * 1.2
    priority += (out.get("Priceword_Near", "").astype(str).str.lower().eq("no")).astype(float) * 1.0

    if "RAW_Group_IsOutlier" in out.columns:
        priority += out["RAW_Group_IsOutlier"].fillna(False).astype(float) * 2.0
    if "STD_Group_IsOutlier" in out.columns:
        priority += out["STD_Group_IsOutlier"].fillna(False).astype(float) * 2.0

    # v12 signal-based priority factors
    if "Signal_Strength" in out.columns:
        sig = pd.to_numeric(out["Signal_Strength"], errors="coerce").fillna(0)
        priority += (sig < 0.5).astype(float) * 1.5
    if "Neg_Signals" in out.columns:
        has_neg = out["Neg_Signals"].astype(str).str.len() > 0
        priority += has_neg.astype(float) * 1.5

    out["ReviewPriority"] = priority

    # keep only rows that have at least one reason
    out = out[out["ReviewReason"].str.len() > 0].copy()
    out = out.sort_values(["ReviewPriority", "Score"], ascending=[False, True])

    return out


# ============================================================
# MAIN
# ============================================================

def main(argv: list[str] | None = None):
    args = parse_args(argv)

    infile = args.input
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    min_score_main = args.min_score
    drop_ambiguous_main = args.drop_ambiguous
    year_bin = args.year_bin
    review_top_n = args.review_top_n
    robust_z_thresh = args.robust_z
    smooth_window = args.smooth_window
    index_baseline_start, index_baseline_end = args.index_baseline
    top_places_to_plot = args.top_places
    min_n_per_bin = args.min_n_bin

    if not infile.exists():
        raise FileNotFoundError(f"Input CSV not found: {infile.resolve()}")

    df_raw = pd.read_csv(infile).fillna("")
    print("Loaded:", infile, "rows =", len(df_raw))

    # Primary selection
    df = select_primary_rows(df_raw)
    print("Primary rows:", len(df))

    # Canonicalize + numeric
    df["Grain_Canon"] = df["Grain_Form"].apply(canon_grain)
    df["Unit_Canon"] = df["Qty_Unit"].apply(canon_unit)
    df["Cur_Canon"] = df["Price_Cur"].apply(canon_currency)
    df["Place_Canon"] = df.get("Place", "").apply(canon_place)

    df["QtyNum"] = df["Qty_Value"].apply(to_float)
    df["PriceNum"] = df["Price_Value"].apply(to_float)

    df["UnitPrice_raw"] = np.where(df["QtyNum"] > 0, df["PriceNum"] / df["QtyNum"], np.nan)

    # Date fields
    df["Year"] = df.apply(infer_year, axis=1)
    df["YearBin"] = np.where(
        np.isfinite(df["Year"]),
        (np.floor(df["Year"] / year_bin) * year_bin).astype("Int64"),
        pd.NA,
    )

    # Context cleaning
    if "Context_Window" in df.columns:
        df["Context_Clean"] = df["Context_Window"].apply(clean_context)
        df["Context_OneLine"] = df["Context_Clean"].str.replace("\n", " ", regex=False)

    # Quality tiers + standardized prices
    df = add_quality_tiers(df, min_score_main)
    df = add_standardized_prices(df)

    # Export primary cleaned
    out_primary = outdir / "PRIMARY_cleaned.csv"
    df.to_csv(out_primary, index=False, encoding="utf-8-sig")
    print("Saved:", out_primary)

    # Main filtered dataset
    df_main = filter_main_dataset(df, min_score_main, drop_ambiguous_main,
                                   require_price_context=True)
    print("Main filtered rows:", len(df_main))

    # Outlier flags
    df_main = add_group_outlier_flags(df_main, value_col="UnitPrice_raw",
                                      out_col_prefix="RAW_Group", z_thresh=robust_z_thresh)
    df_main = add_group_outlier_flags(df_main, value_col="Price_per_Liter_DrachmaEq",
                                      out_col_prefix="STD_Group", z_thresh=robust_z_thresh)

    out_main = outdir / "MAIN_filtered.csv"
    df_main.to_csv(out_main, index=False, encoding="utf-8-sig")
    print("Saved:", out_main)

    # Subset: wheat + artaba + drachma
    subset = df_main[
        (df_main["Grain_Canon"] == "wheat (pyros)") &
        (df_main["Unit_Canon"] == "artaba") &
        (df_main["Cur_Canon"] == "drachma")
    ].copy()

    out_subset = outdir / "subset_wheat_artaba_drachma.csv"
    subset.to_csv(out_subset, index=False, encoding="utf-8-sig")
    print("Saved subset:", out_subset, "rows =", len(subset))

    # Trend: all places combined
    trend_all = make_trend_table(
        subset,
        value_col="UnitPrice_raw",
        group_cols=["YearBin"],
        smooth_window=smooth_window,
    )
    out_trend_all = outdir / "trend_wheat_artaba_drachma_all_places.csv"
    trend_all.to_csv(out_trend_all, index=False, encoding="utf-8-sig")
    print("Saved:", out_trend_all)

    # Trend: by place
    trend_place = make_trend_table(
        subset,
        value_col="UnitPrice_raw",
        group_cols=["Place_Canon", "YearBin"],
        smooth_window=smooth_window,
    )
    out_trend_place = outdir / "trend_wheat_artaba_drachma_top_places.csv"
    trend_place.to_csv(out_trend_place, index=False, encoding="utf-8-sig")
    print("Saved:", out_trend_place)

    # Price index
    price_index = compute_price_index(trend_all, index_baseline_start, index_baseline_end)
    out_index = outdir / "price_index_wheat_artaba_drachma.csv"
    price_index.to_csv(out_index, index=False, encoding="utf-8-sig")
    print("Saved:", out_index)

    # Manual review queue
    review = build_review_queue(df_main)
    review = review.head(review_top_n)
    out_review = outdir / "manual_review_queue.csv"
    review.to_csv(out_review, index=False, encoding="utf-8-sig")
    print("Saved review queue:", out_review, "rows =", len(review))

    # ============================================================
    # PLOTS
    # ============================================================

    if not args.no_plots:
        # 1) score histogram
        if "Score" in df.columns:
            s = pd.to_numeric(df["Score"], errors="coerce").dropna()
            plt.figure()
            plt.hist(s, bins=30)
            plt.xlabel("Score")
            plt.ylabel("Count")
            plt.title("Extraction confidence score distribution (primary rows)")
            plt.show()

        # 2) trend lines (all places)
        if len(trend_all) > 0:
            plot_trend_lines(
                trend_all,
                title="Wheat: drachma/artaba (median_smooth) — all places",
                label_col=None,
                ycol="median_smooth",
                min_n=min_n_per_bin,
                year_bin=year_bin,
                top_places=top_places_to_plot,
            )

        # 3) trend lines (top places)
        if len(trend_place) > 0:
            plot_trend_lines(
                trend_place,
                title="Wheat: drachma/artaba (median_smooth) — top places",
                label_col="Place_Canon",
                ycol="median_smooth",
                min_n=min_n_per_bin,
                year_bin=year_bin,
                top_places=top_places_to_plot,
            )

        # 4) price index plot
        if len(price_index) > 0 and "price_index_smooth" in price_index.columns:
            px = price_index.dropna(subset=["price_index_smooth", "YearBin"]).copy()
            if len(px) > 0:
                plt.figure()
                plt.plot(px["YearBin"].astype(float), px["price_index_smooth"].astype(float), linewidth=2)
                plt.xlabel(f"Year bin ({year_bin}y)")
                plt.ylabel("Price index (baseline=100)")
                plt.title(f"Wheat price index (drachma/artaba), baseline {index_baseline_start}-{index_baseline_end} CE")
                plt.show()

        # 5) scatter of subset (year vs unit price)
        scat = subset.dropna(subset=["Year", "UnitPrice_raw"])
        if len(scat) > 0:
            plt.figure()
            plt.scatter(scat["Year"], scat["UnitPrice_raw"], alpha=0.4)
            plt.xlabel("Year (approx)")
            plt.ylabel("Unit price (drachma/artaba)")
            plt.title("Wheat unit price scatter (main filtered)")
            plt.show()

    print("\nDONE. All outputs are in:", outdir.resolve())


if __name__ == "__main__":
    main()
