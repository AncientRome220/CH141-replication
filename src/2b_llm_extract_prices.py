#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Stage 2B: LLM-based grain price extractor (Gemini Flash)

Parallel LLM extraction pipeline for the CH141 comparison study.
For each candidate document from Stage 1, parses the EpiDoc XML, finds grain
mentions, creates narrow text windows (~200 chars), and sends each window to
Gemini Flash for structured extraction.

The output schema mirrors Stage 2 (rule-based) so the two methods can be
compared directly against a gold-standard annotation set.

CLI arguments
-------------
Required:
  --candidates FILE      Stage 1 candidates CSV
  --ddb-dir DIR          Path to DDB EpiDoc XML directory

Output:
  --out FILE             LLM extraction CSV (default: data/extracted_price_mentions_llm.csv)

LLM settings:
  --model NAME           Gemini model name (default: gemini-2.5-flash)
  --max-workers N        Concurrent API threads (default: 4)
  --max-retries N        Retries per API call (default: 5)
  --dry-run              Extract windows but skip API calls; write prompt to stdout
  --resume               Skip DDB_IDs already in --out file; append new results

Processing:
  --max-docs N           Process only first N documents; 0 = all (default: 0)
  --encoding ENC         CSV encoding (default: utf-8-sig)
  --debug                Enable verbose logging

Usage examples
--------------
Dry run (no API calls):
  python src/2b_llm_extract_prices.py ^
    --candidates data/candidate_documents.csv ^
    --ddb-dir "C:\research\DDB_EpiDoc_XML" ^
    --dry-run --max-docs 5

Full extraction:
  python src/2b_llm_extract_prices.py ^
    --candidates data/candidate_documents.csv ^
    --ddb-dir "C:\research\DDB_EpiDoc_XML" ^
    --out data/extracted_price_mentions_llm.csv ^
    --max-workers 4
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from lxml import etree

from pipeline_shared import (
    NS,
    GRAIN_RE,
    UNIT_RE,
    MONEY_RE,
    PRICEWORD_RE,
    normalize_for_search,
)

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# XML text extraction (shared logic with Stage 2)
# ──────────────────────────────────────────────

def safe_string(node) -> str:
    if node is None:
        return ""
    try:
        return node.xpath("string()").strip()
    except Exception:
        return ""


def extract_text_from_block(elem: etree._Element) -> str:
    """
    Extract readable text from an EpiDoc XML block.
    <num value="25">κε</num> -> "25" (use numeric value).
    Editorial markup (<supplied>, <unclear>, etc.) is flattened to text.
    <lb/> -> " | " (line break marker for human readability).
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
                pieces.append(" | ")
            elif ln in ("cb", "pb"):
                pieces.append(" || ")
            else:
                walk(ch)
            if ch.tail:
                pieces.append(ch.tail)

    walk(elem)
    return " ".join(" ".join(pieces).split())


def iter_text_blocks(tree: etree._ElementTree) -> List[Tuple[str, str]]:
    """
    Yield (block_tag, text) for every text-bearing block in the document body.
    """
    results = []
    for div in tree.xpath("//tei:div[@type='edition']", namespaces=NS):
        for b in div.xpath(
            ".//tei:ab | .//tei:p | .//tei:seg | .//tei:l", namespaces=NS
        ):
            tag = etree.QName(b).localname
            text = extract_text_from_block(b)
            if text.strip():
                results.append((tag, text))
    return results


# ──────────────────────────────────────────────
# Window extraction
# ──────────────────────────────────────────────

def find_grain_windows(
    blocks: List[Tuple[str, str]],
    window_chars: int = 200,
    require_number: bool = True,
) -> List[Dict[str, Any]]:
    """
    Find grain mentions across all blocks and return narrow text windows.
    Each window is centered on a grain mention with `window_chars` total context.
    """
    # Concatenate all blocks into a single text with block separators
    full_text = " ||| ".join(text for _, text in blocks)
    norm_text = normalize_for_search(full_text)

    windows = []
    seen_positions = set()

    for m in GRAIN_RE.finditer(norm_text):
        grain_start = m.start()
        grain_end = m.end()
        grain_form = m.group(0)

        # Deduplicate overlapping windows (within 50 chars)
        bucket = grain_start // 50
        if bucket in seen_positions:
            continue
        seen_positions.add(bucket)

        # Extract window from the ORIGINAL (non-normalized) text
        half = window_chars // 2
        win_start = max(0, grain_start - half)
        win_end = min(len(full_text), grain_end + half)
        window_text = full_text[win_start:win_end].strip()

        # Also get normalized window for pattern checks
        norm_window = norm_text[win_start:win_end].strip()

        # Check if window has money/unit cues (lightweight pre-filter)
        has_money = bool(MONEY_RE.search(norm_window))
        has_unit = bool(UNIT_RE.search(norm_window))
        has_number = bool(re.search(r"\d+", window_text))
        has_priceword = bool(PRICEWORD_RE.search(norm_window))

        # Pre-filter: skip windows with no number (cannot be a price)
        if require_number and not has_number:
            continue

        windows.append({
            "grain_form": grain_form,
            "window_text": window_text,
            "norm_window": norm_window,
            "char_offset": grain_start,
            "has_money": has_money,
            "has_unit": has_unit,
            "has_number": has_number,
            "has_priceword": has_priceword,
        })

    return windows


# ──────────────────────────────────────────────
# LLM prompt
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert papyrologist specializing in the economic history of Roman Egypt.
You analyze short text excerpts from Greek papyri to identify grain price statements.

A "grain price" is an exchange rate between a grain type (wheat/πυρός, barley/κριθή, etc.)
and a monetary amount (drachmas/δραχμαί, obols/ὀβολοί, etc.), typically with a quantity
measured in artabas (ἀρτάβαι) or similar units.

Important distinctions:
- INCLUDE: sale prices, purchase prices, valuations (τιμή), rate expressions (ὡς τῆς ἀρτάβης)
- EXCLUDE: tax-in-kind, freight fees, milling fees, rations, loans of grain (unless interest is priced)
- A passage mentioning grain AND money does NOT automatically contain a price — the two must be linked
  in a transactional or rate expression.
"""

USER_PROMPT_TEMPLATE = """\
Analyze the following excerpt from a Greek papyrus (document: {doc_id}).
The text contains a mention of a grain type ({grain_form}).

Determine whether this passage contains an explicit grain price observation.
If it does, extract the structured fields below.

Here are two examples to guide your judgment:

Example 1 (IS a grain price):
Text: "Ὀρσενούφ ι Φομβῶ τος τιμῆ ς πυροῦ ἀρταβῶν 3 1/2 . δραχμαὶ 20"
Answer:
{{
  "is_price": true,
  "confidence": "high",
  "commodity": "wheat",
  "commodity_greek": "πυροῦ",
  "quantity_value": 3.5,
  "quantity_unit": "artaba",
  "price_value": 20,
  "price_currency": "drachma",
  "transaction_type": "sale",
  "has_damage": false,
  "reasoning": "Explicit τιμῆς (price-of) construction links 3.5 artabas of wheat to 20 drachmas."
}}

Example 2 (NOT a grain price):
Text: "38 ἔτους χωματι κὸν δραχμὰς 16 | ἁλικῆς δραχμὰς 11 | φυλακ ιτικὸν πυροῦ 3 | γίνονται δραχμαὶ 177 5 πυροῦ 9"
Answer:
{{
  "is_price": false,
  "confidence": "high",
  "commodity": null,
  "commodity_greek": null,
  "quantity_value": null,
  "quantity_unit": null,
  "price_value": null,
  "price_currency": null,
  "transaction_type": null,
  "has_damage": null,
  "reasoning": "Tax context: χωματικόν (dyke tax), ἁλική (salt tax), φυλακιτικόν (guard tax). Wheat (πυροῦ 3) is paid as tax-in-kind; the drachma amounts are separate tax payments, not grain prices."
}}

Now analyze the following excerpt:

Text excerpt:
---
{window_text}
---

Respond with a JSON object. If the passage does NOT contain a grain price, set
"is_price" to false and leave other fields null. If it does, fill in as many
fields as you can identify from the text.

{{
  "is_price": true or false,
  "confidence": "high" or "medium" or "low",
  "commodity": "wheat" or "barley" or "other grain type" or null,
  "commodity_greek": "the Greek word as it appears" or null,
  "quantity_value": number or null,
  "quantity_unit": "artaba" or "choenix" or "medimnos" or other unit or null,
  "price_value": number or null,
  "price_currency": "drachma" or "obol" or "denarius" or "talent" or other or null,
  "transaction_type": "sale" or "purchase" or "valuation" or "rate" or "other" or null,
  "has_damage": true or false,
  "reasoning": "brief explanation of your judgment (1-2 sentences)"
}}

Output ONLY the JSON object, no other text.
"""


def build_prompt(doc_id: str, grain_form: str, window_text: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        doc_id=doc_id,
        grain_form=grain_form,
        window_text=window_text,
    )


# ──────────────────────────────────────────────
# Rate-limit exception
# ──────────────────────────────────────────────

class RateLimitError(Exception):
    """Raised when API returns 429 / Resource Exhausted."""
    pass


# ──────────────────────────────────────────────
# Gemini API call with retry
# ──────────────────────────────────────────────

def call_gemini(
    client,
    prompt: str,
    system_prompt: str,
    max_retries: int = 5,
    model_name: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    """
    Call Gemini API with exponential backoff retry.
    Returns parsed JSON dict or error dict.
    """
    from google.genai import types as genai_types

    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )

            # Extract text from response
            if hasattr(resp, "text"):
                text = resp.text.strip()
            elif hasattr(resp, "candidates") and resp.candidates:
                text = resp.candidates[0].content.parts[0].text.strip()
            else:
                raise ValueError(f"Unexpected response format: {type(resp)}")

            if not text:
                raise ValueError("Empty response")

            # Strip markdown fences
            if "```" in text:
                text = re.sub(
                    r"```(?:json)?\s*|\s*```", "", text, flags=re.IGNORECASE
                ).strip()

            # Extract JSON object
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                text = text[json_start:json_end]

            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError(f"Response is not a JSON object: {type(data)}")

            return data

        except Exception as e:
            err_str = str(e).lower()
            if ("429" in err_str
                    or ("resource" in err_str and "exhaust" in err_str)
                    or "too many" in err_str):
                raise RateLimitError(f"Rate limit hit: {e}") from e
            last_error = e
            if attempt < max_retries - 1:
                sleep_s = (2 ** attempt) + random.random()
                log.debug(
                    "API retry %d/%d: %s (sleeping %.1fs)",
                    attempt + 1, max_retries, e, sleep_s,
                )
                time.sleep(sleep_s)
            else:
                log.warning(
                    "API failed after %d attempts: %s: %s",
                    max_retries, type(e).__name__, e,
                )

    return {"_error": str(last_error), "is_price": False}


# ──────────────────────────────────────────────
# Document processing
# ──────────────────────────────────────────────

# Output columns (mirrors Stage 2 where possible)
OUT_FIELDS = [
    "DDB_ID",
    "Mention_ID",
    "Grain_Index",
    "Title",
    "Place",
    "Date_Text",
    "Date_When",
    "Date_NotBefore",
    "Date_NotAfter",
    "Grain_Form",
    "Qty_Value",
    "Qty_Unit",
    "Price_Value",
    "Price_Cur",
    "Is_Price",
    "Confidence",
    "Transaction_Type",
    "Has_Damage",
    "Reasoning",
    "Has_Money_Cue",
    "Has_Unit_Cue",
    "Has_Number",
    "Has_Priceword",
    "Context_Window",
    "LLM_Error",
]


def process_one_document(
    row: Dict[str, Any],
    ddb_dir: Path,
    model,
    max_retries: int,
    dry_run: bool = False,
    require_number: bool = True,
    max_windows: int = 20,
    model_name: str = "gemini-2.5-flash",
) -> List[Dict[str, Any]]:
    """
    Process a single candidate document: parse XML, find grain windows,
    call LLM for each window, return list of result dicts.
    """
    ddb_id = row.get("DDB_ID", "")
    xml_rel = row.get("XML_RelPath", "")
    title = row.get("Title", "")
    place = row.get("Place", "")
    date_text = row.get("Date_Text", "")
    date_when = row.get("Date_When", "")
    date_nb = row.get("Date_NotBefore", "")
    date_na = row.get("Date_NotAfter", "")

    results = []

    # Parse XML
    xml_path = ddb_dir / xml_rel
    if not xml_path.exists():
        log.debug("XML not found: %s", xml_path)
        return results

    try:
        tree = etree.parse(str(xml_path))
    except Exception as e:
        log.warning("XML parse error for %s: %s", ddb_id, e)
        return results

    # Extract text blocks
    blocks = iter_text_blocks(tree)
    if not blocks:
        return results

    # Find grain windows
    windows = find_grain_windows(blocks, require_number=require_number)
    if not windows:
        return results

    # Cap windows per document to avoid huge accounts
    if max_windows > 0 and len(windows) > max_windows:
        # Prioritize windows with money cues
        windows.sort(key=lambda w: (w["has_money"], w["has_priceword"], w["has_unit"]), reverse=True)
        windows = windows[:max_windows]

    for gi, win in enumerate(windows):
        mention_id = f"{ddb_id}__llm_g{gi}"
        prompt = build_prompt(ddb_id, win["grain_form"], win["window_text"])

        if dry_run:
            # In dry-run mode, print the prompt and create a placeholder result
            try:
                out = sys.stdout
                if hasattr(out, "buffer"):
                    import io
                    out = io.TextIOWrapper(out.buffer, encoding="utf-8", errors="replace")
                out.write(f"\n{'='*60}\n")
                out.write(f"Document: {ddb_id} | Window {gi}\n")
                out.write(f"{'='*60}\n")
                out.write(prompt + "\n")
                out.flush()
            except Exception:
                log.debug("Could not print prompt for %s window %d (encoding)", ddb_id, gi)
            llm_data = {"is_price": None, "_dry_run": True}
        else:
            llm_data = call_gemini(model, prompt, SYSTEM_PROMPT, max_retries, model_name)

        result = {
            "DDB_ID": ddb_id,
            "Mention_ID": mention_id,
            "Grain_Index": gi,
            "Title": title,
            "Place": place,
            "Date_Text": date_text,
            "Date_When": date_when,
            "Date_NotBefore": date_nb,
            "Date_NotAfter": date_na,
            "Grain_Form": win["grain_form"],
            "Qty_Value": llm_data.get("quantity_value"),
            "Qty_Unit": llm_data.get("quantity_unit"),
            "Price_Value": llm_data.get("price_value"),
            "Price_Cur": llm_data.get("price_currency"),
            "Is_Price": llm_data.get("is_price"),
            "Confidence": llm_data.get("confidence"),
            "Transaction_Type": llm_data.get("transaction_type"),
            "Has_Damage": llm_data.get("has_damage"),
            "Reasoning": llm_data.get("reasoning"),
            "Has_Money_Cue": win["has_money"],
            "Has_Unit_Cue": win["has_unit"],
            "Has_Number": win["has_number"],
            "Has_Priceword": win["has_priceword"],
            "Context_Window": win["window_text"],
            "LLM_Error": llm_data.get("_error"),
        }
        results.append(result)

    return results


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Stage 2B: LLM-based grain price extraction (Gemini Flash)",
    )
    p.add_argument("--candidates", required=True, help="Stage 1 candidates CSV")
    p.add_argument("--ddb-dir", required=True, help="DDB EpiDoc XML directory")
    p.add_argument(
        "--out",
        default="data/extracted_price_mentions_llm.csv",
        help="Output CSV (default: %(default)s)",
    )
    p.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model name (default: %(default)s)",
    )
    p.add_argument(
        "--max-workers", type=int, default=4,
        help="Concurrent API threads (default: %(default)s)",
    )
    p.add_argument(
        "--max-retries", type=int, default=5,
        help="API call retries (default: %(default)s)",
    )
    p.add_argument(
        "--max-docs", type=int, default=0,
        help="Process only first N docs; 0 = all (default: %(default)s)",
    )
    p.add_argument(
        "--no-filter", action="store_true",
        help="Send ALL grain windows to LLM, even those without numbers",
    )
    p.add_argument(
        "--max-windows-per-doc", type=int, default=20,
        help="Max windows to extract per document; 0 = unlimited (default: %(default)s)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Extract windows and print prompts; no API calls",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output: skip DDB_IDs already in --out file",
    )
    p.add_argument(
        "--encoding", default="utf-8-sig",
        help="CSV encoding (default: %(default)s)",
    )
    p.add_argument("--debug", action="store_true", help="Verbose logging")
    return p.parse_args(argv)


def _load_dotenv(path: str = ".env") -> None:
    """Load .env file into os.environ (overrides existing values)."""
    p = Path(path)
    if not p.exists():
        # Try project root (parent of src/)
        p = Path(__file__).resolve().parent.parent / path
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip("\"'")


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    # Load candidates
    df = pd.read_csv(args.candidates, encoding=args.encoding, dtype=str)
    log.info("Loaded %d candidate documents", len(df))

    if args.max_docs > 0:
        df = df.head(args.max_docs)
        log.info("Limited to first %d documents", len(df))

    ddb_dir = Path(args.ddb_dir)
    if not ddb_dir.is_dir():
        log.error("DDB directory not found: %s", ddb_dir)
        sys.exit(1)

    # Load .env file (overrides stale system env vars)
    _load_dotenv()

    # Initialize Gemini client (skip if dry-run)
    model = None
    if not args.dry_run:
        try:
            from google import genai

            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                log.error(
                    "GEMINI_API_KEY environment variable not set. "
                    "Set it before running: set GEMINI_API_KEY=your_key"
                )
                sys.exit(1)
            model = genai.Client(api_key=api_key)
            log.info("Initialized Gemini client, model: %s", args.model)
        except ImportError:
            log.error(
                "google-genai package not installed. "
                "Install it: pip install google-genai"
            )
            sys.exit(1)

    # Prepare output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build task list
    tasks = []
    for _, r in df.iterrows():
        tasks.append(dict(r))

    # Resume: skip already-processed documents
    if args.resume and out_path.exists():
        existing_df = pd.read_csv(out_path, encoding=args.encoding, dtype=str, usecols=["DDB_ID"])
        done_ids = set(existing_df["DDB_ID"].unique())
        before = len(tasks)
        tasks = [t for t in tasks if t["DDB_ID"] not in done_ids]
        log.info("Resume: skipping %d already-processed docs, %d remaining", before - len(tasks), len(tasks))

    require_number = not args.no_filter
    max_win = args.max_windows_per_doc
    log.info(
        "Processing %d documents with %d workers (filter=%s, max_win=%s)...",
        len(tasks), args.max_workers,
        "number required" if require_number else "off",
        max_win if max_win > 0 else "unlimited",
    )

    # Process documents
    all_results: List[Dict[str, Any]] = []
    completed = 0
    errors = 0
    rate_limited = False

    if args.dry_run or args.max_workers <= 1:
        # Sequential processing (for dry-run or debugging)
        for task in tasks:
            try:
                results = process_one_document(
                    task, ddb_dir, model, args.max_retries, args.dry_run,
                    require_number, max_win, args.model,
                )
                all_results.extend(results)
            except RateLimitError as e:
                log.error("Rate limit error: %s", e)
                rate_limited = True
                break
            except Exception as e:
                log.warning("Error processing %s: %s", task.get("DDB_ID"), e)
                errors += 1
            completed += 1
            if completed % 100 == 0:
                log.info(
                    "Progress: %d/%d docs, %d windows extracted",
                    completed, len(tasks), len(all_results),
                )
    else:
        # Parallel processing with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    process_one_document,
                    task, ddb_dir, model, args.max_retries, args.dry_run,
                    require_number, max_win, args.model,
                ): task
                for task in tasks
            }

            for future in as_completed(futures):
                task = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                except RateLimitError as e:
                    log.error("Rate limit error: %s", e)
                    rate_limited = True
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                except Exception as e:
                    log.warning(
                        "Error processing %s: %s",
                        task.get("DDB_ID"), e,
                    )
                    errors += 1
                completed += 1
                if completed % 100 == 0:
                    log.info(
                        "Progress: %d/%d docs, %d windows, %d errors",
                        completed, len(tasks), len(all_results), errors,
                    )

    if rate_limited:
        log.warning(
            "Rate limit reached after %d/%d docs. "
            "Saving %d partial results to %s",
            completed, len(tasks), len(all_results), out_path,
        )
    else:
        log.info(
            "Done. %d documents processed, %d windows extracted, %d errors.",
            completed, len(all_results), errors,
        )

    # Write output
    if all_results:
        df_out = pd.DataFrame(all_results, columns=OUT_FIELDS)
        if args.resume and out_path.exists():
            df_out.to_csv(out_path, mode="a", header=False, index=False, encoding=args.encoding)
        else:
            df_out.to_csv(out_path, index=False, encoding=args.encoding)
        log.info("Wrote %d rows to %s", len(df_out), out_path)
    else:
        log.warning("No results extracted.")

    # Summary statistics
    if all_results and not args.dry_run:
        if args.resume and out_path.exists():
            df_out = pd.read_csv(out_path, encoding=args.encoding, dtype=str)
        else:
            df_out = pd.DataFrame(all_results)
        n_price = df_out["Is_Price"].apply(
            lambda x: str(x).lower() in ("true", "1", "yes")
        ).sum()
        log.info(
            "Summary: %d windows total, %d classified as price (%.1f%%)",
            len(df_out), n_price, 100 * n_price / max(len(df_out), 1),
        )
        if "Confidence" in df_out.columns:
            conf_counts = df_out.loc[
                df_out["Is_Price"].apply(
                    lambda x: str(x).lower() in ("true", "1", "yes")
                ),
                "Confidence",
            ].value_counts()
            for level, count in conf_counts.items():
                log.info("  Confidence %s: %d", level, count)


if __name__ == "__main__":
    main()
