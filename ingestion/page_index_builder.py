"""
ingestion/page_index_builder.py
──────────────────────────────────────────────
WHAT THIS FILE DOES:
  Reads the 4 legal PDFs and builds a structured
  tree index (like a Table of Contents) using GPT-4o.

WHY WE NEED IT:
  Without this, the system does not know the structure
  of the law. It would be like having books with no
  table of contents.

WHAT IT PRODUCES:
  data/index/BNS_index.json   ← the tree structure
  data/index/BNS_pages.json   ← all page texts
  (same for BNSS, BSA, DPDP)

HOW TO RUN:
  python -m ingestion.page_index_builder --act ALL
  python -m ingestion.page_index_builder --act BNS --file data/raw/bns_2023.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import pdfplumber
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.core.config import get_settings
from app.core.logger import get_logger, setup_logging

log = get_logger(__name__)

# ── Act metadata ──────────────────────────────────────────────────────
# Information about each Act we support
ACT_META = {
    "BNS": {
        "full_name": "Bharatiya Nyaya Sanhita 2023",
        "replaces":  "Indian Penal Code 1860",
        "effective": "1 July 2024",
        "file_stems": ["bns_2023", "bns"],
    },
    "BNSS": {
        "full_name": "Bharatiya Nagarik Suraksha Sanhita 2023",
        "replaces":  "Code of Criminal Procedure 1973",
        "effective": "1 July 2024",
        "file_stems": ["bnss_2023", "bnss"],
    },
    "BSA": {
        "full_name": "Bharatiya Sakshya Adhiniyam 2023",
        "replaces":  "Indian Evidence Act 1872",
        "effective": "1 July 2024",
        "file_stems": ["bsa_2023", "bsa"],
    },
    "DPDP": {
        "full_name": "Digital Personal Data Protection Act 2023",
        "replaces":  None,
        "effective": "Pending full notification",
        "file_stems": ["dpdp_2023", "dpdp"],
    },
}

# How many pages to send to GPT-4o per batch
PAGES_PER_BATCH = 15


# ── STEP 1: Read every page from the PDF ─────────────────────────────

def extract_pages(pdf_path: Path) -> list[dict]:
    """
    Opens the PDF and reads text from every page.
    Uses pdfplumber which handles Indian gazette
    two-column format better than other libraries.

    Returns:
        List of dicts: [{"page_num": 1, "text": "..."}]
    """
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            # Crop top and bottom 45px to remove headers/footers
            cropped = page.crop((0, 45, page.width, page.height - 45))
            text = cropped.extract_text(x_tolerance=2, y_tolerance=3) or ""
            pages.append({
                "page_num": i,
                "text": text.strip()
            })

    log.info("pages_extracted", file=pdf_path.name, total=len(pages))
    return pages


# ── STEP 2: Send pages to GPT-4o to identify structure ───────────────

# This prompt tells GPT-4o exactly what we want
TREE_BUILD_SYSTEM = """You are a legal document analyst specialising in Indian law.
Analyse the pages of an Indian legal Act and identify its structure.

Return ONLY a valid JSON array. Each element is one structural node:
{
  "node_id": "unique string e.g. ch1_sec3",
  "type": "chapter" or "section" or "sub_section",
  "title": "exact heading from the document",
  "section_number": "e.g. Section 103 or Chapter III",
  "start_page": <integer>,
  "end_page": <integer>,
  "summary": "one sentence about what this section covers"
}

Rules:
- Include EVERY chapter and section you find.
- Use actual page numbers provided.
- Return ONLY the JSON array. No explanation. No markdown.
"""

# This prompt merges all batches into one final tree
TREE_MERGE_SYSTEM = """You are a legal document analyst.
You have multiple JSON arrays describing parts of an Indian legal Act.

Merge them into ONE complete hierarchical JSON tree.
Remove duplicates. Build parent-child relationships.

Return this exact structure:
{
  "act_code": "<ACT_CODE>",
  "act_full_name": "<FULL NAME>",
  "replaces": "<old law or null>",
  "effective_date": "<date>",
  "total_pages": <int>,
  "nodes": [
    {
      "node_id": "...",
      "type": "chapter",
      "title": "...",
      "section_number": "...",
      "start_page": <int>,
      "end_page": <int>,
      "summary": "...",
      "children": [
        {
          "node_id": "...",
          "type": "section",
          "title": "...",
          "section_number": "...",
          "start_page": <int>,
          "end_page": <int>,
          "summary": "...",
          "children": []
        }
      ]
    }
  ]
}

Return ONLY the JSON object. No explanation. No markdown.
"""


def _call_gpt4o(client: OpenAI, system: str, user: str, retries: int = 3) -> str:
    """
    Calls GPT-4o with retry logic.
    If it fails, waits and tries again up to 3 times.
    """
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0,
                max_tokens=4096,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            log.warning("gpt4o_retry", attempt=attempt + 1, error=str(exc))
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # wait 1s, 2s, 4s between retries
    raise RuntimeError("GPT-4o failed after all retries")


def _parse_json(text: str) -> Optional[dict | list]:
    """
    Safely parse JSON from GPT-4o response.
    Removes markdown code fences if present.
    """
    text = text.strip()
    # Remove ```json ... ``` if GPT-4o added them
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.error("json_parse_failed", error=str(e), preview=text[:200])
        return None


# ── STEP 3: Build the complete tree ──────────────────────────────────

def build_tree_index(act_code: str, pages: list[dict], client: OpenAI) -> dict:
    """
    Core PageIndex logic.

    Sends pages in batches of 15 to GPT-4o.
    GPT-4o identifies chapters and sections in each batch.
    Then merges all batches into one hierarchical tree.

    Returns the complete tree as a Python dict.
    """
    act_meta    = ACT_META[act_code]
    total_pages = len(pages)
    log.info("tree_build_start", act=act_code, total_pages=total_pages)

    all_batch_results = []

    # Process 15 pages at a time
    for batch_start in range(0, total_pages, PAGES_PER_BATCH):
        batch      = pages[batch_start: batch_start + PAGES_PER_BATCH]
        first_page = batch[0]["page_num"]
        last_page  = batch[-1]["page_num"]

        # Format pages as text for GPT-4o
        pages_text = "\n\n".join(
            f"=== PAGE {p['page_num']} ===\n{p['text']}"
            for p in batch
        )

        user_message = (
            f"Act: {act_meta['full_name']}\n"
            f"Pages {first_page} to {last_page} of {total_pages}:\n\n"
            f"{pages_text}"
        )

        log.info(
            "processing_batch",
            act=act_code,
            pages=f"{first_page}-{last_page}",
        )

        result = _call_gpt4o(client, TREE_BUILD_SYSTEM, user_message)
        all_batch_results.append(result)

        # Small delay to avoid hitting API rate limits
        time.sleep(0.5)

    # Merge all batches into one tree
    log.info("merging_batches", act=act_code, total_batches=len(all_batch_results))

    merge_user = (
        f"Act code: {act_code}\n"
        f"Full name: {act_meta['full_name']}\n"
        f"Replaces: {act_meta.get('replaces') or 'N/A'}\n"
        f"Effective: {act_meta['effective']}\n"
        f"Total pages: {total_pages}\n\n"
        f"Merge these {len(all_batch_results)} batch results:\n\n"
        + "\n\n--- BATCH SEPARATOR ---\n\n".join(all_batch_results)
    )

    merged_text = _call_gpt4o(client, TREE_MERGE_SYSTEM, merge_user)
    tree = _parse_json(merged_text)

    if not tree:
        log.warning("merge_failed_using_fallback", act=act_code)
        tree = _build_fallback_tree(act_code, all_batch_results, act_meta, total_pages)

    log.info("tree_build_complete", act=act_code)
    return tree


def _build_fallback_tree(
    act_code: str,
    batch_results: list[str],
    act_meta: dict,
    total_pages: int,
) -> dict:
    """
    If the merge step fails, build a simple flat tree
    by collecting all nodes from all batches.
    """
    all_nodes  = []
    seen       = set()

    for batch_text in batch_results:
        parsed = _parse_json(batch_text)
        if isinstance(parsed, list):
            for node in parsed:
                sec = node.get("section_number", "")
                if sec not in seen:
                    seen.add(sec)
                    node.setdefault("children", [])
                    all_nodes.append(node)

    return {
        "act_code":      act_code,
        "act_full_name": act_meta["full_name"],
        "replaces":      act_meta.get("replaces"),
        "effective_date": act_meta["effective"],
        "total_pages":   total_pages,
        "nodes":         all_nodes,
    }


# ── STEP 4: Save the index files ─────────────────────────────────────

def save_index(
    act_code:  str,
    tree:      dict,
    pages:     list[dict],
    index_dir: Path,
) -> None:
    """
    Saves two files to data/index/:
      BNS_index.json  ← the tree structure
      BNS_pages.json  ← all page texts
    """
    index_dir.mkdir(parents=True, exist_ok=True)

    tree_path  = index_dir / f"{act_code}_index.json"
    pages_path = index_dir / f"{act_code}_pages.json"

    tree_path.write_text(json.dumps(tree,  indent=2, ensure_ascii=False))
    pages_path.write_text(json.dumps(pages, indent=2, ensure_ascii=False))

    log.info(
        "index_saved",
        act=act_code,
        nodes=len(tree.get("nodes", [])),
        pages=len(pages),
    )


# ── Public entry point ────────────────────────────────────────────────

def build_act_index(act_code: str, file_path: Path, index_dir: Path) -> dict:
    """
    Main function — builds index for one Act.
    Called from CLI or from build_all_indexes().
    """
    settings = get_settings()
    client   = OpenAI(api_key=settings.openai_api_key)

    act_code = act_code.upper()
    if act_code not in ACT_META:
        raise ValueError(f"Unknown act: {act_code}. Choose from {list(ACT_META)}")
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    pages = extract_pages(file_path)
    tree  = build_tree_index(act_code, pages, client)
    save_index(act_code, tree, pages, index_dir)
    return tree


def build_all_indexes(raw_dir: Path, index_dir: Path) -> dict[str, int]:
    """
    Builds index for ALL Acts found in raw_dir.
    Skips Acts whose PDF is not found.
    """
    results = {}

    for act_code, meta in ACT_META.items():
        found = False
        for stem in meta["file_stems"]:
            for ext in [".pdf", ".txt"]:
                candidate = raw_dir / f"{stem}{ext}"
                if candidate.exists():
                    tree = build_act_index(act_code, candidate, index_dir)
                    results[act_code] = len(tree.get("nodes", []))
                    found = True
                    break
            if found:
                break

        if not found:
            log.warning("pdf_not_found", act=act_code, looked_in=str(raw_dir))

    return results


# ── CLI entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build PageIndex tree from Indian legal Act PDFs"
    )
    parser.add_argument(
        "--act",
        choices=[*ACT_META.keys(), "ALL"],
        default="ALL",
        help="Which Act to process (default: ALL)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Path to PDF file (required when --act is not ALL)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Folder containing PDFs (default: data/raw)",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/index"),
        help="Folder to save index files (default: data/index)",
    )
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.act == "ALL":
        results = build_all_indexes(args.raw_dir, args.index_dir)
        for act, nodes in results.items():
            print(f"  {act}: {nodes} nodes in tree")
    else:
        if not args.file:
            print("ERROR: --file is required when --act is not ALL")
            sys.exit(1)
        tree = build_act_index(args.act, args.file, args.index_dir)
        print(f"  {args.act}: {len(tree.get('nodes', []))} nodes in tree")