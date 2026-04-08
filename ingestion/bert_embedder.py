"""
ingestion/bert_embedder.py
──────────────────────────────────────────────
WHAT THIS FILE DOES:
  Reads the PageIndex tree (BNS_index.json) and
  generates a BERT embedding for every section.
  Saves embeddings as BNS_embeddings.pkl

WHY WE NEED IT:
  Embeddings are what make retrieval smart.
  When a lawyer asks a question, we convert the
  question to the same format and find the most
  similar sections. BERT understands legal context
  not just keyword matching.

WHAT IT PRODUCES:
  data/index/BNS_embeddings.pkl
  data/index/BNSS_embeddings.pkl
  data/index/BSA_embeddings.pkl
  data/index/DPDP_embeddings.pkl

HOW TO RUN (after page_index_builder finishes):
  python -m ingestion.bert_embedder --act ALL
  python -m ingestion.bert_embedder --act BNS
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import BertTokenizer, BertModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.core.logger import get_logger, setup_logging

log = get_logger(__name__)

# ── BERT model — loaded once and reused ──────────────────────────────
# Loading BERT takes ~10 seconds. We load it once and keep in memory.
_tokenizer: Optional[BertTokenizer] = None
_model:     Optional[BertModel]     = None


def _get_bert():
    """
    Loads BERT model the first time it is called.
    After that returns the cached version.
    Uses bert-base-uncased — 768 dimensions, good for legal text.
    """
    global _tokenizer, _model
    if _tokenizer is None:
        log.info("bert_loading", model="bert-base-uncased")
        _tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        _model     = BertModel.from_pretrained("bert-base-uncased")
        _model.eval()  # put in evaluation mode (not training mode)
        log.info("bert_loaded")
    return _tokenizer, _model


def embed_text(text: str) -> np.ndarray:
    """
    Converts a piece of text into a 768-number embedding.

    How it works:
      1. Tokenize text (split into word pieces)
      2. Run through BERT
      3. Mean pool all token embeddings → one vector
      4. Return numpy array of shape (768,)

    Why mean pooling?
      BERT gives one embedding per token (word piece).
      We average all of them to get one embedding for
      the whole section. This works better than just
      taking the [CLS] token for long legal text.
    """
    tokenizer, model = _get_bert()

    # Tokenize — BERT max is 512 tokens
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )

    # Run through BERT — no gradient needed (we are not training)
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling
    token_embeddings = outputs.last_hidden_state  # shape: (1, seq_len, 768)
    attention_mask   = inputs["attention_mask"]    # shape: (1, seq_len)

    # Expand mask so we can multiply with embeddings
    mask_expanded    = attention_mask.unsqueeze(-1).float()
    sum_embeddings   = (token_embeddings * mask_expanded).sum(dim=1)
    sum_mask         = mask_expanded.sum(dim=1).clamp(min=1e-9)
    mean_embedding   = sum_embeddings / sum_mask

    return mean_embedding.squeeze().numpy()  # shape: (768,)


# ── Tree utilities ────────────────────────────────────────────────────

def _flatten_nodes(nodes: list[dict], result: list | None = None) -> list[dict]:
    """
    Flattens the hierarchical tree into a simple flat list.

    Example:
      Tree:  Chapter I → Section 1, Section 2
      Flat:  [Chapter I, Section 1, Section 2]
    """
    if result is None:
        result = []
    for node in nodes:
        result.append(node)
        if node.get("children"):
            _flatten_nodes(node["children"], result)
    return result


def _get_section_text(node: dict, pages: list[dict]) -> str:
    """
    Builds the text to embed for one section node.

    Combines:
      - Section number (e.g. "Section 103")
      - Section title (e.g. "Punishment for murder")
      - Summary (one line description)
      - Actual page text from the PDF

    Richer text = better embedding = smarter retrieval.
    """
    parts = []

    # Add metadata as context
    if node.get("section_number"):
        parts.append(f"Section: {node['section_number']}")
    if node.get("title"):
        parts.append(f"Title: {node['title']}")
    if node.get("summary"):
        parts.append(f"Summary: {node['summary']}")

    # Add actual page content
    start_page  = node.get("start_page", 1)
    end_page    = node.get("end_page", start_page)
    page_lookup = {p["page_num"]: p["text"] for p in pages}

    page_texts = []
    for pnum in range(start_page, end_page + 1):
        if pnum in page_lookup:
            page_texts.append(page_lookup[pnum])

    if page_texts:
        parts.append("Content: " + " ".join(page_texts))

    full_text = " | ".join(parts)

    # BERT handles max 512 tokens (~380 words) — truncate if needed
    words = full_text.split()
    if len(words) > 350:
        full_text = " ".join(words[:350])

    return full_text


# ── Main embedding function ───────────────────────────────────────────

def embed_act(act_code: str, index_dir: Path) -> int:
    """
    Generates BERT embeddings for every section in one Act.
    Saves as {act_code}_embeddings.pkl

    Returns number of sections embedded.
    """
    act_code   = act_code.upper()
    tree_path  = index_dir / f"{act_code}_index.json"
    pages_path = index_dir / f"{act_code}_pages.json"
    emb_path   = index_dir / f"{act_code}_embeddings.pkl"

    # Check files exist
    if not tree_path.exists():
        log.error("tree_missing", act=act_code, path=str(tree_path))
        print(f"ERROR: {tree_path} not found. Run page_index_builder first.")
        return 0

    if not pages_path.exists():
        log.error("pages_missing", act=act_code, path=str(pages_path))
        return 0

    # Load tree and pages
    tree  = json.loads(tree_path.read_text(encoding="utf-8"))
    pages = json.loads(pages_path.read_text(encoding="utf-8"))

    # Flatten tree to get all nodes
    all_nodes = _flatten_nodes(tree.get("nodes", []))
    log.info("embedding_start", act=act_code, total_nodes=len(all_nodes))
    print(f"\n  Embedding {len(all_nodes)} sections for {act_code}...")

    embeddings = []

    for i, node in enumerate(all_nodes):
        # Build text for this section
        text = _get_section_text(node, pages)
        if not text.strip():
            continue

        # Generate embedding
        embedding = embed_text(text)

        # Save everything we need for retrieval later
        embeddings.append({
            "node_id":        node["node_id"],
            "act_code":       act_code,
            "section_number": node.get("section_number", ""),
            "title":          node.get("title", ""),
            "start_page":     node.get("start_page"),
            "end_page":       node.get("end_page"),
            "summary":        node.get("summary", ""),
            "type":           node.get("type", "section"),
            "embedding":      embedding,  # numpy array (768,)
        })

        # Show progress every 10 sections
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(all_nodes)} sections done...")

    # Save all embeddings to disk
    with open(emb_path, "wb") as f:
        pickle.dump(embeddings, f)

    log.info("embedding_complete", act=act_code, sections=len(embeddings))
    print(f"  Saved {len(embeddings)} embeddings to {emb_path}")
    return len(embeddings)


def embed_all(index_dir: Path) -> dict[str, int]:
    """
    Embeds all Acts whose index files exist in index_dir.
    Skips Acts that have not been indexed yet.
    """
    act_codes = ["BNS", "BNSS", "BSA", "DPDP"]
    results   = {}

    for act in act_codes:
        if (index_dir / f"{act}_index.json").exists():
            results[act] = embed_act(act, index_dir)
        else:
            log.warning("index_missing_skipping", act=act)
            print(f"  Skipping {act} — index file not found")

    return results


# ── CLI entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate BERT embeddings for PageIndex sections"
    )
    parser.add_argument(
        "--act",
        choices=["BNS", "BNSS", "BSA", "DPDP", "ALL"],
        default="ALL",
        help="Which Act to embed (default: ALL)",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/index"),
        help="Folder containing index JSON files (default: data/index)",
    )
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    setup_logging(args.log_level)

    print("\nBERT Embedder — Legal RAG Chatbot")
    print("=" * 40)

    if args.act == "ALL":
        results = embed_all(args.index_dir)
        print("\nDone!")
        for act, n in results.items():
            print(f"  {act}: {n} sections embedded")
    else:
        n = embed_act(args.act, args.index_dir)
        print(f"\nDone! {args.act}: {n} sections embedded")