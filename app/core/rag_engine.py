"""
app/core/rag_engine.py
──────────────────────────────────────────────
WHAT THIS FILE DOES:
  This is the heart of the entire system.
  Every time a lawyer asks a question, this file runs.

THE 4 STEPS IT DOES:
  1. LAW ROUTER   → which Act(s) to search?
  2. BERT         → encode question → find top 5 relevant sections
  3. FETCH PAGES  → get exact page texts for those sections
  4. GPT-4o       → read pages → write cited answer

WHY THIS DESIGN:
  BERT handles retrieval (fast, free, runs locally)
  GPT-4o handles generation (one API call only)
  This gives best accuracy at lowest cost.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from openai import OpenAI
from transformers import BertTokenizer, BertModel

from app.core.config import get_settings
from app.core.logger import get_logger

log = get_logger(__name__)


# ── Response dataclass ────────────────────────────────────────────────
# This is what the function returns after answering a question

@dataclass
class LegalAnswer:
    answer:      str         # the full answer text
    citations:   list[dict]  # list of sections cited
    act_scope:   list[str]   # which Acts were searched
    query_hash:  str         # unique hash of the question
    error:       Optional[str] = None  # error message if something went wrong


# ── GPT-4o prompts ────────────────────────────────────────────────────

ANSWER_SYSTEM = """You are a senior legal assistant specialising in Indian law.
You will be given exact pages from an Indian legal Act and a lawyer's question.

You MUST follow these rules:
1. Answer clearly using ONLY the provided legal text.
2. Cite every claim with exact section: [Act Name, Section X]
   Examples: [BNS, Section 103]  [DPDP Act, Section 9]  [BNSS, Section 173]
3. If a provision changed from old law (IPC/CrPC/Evidence Act), mention the change.
4. If something is not in the provided text, say "not found in provided sections."
5. End your answer with "Key Sections Referenced:" and a bullet list.

NEVER fabricate section numbers.
NEVER use general knowledge to fill gaps.
ONLY use the text provided."""

ANSWER_USER = """Act(s): {act_names}

Relevant legal sections retrieved for your reference:
{pages_content}

Lawyer's question: {question}

Please answer with exact section citations:"""


# ── BERT — load once, reuse forever ──────────────────────────────────

_bert_tokenizer: Optional[BertTokenizer] = None
_bert_model:     Optional[BertModel]     = None


def _get_bert():
    """Loads BERT once. Returns cached version after first load."""
    global _bert_tokenizer, _bert_model
    if _bert_tokenizer is None:
        log.info("bert_loading")
        _bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        _bert_model     = BertModel.from_pretrained("bert-base-uncased")
        _bert_model.eval()
        log.info("bert_ready")
    return _bert_tokenizer, _bert_model


def _bert_embed(text: str) -> np.ndarray:
    """
    Converts text to 768-number BERT embedding.
    Same function as in bert_embedder.py — must stay identical.
    """
    tokenizer, model = _get_bert()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    token_embeddings = outputs.last_hidden_state
    attention_mask   = inputs["attention_mask"]
    mask_expanded    = attention_mask.unsqueeze(-1).float()
    sum_embeddings   = (token_embeddings * mask_expanded).sum(dim=1)
    sum_mask         = mask_expanded.sum(dim=1).clamp(min=1e-9)

    return (sum_embeddings / sum_mask).squeeze().numpy()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Measures how similar two embeddings are.
    Returns a number between -1 and 1.
    1.0  = identical meaning
    0.0  = unrelated
    -1.0 = opposite meaning

    We use this to find which sections are most
    similar in meaning to the lawyer's question.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ── Index loaders — cached so files are read only once ───────────────

@lru_cache(maxsize=8)
def _load_tree(act_code: str) -> Optional[dict]:
    """Loads BNS_index.json — the PageIndex tree."""
    settings = get_settings()
    path = Path(settings.index_dir) / f"{act_code}_index.json"
    if not path.exists():
        log.error("tree_file_missing", act=act_code, path=str(path))
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def _load_pages(act_code: str) -> Optional[list]:
    """Loads BNS_pages.json — all page texts."""
    settings = get_settings()
    path = Path(settings.index_dir) / f"{act_code}_pages.json"
    if not path.exists():
        log.error("pages_file_missing", act=act_code, path=str(path))
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def _load_embeddings(act_code: str) -> Optional[list]:
    """Loads BNS_embeddings.pkl — BERT embeddings for each section."""
    settings = get_settings()
    path = Path(settings.index_dir) / f"{act_code}_embeddings.pkl"
    if not path.exists():
        log.error("embeddings_file_missing", act=act_code, path=str(path))
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ── STEP 1: Law Router ────────────────────────────────────────────────

# Keywords that tell us which Act to search
_ROUTING_HINTS: dict[str, list[str]] = {
    "BNS": [
        "murder", "theft", "fraud", "assault", "kidnapping", "rape",
        "bns", "nyaya sanhita", "ipc", "offense", "offence",
        "punishment", "culpable homicide", "criminal", "abetment",
        "cheating", "robbery", "dacoity", "sedition",
    ],
    "BNSS": [
        "arrest", "bail", "trial", "fir", "chargesheet", "magistrate",
        "police", "investigation", "custody", "remand", "cognizable",
        "bnss", "suraksha", "summons", "warrant", "search", "seizure",
        "crpc", "charge sheet", "non-cognizable", "anticipatory bail",
    ],
    "BSA": [
        "evidence", "witness", "admission", "confession", "document",
        "proof", "testimony", "electronic record", "digital evidence",
        "sakshya", "bsa", "burden of proof", "relevancy", "hearsay",
        "admissibility",
    ],
    "DPDP": [
        "personal data", "data protection", "privacy", "consent",
        "data fiduciary", "data principal", "dpdp", "digital data",
        "data breach", "right to erasure", "cross border",
        "data localisation", "processing",
    ],
}


def _route_query(question: str, act_filter: Optional[str]) -> list[str]:
    """
    Decides which Act(s) to search based on the question.

    If act_filter is set (e.g. "BNS") → search only that Act.
    Otherwise scan question for keywords and pick matching Acts.
    If no keywords match → search all 4 Acts.
    """
    if act_filter and act_filter != "ALL":
        return [act_filter.upper()]

    q       = question.lower()
    matched = [
        act for act, keywords in _ROUTING_HINTS.items()
        if any(keyword in q for keyword in keywords)
    ]
    return matched if matched else list(_ROUTING_HINTS.keys())


# ── STEP 2: BERT Retrieval ────────────────────────────────────────────

def _bert_retrieve(question: str, act_code: str, top_k: int = 5) -> list[dict]:
    """
    Finds the most relevant sections for a question.

    How it works:
      1. Load all section embeddings for this Act
      2. Encode the question with BERT → 768 numbers
      3. Compare question embedding vs every section embedding
      4. Filter out generic/preliminary sections
      5. Return top_k sections with highest similarity score
    """
    embeddings = _load_embeddings(act_code)
    if not embeddings:
        return []

    # Skip sections that are unlikely to answer legal questions
    # Section 1 (short title), Section 2 (definitions intro) etc.
    SKIP_SECTIONS = {
        "Section 1", "Section 2", "Section 3",
        "Chapter I", "Chapter II",
    }
    SKIP_TITLES = {
        "short title", "commencement", "application",
        "repeal", "savings", "extent",
    }

    # Enrich question with legal context for better BERT matching
    enriched_question = f"Indian law legal provision: {question}"

    # Encode the enriched question
    question_embedding = _bert_embed(enriched_question)

    # Score every section
    scored = []
    for entry in embeddings:
        sec_num = entry.get("section_number", "")
        title   = entry.get("title", "").lower()

        # Skip generic preliminary sections
        if sec_num in SKIP_SECTIONS:
            continue
        if any(skip in title for skip in SKIP_TITLES):
            continue

        score = _cosine_similarity(question_embedding, entry["embedding"])

        # Boost sections that contain keywords from the question
        question_words = set(question.lower().split())
        title_words    = set(title.lower().split())
        overlap        = question_words & title_words
        if overlap:
            score += 0.05 * len(overlap)  # small boost per matching word

        scored.append({**entry, "score": score})

    # Sort by score, return top K
    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:top_k]

    log.info(
        "bert_retrieval_done",
        act=act_code,
        top_section=top[0]["section_number"] if top else "none",
        top_score=round(top[0]["score"], 3) if top else 0,
    )
    return top


# ── STEP 3: Fetch exact page texts ───────────────────────────────────

def _fetch_pages(sections: list[dict], act_code: str) -> tuple[str, list[dict]]:
    """
    Gets the actual page texts for the retrieved sections.

    Example:
      Section 103 is on pages 21-22
      → Fetch page 21 text and page 22 text
      → Combine them

    Returns:
      (formatted_text_string, citations_list)
    """
    pages = _load_pages(act_code)
    tree  = _load_tree(act_code)
    if not pages:
        return "", []

    page_lookup  = {p["page_num"]: p["text"] for p in pages}
    seen_pages: set[int] = set()
    content_blocks: list[str] = []
    citations: list[dict] = []

    for sec in sections:
        start = sec.get("start_page") or 1
        end   = sec.get("end_page") or start

        page_texts = []
        for pnum in range(start, end + 1):
            if pnum not in seen_pages and pnum in page_lookup:
                seen_pages.add(pnum)
                page_texts.append(f"[Page {pnum}]\n{page_lookup[pnum]}")

        if page_texts:
            # Format section content
            block = (
                f"--- {sec.get('section_number', '')} "
                f"— {sec.get('title', '')} ---\n"
                + "\n".join(page_texts)
            )
            content_blocks.append(block)

            # Build citation entry
            act_full_name = (
                tree.get("act_full_name", act_code) if tree else act_code
            )
            citations.append({
                "act":            act_code,
                "act_full_name":  act_full_name,
                "section_number": sec.get("section_number", "?"),
                "title":          sec.get("title", ""),
                "pages":          f"{start}–{end}",
                "score":          round(sec.get("score", 0), 4),
                "summary":        sec.get("summary", ""),
            })

    return "\n\n".join(content_blocks), citations


# ── STEP 4: GPT-4o Answer Generation ─────────────────────────────────

def _call_gpt4o(client: OpenAI, system: str, user: str) -> str:
    """
    Calls GPT-4o with retry logic.
    This is called ONCE per query — only for the final answer.
    """
    settings = get_settings()
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=settings.openai_llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=settings.openai_temperature,
                max_tokens=settings.openai_max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            log.warning("gpt4o_retry", attempt=attempt + 1, error=str(exc))
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError("GPT-4o failed after 3 retries")


# ── Main query function ───────────────────────────────────────────────

async def query_legal(
    question:   str,
    act_filter: Optional[str] = None,
    session_id: Optional[str] = None,
) -> LegalAnswer:
    """
    THE MAIN FUNCTION — called by FastAPI for every question.

    Full flow:
      1. Route → which Act(s) to search
      2. BERT  → find top 5 relevant sections per Act
      3. Fetch → get exact page texts for those sections
      4. GPT-4o → write cited answer from page texts

    Parameters:
      question   : the lawyer's question
      act_filter : "BNS", "BNSS", "BSA", "DPDP", "ALL", or None (auto)
      session_id : for logging only

    Returns:
      LegalAnswer with answer text + citations list
    """
    settings = get_settings()
    client   = OpenAI(api_key=settings.openai_api_key)

    # Create a unique hash for this question (used for caching)
    query_hash = hashlib.sha256(
        f"{question}:{act_filter}".encode()
    ).hexdigest()[:16]

    # Step 1 — Route
    acts = _route_query(question, act_filter)
    log.info("query_start", question=question[:80], acts=acts)

    all_content:   list[str]  = []
    all_citations: list[dict] = []
    acts_searched: list[str]  = []

    for act_code in acts:
        # Check all required files exist
        if not _load_embeddings(act_code):
            log.warning("embeddings_not_ready", act=act_code)
            continue
        if not _load_tree(act_code):
            continue

        acts_searched.append(act_code)

        # Step 2 — BERT retrieval
        top_sections = _bert_retrieve(
            question, act_code, top_k=settings.retrieval_top_k
        )
        if not top_sections:
            continue

        # Step 3 — Fetch pages
        content, citations = _fetch_pages(top_sections, act_code)
        if content:
            tree = _load_tree(act_code)
            act_name = tree.get("act_full_name", act_code) if tree else act_code
            all_content.append(f"=== {act_name} ===\n{content}")
            all_citations.extend(citations)

    # No results — index not built yet
    if not all_content:
        return LegalAnswer(
            answer=(
                "The index has not been built yet.\n\n"
                "Please run these commands first:\n"
                "  python -m ingestion.page_index_builder --act ALL\n"
                "  python -m ingestion.bert_embedder --act ALL"
            ),
            citations=[],
            act_scope=acts,
            query_hash=query_hash,
            error="index_not_built",
        )

    # Step 4 — GPT-4o final answer
    act_names = ", ".join(
        (_load_tree(a) or {}).get("act_full_name", a)
        for a in acts_searched
    )

    answer_text = _call_gpt4o(
        client,
        ANSWER_SYSTEM,
        ANSWER_USER.format(
            act_names=act_names,
            pages_content="\n\n".join(all_content),
            question=question,
        ),
    )

    log.info(
        "query_complete",
        acts=acts_searched,
        citations=len(all_citations),
        hash=query_hash,
    )

    return LegalAnswer(
        answer=answer_text,
        citations=all_citations,
        act_scope=acts_searched,
        query_hash=query_hash,
    )