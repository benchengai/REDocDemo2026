"""
ask_requirements.py
RAG module for the Ask Requirements page.

No extra dependencies — uses only the existing google-genai SDK
and Python standard library (math).

Pipeline:
  1. build_chunks()     — split processed report JSON into text chunks
  2. RequirementIndex   — embed chunks + store vectors in memory
  3. answer()           — retrieve top-k chunks, call Gemini to answer
"""

import math
import os
from pathlib import Path

from dotenv import load_dotenv
import google.genai as genai
import google.genai.types as gtypes

# ── Models ────────────────────────────────────────────────────────────────────

EMBED_MODEL  = "gemini-embedding-001"
ANSWER_MODEL = "gemini-2.5-pro"

# ── Embedding client (v1beta — gemini-embedding-001 works with v1beta) ────────
# Both the answer model and the embedding model use the same v1beta API.
# We keep a lazy singleton so we don't create a new client on every call.

load_dotenv()
_embed_client: genai.Client | None = None


def _get_embed_client() -> genai.Client:
    global _embed_client
    if _embed_client is None:
        api_key = os.getenv("GEMINI_API_KEY", "")
        _embed_client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1beta"},
        )
    return _embed_client

# ── Chunking ──────────────────────────────────────────────────────────────────

def build_chunks(data: dict, meeting_id: str) -> list[dict]:
    """
    Convert one processed report JSON into a flat list of text chunks.
    Each chunk is  {"id": str, "text": str, "meta": dict}.
    """
    chunks = []
    summary = data.get("meeting_summary", {})

    # ── Meeting overview ───────────────────────────────────────────────────────
    dur_s = summary.get("duration_seconds", 0)
    dur_str = f"{dur_s // 60}m {dur_s % 60}s" if isinstance(dur_s, int) else str(dur_s)
    chunks.append({
        "id":   f"{meeting_id}::summary",
        "text": (
            f"Meeting {meeting_id} — overview. "
            f"Summary: {summary.get('brief', '')} "
            f"Key themes: {', '.join(summary.get('key_themes', []))}. "
            f"Overall sentiment: {summary.get('overall_sentiment', '')}. "
            f"Dominant emotions: {', '.join(summary.get('dominant_emotions', []))}. "
            f"Duration: {dur_str}. "
            f"Number of speakers: {summary.get('num_speakers', '?')}. "
            f"Language: {summary.get('language', '?')}."
        ),
        "meta": {"meeting_id": meeting_id, "type": "summary"},
    })

    # ── Functional Requirements ────────────────────────────────────────────────
    for req in data.get("requirements", []):
        rid  = req.get("id", "?")
        ac   = "; ".join(req.get("acceptance_criteria", []))
        stk  = ", ".join(req.get("stakeholders", []))
        emos = ", ".join(req.get("detected_emotions", []))
        chunks.append({
            "id":   f"{meeting_id}::{rid}",
            "text": (
                f"[{meeting_id}] {rid} — {req.get('title', '')}. "
                f"Type: Functional Requirement. "
                f"Priority: {req.get('priority', '')}. "
                f"Conflict level: {req.get('conflict_level', 'none')}. "
                f"Unresolved: {req.get('unresolved', False)}. "
                f"Statement: {req.get('statement', '')} "
                f"Acceptance criteria: {ac}. "
                f"First mentioned at {req.get('first_mentioned', '?')} in the recording. "
                f"Stakeholders: {stk}. "
                f"Detected emotions: {emos}. "
                f"Evidence: {req.get('evidence', '')}."
            ),
            "meta": {
                "meeting_id":    meeting_id,
                "type":          "FR",
                "id":            rid,
                "priority":      req.get("priority", ""),
                "conflict_level":req.get("conflict_level", "none"),
                "unresolved":    req.get("unresolved", False),
                "first_mentioned": req.get("first_mentioned", ""),
            },
        })

    # ── Non-Functional Requirements ───────────────────────────────────────────
    for nfr in data.get("non_functional_requirements", []):
        rid = nfr.get("id", "?")
        ac  = "; ".join(nfr.get("acceptance_criteria", []))
        emos = ", ".join(nfr.get("detected_emotions", []))
        chunks.append({
            "id":   f"{meeting_id}::{rid}",
            "text": (
                f"[{meeting_id}] {rid} — {nfr.get('title', '')}. "
                f"Type: Non-Functional Requirement. "
                f"Category: {nfr.get('category', '')}. "
                f"Priority: {nfr.get('priority', '')}. "
                f"Statement: {nfr.get('statement', '')} "
                f"Acceptance criteria: {ac}. "
                f"First mentioned at {nfr.get('first_mentioned', '?')} in the recording. "
                f"Detected emotions: {emos}. "
                f"Evidence: {nfr.get('evidence', '')}."
            ),
            "meta": {
                "meeting_id":    meeting_id,
                "type":          "NFR",
                "id":            rid,
                "priority":      nfr.get("priority", ""),
                "category":      nfr.get("category", ""),
                "first_mentioned": nfr.get("first_mentioned", ""),
            },
        })

    # ── Open Issues ───────────────────────────────────────────────────────────
    for oi in data.get("open_issues", []):
        oid = oi.get("id", "?")
        chunks.append({
            "id":   f"{meeting_id}::{oid}",
            "text": (
                f"[{meeting_id}] {oid} — {oi.get('description', '')}. "
                f"Type: Open Issue. "
                f"Conflict level: {oi.get('conflict_level', '')}. "
                f"Raised by: {', '.join(oi.get('raised_by', []))}. "
                f"Related requirements: {', '.join(oi.get('related_requirements', []))}. "
                f"Suggested resolution: {oi.get('suggested_resolution', '')}."
            ),
            "meta": {
                "meeting_id":    meeting_id,
                "type":          "open_issue",
                "id":            oid,
                "conflict_level":oi.get("conflict_level", ""),
            },
        })

    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

def _embed_one(text: str, client: genai.Client | None = None) -> list[float]:
    """Embed a single text using the v1 embedding client."""
    result = _get_embed_client().models.embed_content(
        model=EMBED_MODEL, contents=text
    )
    return list(result.embeddings[0].values)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


# ── Index ─────────────────────────────────────────────────────────────────────

class RequirementIndex:
    """
    In-memory vector index built from one or more processed report JSONs.

    Usage:
        idx = RequirementIndex()
        idx.build({"ES2014a": data_dict, ...}, client, progress_cb=st.progress)
        chunks = idx.search("high priority disputed requirements", client)
    """

    def __init__(self) -> None:
        self.chunks:  list[dict]        = []
        self.vectors: list[list[float]] = []
        self.meeting_ids: set[str]      = set()

    def build(
        self,
        reports: dict[str, dict],
        client: genai.Client,
        progress_cb=None,   # optional callable(float 0-1) for progress bar
    ) -> None:
        """
        Embed all chunks from the given reports dict.
        Replaces any existing index.
        """
        all_chunks: list[dict] = []
        for meeting_id, data in reports.items():
            all_chunks.extend(build_chunks(data, meeting_id))

        vectors: list[list[float]] = []
        total = len(all_chunks)
        for i, chunk in enumerate(all_chunks):
            vectors.append(_embed_one(chunk["text"], client))
            if progress_cb and total > 0:
                progress_cb((i + 1) / total)

        self.chunks      = all_chunks
        self.vectors     = vectors
        self.meeting_ids = set(reports.keys())

    @property
    def size(self) -> int:
        return len(self.chunks)

    def search(
        self,
        query: str,
        client: genai.Client,
        top_k: int = 8,
    ) -> list[dict]:
        """Return the top-k most relevant chunks for a query."""
        if not self.chunks:
            return []
        q_vec = _embed_one(query, client)
        scored = sorted(
            range(len(self.chunks)),
            key=lambda i: _cosine(q_vec, self.vectors[i]),
            reverse=True,
        )
        return [self.chunks[i] for i in scored[:top_k]]


# ── Answer ────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are an expert Requirements Engineering assistant with deep knowledge of IEEE 29148.
You answer questions about stakeholder meeting requirements documents.

You are given CONTEXT chunks retrieved from a vector index of one or more processed meeting reports.
Each chunk contains structured requirements data including IDs, statements, priorities, conflict levels, evidence, and timestamps (MM:SS) of when topics were first mentioned in the original recording.

ANSWERING RULES:
- Answer only from the provided context. If information is missing, say so.
- Always cite requirement IDs (e.g. FR-003, NFR-001, OI-002) when referencing specific items.
- For timestamp questions ("when was X discussed"): use the "first mentioned at MM:SS" field.
- For disputed/controversial items: look for conflict_level=high or unresolved=true.
- For priority ranking: order as high > medium > low, list IDs and titles.
- For gaps in requirements: reason based on themes and what typical systems need.
- For multi-meeting questions: clearly attribute which meeting each point comes from.
- Format answers cleanly. Use bullet points for lists, bold for IDs.
- Keep answers concise but complete. No filler phrases.
"""


def _build_turns(
    query: str,
    index: RequirementIndex,
    history: list[dict],
    client: genai.Client,
) -> tuple[list[gtypes.Content] | None, str]:
    """
    Retrieve context chunks and build conversation turns for Gemini.
    Returns (turns, error_message). If no chunks, turns is None.
    """
    chunks = index.search(query, client, top_k=8)
    if not chunks:
        return None, "No requirements documents are indexed. Please select meetings in the sidebar."

    context_parts = []
    for c in chunks:
        meta = c["meta"]
        tag  = f"{meta.get('type','?').upper()} · {meta.get('meeting_id','?')}"
        context_parts.append(f"[{tag}]\n{c['text']}")
    context = "\n\n---\n\n".join(context_parts)

    # Last 3 exchanges as conversation context (avoid very long histories)
    turns: list[gtypes.Content] = []
    for msg in history[-6:]:
        role = "user" if msg["role"] == "user" else "model"
        turns.append(gtypes.Content(role=role, parts=[gtypes.Part(text=msg["content"])]))

    user_text = (
        f"RETRIEVED CONTEXT:\n{context}"
        f"\n\n---\n\nQUESTION: {query}"
    )
    turns.append(gtypes.Content(role="user", parts=[gtypes.Part(text=user_text)]))
    return turns, ""


def answer(
    query: str,
    index: RequirementIndex,
    history: list[dict],
    client: genai.Client,
) -> str:
    """
    Retrieve relevant chunks and generate a grounded answer using Gemini.

    Args:
        query:   The user's question.
        index:   A built RequirementIndex.
        history: Chat history list of {"role": "user"|"assistant", "content": str}.
        client:  Authenticated genai.Client.

    Returns:
        Answer string from Gemini.
    """
    turns, err = _build_turns(query, index, history, client)
    if turns is None:
        return err

    response = client.models.generate_content(
        model=ANSWER_MODEL,
        contents=turns,
        config=gtypes.GenerateContentConfig(
            system_instruction=_SYSTEM,
            temperature=0.3,
            max_output_tokens=2048,
        ),
    )
    return response.text or "_(No response generated)_"


def stream_answer(
    query: str,
    index: RequirementIndex,
    history: list[dict],
    client: genai.Client,
):
    """
    Generator that streams the Gemini answer token by token.
    Yields text chunks as they arrive — pass directly to st.write_stream().

    Args:
        query:   The user's question.
        index:   A built RequirementIndex.
        history: Chat history list of {"role": "user"|"assistant", "content": str}.
        client:  Authenticated genai.Client.
    """
    turns, err = _build_turns(query, index, history, client)
    if turns is None:
        yield err
        return

    for chunk in client.models.generate_content_stream(
        model=ANSWER_MODEL,
        contents=turns,
        config=gtypes.GenerateContentConfig(
            system_instruction=_SYSTEM,
            temperature=0.3,
            max_output_tokens=2048,
        ),
    ):
        if chunk.text:
            yield chunk.text
