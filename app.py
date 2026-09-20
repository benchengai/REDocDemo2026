"""
app.py  —  RE Meeting Analysis · Streamlit Frontend
Pipeline: Select audio → Step 1 (Gemini audio analysis) → Step 2 (SRS generation) → Report
"""

import datetime
import io
import json
import os
import re
import sys
import time
import wave
from collections import defaultdict
from pathlib import Path

from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

import streamlit as st
from dotenv import load_dotenv

import google.genai as genai
import google.genai.types as gtypes

# ── env & API ─────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("GEMINI_API_KEY not found in .env")
    st.stop()

client = genai.Client(api_key=API_KEY)

AUDIO_DIR  = Path("data/audio")
OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_AUDIO   = "gemini-2.5-pro"
MODEL_SRS     = "gemini-2.5-pro"
CHUNK_MINUTES = 30       # Split audio into this many minutes per chunk

MIME_MAP = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".mp4": "video/mp4",
    ".m4a": "audio/mp4", ".flac": "audio/flac", ".ogg": "audio/ogg",
}

# ── Import prompts from existing scripts ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from step1_audio_analysis import PROMPT as AUDIO_PROMPT
from generate_srs import SYSTEM_PROMPT, build_user_prompt, attach_timestamps
from ask_requirements import RequirementIndex, answer as rag_answer, stream_answer as rag_stream

# ══════════════════════════════════════════════════════════════════════════════
# Pipeline functions
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# WAV chunking helpers (built-in wave module, no extra deps)
# ══════════════════════════════════════════════════════════════════════════════

def wav_duration_minutes(path: Path) -> float:
    """Return duration in minutes for a WAV file."""
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / wf.getframerate() / 60
    except Exception:
        return 0.0


def split_wav(path: Path, chunk_minutes: int = CHUNK_MINUTES) -> list[tuple[Path, int]]:
    """Split WAV into chunks. Returns [(chunk_path, offset_seconds), ...]."""
    chunks = []
    with wave.open(str(path), "rb") as wf:
        rate   = wf.getframerate()
        n_ch   = wf.getnchannels()
        sw     = wf.getsampwidth()
        chunk_frames = chunk_minutes * 60 * rate
        offset_sec = 0
        idx = 0
        while True:
            frames = wf.readframes(chunk_frames)
            if not frames:
                break
            chunk_path = OUTPUT_DIR / f"_chunk_{idx}_{path.stem}.wav"
            with wave.open(str(chunk_path), "wb") as out:
                out.setnchannels(n_ch)
                out.setsampwidth(sw)
                out.setframerate(rate)
                out.writeframes(frames)
            chunks.append((chunk_path, offset_sec))
            offset_sec += chunk_minutes * 60
            idx += 1
    return chunks


def offset_timestamps(utterances: list[dict], offset_sec: int) -> list[dict]:
    """Shift MM:SS timestamps in a list of utterances by offset_sec."""
    def shift(t: str) -> str:
        try:
            m, s = t.split(":")
            total = int(m) * 60 + int(s) + offset_sec
            return f"{total // 60:02d}:{total % 60:02d}"
        except Exception:
            return t
    for u in utterances:
        u["start_time"] = shift(u.get("start_time", "00:00"))
        u["end_time"]   = shift(u.get("end_time",   "00:00"))
    return utterances


def cleanup_chunks(chunks: list[tuple[Path, int]]) -> None:
    for path, _ in chunks:
        try:
            path.unlink()
        except Exception:
            pass


def upload_audio(path: Path, status) -> gtypes.File:
    mime = MIME_MAP[path.suffix.lower()]
    status.update(label=f"Uploading {path.name} ({path.stat().st_size/1_048_576:.1f} MB)…")
    with open(path, "rb") as f:
        uploaded = client.files.upload(
            file=f,
            config=gtypes.UploadFileConfig(mime_type=mime, display_name=path.name),
        )
    status.update(label="Waiting for Gemini to process audio…")
    while uploaded.state.name == "PROCESSING":
        time.sleep(5)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state.name != "ACTIVE":
        raise RuntimeError(f"File processing failed: {uploaded.state.name}")
    return uploaded


MAX_RETRIES = 3   # max Gemini call attempts per file (handles stray JSON errors)


def _repair_json(raw: str) -> str:
    """Fix known Gemini JSON syntax typos without external libraries.

    Observed patterns in production:
    1. Merged speaker key+value: "speaker_B" → "speaker": "Speaker_B"
    2. Missing closing quote on MM:SS timestamps: "08:56, → "08:56",
    3. Merged emotion key+value: "emotion_calm" → "emotion": "calm"
    4. Trailing comma before closing bracket
    """
    # 1. "speaker_X" → "speaker": "Speaker_X"
    raw = re.sub(r'"speaker_([A-Za-z0-9]+)"', r'"speaker": "Speaker_\1"', raw)

    # 2. Missing closing quote on timestamps — "MM:SS, → "MM:SS",
    raw = re.sub(r'"(\d{2}:\d{2}),', r'"\1",', raw)

    # 3. Merged emotion key+value
    known_emotions = "anxious|calm|discouraged|enthusiastic|excited|frightened|furious|satisfied"
    raw = re.sub(rf'"emotion_({known_emotions})"', r'"emotion": "\1"', raw)

    # 4. Trailing commas before } or ]
    raw = re.sub(r",(\s*[}\]])", r"\1", raw)

    return raw


def _parse_json_response(raw: str, debug_path: Path) -> dict:
    """Strip markdown fences, try to parse JSON, auto-repair on failure."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    # First attempt: parse as-is
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return {"meeting_meta": {}, "utterances": data}
        return data
    except json.JSONDecodeError:
        pass

    # Second attempt: repair then parse
    try:
        data = json.loads(_repair_json(raw))
        if isinstance(data, list):
            return {"meeting_meta": {}, "utterances": data}
        return data
    except json.JSONDecodeError as e:
        debug_path.write_text(raw, encoding="utf-8")
        raise RuntimeError(
            f"JSON parse error at char {e.pos}: {e.msg}. Raw saved → {debug_path}"
        ) from e


def analyse_single_file(path: Path, status, label: str) -> dict:
    """Upload one audio file, run Gemini analysis, return full parsed dict.
    Retries up to MAX_RETRIES times if Gemini returns malformed JSON.
    """
    audio_file = upload_audio(path, status)
    try:
        for attempt in range(1, MAX_RETRIES + 1):
            if attempt == 1:
                status.update(label=label)
            else:
                status.update(label=f"{label} (retry {attempt - 1}/{MAX_RETRIES - 1})…")
                st.write(f"⚠️ JSON malformed — retrying ({attempt - 1}/{MAX_RETRIES - 1})…")

            response = client.models.generate_content(
                model=MODEL_AUDIO,
                contents=[
                    gtypes.Part.from_uri(file_uri=audio_file.uri, mime_type=audio_file.mime_type),
                    AUDIO_PROMPT,
                ],
                config=gtypes.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=65536,
                ),
            )
            if response.text is None:
                candidate = response.candidates[0] if response.candidates else None
                reason = getattr(candidate, "finish_reason", "unknown") if candidate else "none"
                raise RuntimeError(f"Gemini returned no text. finish_reason={reason}")

            debug_path = OUTPUT_DIR / f"_step1_raw_{path.stem}.txt"
            try:
                return _parse_json_response(response.text, debug_path)
            except RuntimeError:
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(2)   # brief pause before retry

    finally:
        try:
            client.files.delete(name=audio_file.name)
        except Exception:
            pass


def run_step1(audio_path: Path, status) -> dict:
    """Upload audio → Gemini analysis → structured JSON.
    Automatically splits WAV files longer than CHUNK_MINUTES.
    """
    is_wav    = audio_path.suffix.lower() == ".wav"
    duration  = wav_duration_minutes(audio_path) if is_wav else 0

    if is_wav and duration > CHUNK_MINUTES:
        status.update(label=f"Audio is {duration:.0f} min — splitting into {CHUNK_MINUTES}-min chunks…")
        chunks = split_wav(audio_path, CHUNK_MINUTES)
        st.write(f"🔪 Split {audio_path.name} into {len(chunks)} chunks ({CHUNK_MINUTES} min each)")
        try:
            all_utterances: list[dict] = []
            utt_id  = 1
            language = "Unknown"
            for i, (chunk_path, offset_sec) in enumerate(chunks, 1):
                end_min = min((offset_sec + CHUNK_MINUTES * 60) // 60, int(duration))
                label = (
                    f"Analysing chunk {i}/{len(chunks)} "
                    f"({offset_sec//60:02d}:00 – {end_min:02d}:00)…"
                )
                st.write(f"🎵 {label}")
                chunk_data = analyse_single_file(chunk_path, status, label)
                # Take language from the first chunk's meta
                if i == 1:
                    language = chunk_data.get("meeting_meta", {}).get("language", "Unknown")
                utts = chunk_data.get("utterances", [])
                utts = offset_timestamps(utts, offset_sec)
                # Re-number IDs to be continuous across chunks
                for u in utts:
                    u["id"] = utt_id
                    utt_id += 1
                all_utterances.extend(utts)
        finally:
            cleanup_chunks(chunks)

        return {
            "meeting_meta": {
                "duration_seconds": int(duration * 60),
                "num_speakers": len({u.get("speaker") for u in all_utterances}),
                "language": language,
            },
            "utterances": all_utterances,
        }

    # Short audio — process in one shot (with retry)
    return analyse_single_file(
        audio_path, status,
        "Analysing audio with Gemini (emotion · transcript · speaker)…"
    )


def run_step2(step1_data: dict, meeting_id: str, status) -> dict:
    """Step 1 JSON → Gemini SRS generation → return parsed JSON.
    Retries up to MAX_RETRIES times if Gemini returns malformed JSON.
    """
    meta        = step1_data.get("meeting_meta", {})
    utterances  = step1_data.get("utterances", [])
    user_prompt = build_user_prompt(meeting_id, meta, utterances)
    debug_path  = OUTPUT_DIR / "_srs_raw_response.txt"

    for attempt in range(1, MAX_RETRIES + 1):
        if attempt == 1:
            status.update(label="Generating requirements document (SRS)…")
        else:
            status.update(label=f"Generating SRS (retry {attempt - 1}/{MAX_RETRIES - 1})…")
            st.write(f"⚠️ SRS JSON malformed — retrying ({attempt - 1}/{MAX_RETRIES - 1})…")

        response = client.models.generate_content(
            model=MODEL_SRS,
            contents=user_prompt,
            config=gtypes.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=65536,
            ),
        )
        if response.text is None:
            candidate = response.candidates[0] if response.candidates else None
            reason = getattr(candidate, "finish_reason", "unknown") if candidate else "none"
            raise RuntimeError(f"Step 2 returned no text. finish_reason={reason}")

        try:
            data = _parse_json_response(response.text, debug_path)
            return attach_timestamps(data, utterances)
        except RuntimeError:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2)


def save_outputs(step1: dict, processed: dict, stem: str) -> tuple[Path, Path]:
    p1 = OUTPUT_DIR / f"{stem}_step1.json"
    p2 = OUTPUT_DIR / f"{stem}_processed.json"
    p1.write_text(json.dumps(step1,     indent=2, ensure_ascii=False), encoding="utf-8")
    p2.write_text(json.dumps(processed, indent=2, ensure_ascii=False), encoding="utf-8")
    return p1, p2


# ══════════════════════════════════════════════════════════════════════════════
# UI helpers  —  premium monochrome + #1D9E75 teal accent
# ══════════════════════════════════════════════════════════════════════════════

EMOTION_ORDER = [
    "enthusiastic", "excited", "satisfied", "calm",
    "anxious", "discouraged", "frightened", "furious",
]

EMOTION_BAR_COLOR = {
    "enthusiastic": "#1D9E75", "excited": "#1D9E75", "satisfied": "#1D9E75",
    "calm":         "#94a3b8",
    "anxious":      "#D97706", "discouraged": "#D97706",
    "frightened":   "#DC2626", "furious":     "#DC2626",
}


def req_border_color(req: dict) -> str:
    """Left-border color: teal=agreed, amber=clarification, red=disputed."""
    conflict   = req.get("conflict_level", "none")
    unresolved = req.get("unresolved", False)
    emotions   = req.get("detected_emotions", [])
    if conflict == "high" or "furious" in emotions:
        return "#DC2626"
    if conflict == "low" or unresolved or any(
        e in emotions for e in ["anxious", "discouraged", "frightened"]
    ):
        return "#D97706"
    return "#1D9E75"


def req_status_label(req: dict) -> tuple[str, str]:
    """(label_text, label_color) for inline status display."""
    border = req_border_color(req)
    if border == "#DC2626":
        return "DISPUTED", "#DC2626"
    if border == "#D97706":
        return "CLARIFICATION NEEDED", "#D97706"
    return "AGREED", "#1D9E75"


def section_header(text: str) -> None:
    st.markdown(
        f"<div style='margin:32px 0 16px 0'>"
        f"<span style='font-size:14px;font-weight:700;color:#6b7280;"
        f"text-transform:uppercase;letter-spacing:1.2px'>{text}</span>"
        f"<div style='height:2px;background:#1D9E75;width:28px;margin-top:6px'></div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def status_legend() -> None:
    st.markdown(
        "<div style='display:flex;gap:18px;margin-bottom:14px'>"
        "<span style='font-size:14px;color:#9ca3af;display:flex;align-items:center;gap:6px'>"
        "<span style='width:10px;height:10px;background:#1D9E75;border-radius:2px;"
        "display:inline-block'></span>Agreed</span>"
        "<span style='font-size:14px;color:#9ca3af;display:flex;align-items:center;gap:6px'>"
        "<span style='width:10px;height:10px;background:#D97706;border-radius:2px;"
        "display:inline-block'></span>Clarification needed</span>"
        "<span style='font-size:14px;color:#9ca3af;display:flex;align-items:center;gap:6px'>"
        "<span style='width:10px;height:10px;background:#DC2626;border-radius:2px;"
        "display:inline-block'></span>Disputed</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def speaker_emotion_bars(step1_data: dict) -> None:
    """Horizontal stacked emotion-per-speaker bar chart from step1 utterances."""
    if not step1_data:
        return
    utterances = step1_data.get("utterances", [])
    if not utterances:
        return

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for u in utterances:
        spk = u.get("speaker", "?")
        emo = u.get("emotion", "unknown")
        counts[spk][emo] += 1

    speakers = sorted(counts.keys())
    rows = ""
    for spk in speakers:
        emo_counts = counts[spk]
        total = sum(emo_counts.values()) or 1
        bars = ""
        for emo in EMOTION_ORDER:
            n = emo_counts.get(emo, 0)
            if n == 0:
                continue
            pct = n / total * 100
            color = EMOTION_BAR_COLOR.get(emo, "#94a3b8")
            bars += (
                f"<div title='{emo}: {n} ({pct:.0f}%)' style='display:inline-block;"
                f"width:{pct:.1f}%;background:{color};height:12px;vertical-align:top'></div>"
            )
        rows += (
            f"<div style='display:grid;grid-template-columns:100px 1fr;gap:10px;"
            f"align-items:center;margin-bottom:7px'>"
            f"<span style='font-size:14px;color:#6b7280;font-family:\"JetBrains Mono\","
            f"\"Fira Code\",monospace'>{spk}</span>"
            f"<div style='background:#f1f5f9;border-radius:2px;overflow:hidden;height:12px'>"
            f"{bars}</div></div>"
        )

    legend_parts = []
    for e in EMOTION_ORDER:
        color = EMOTION_BAR_COLOR.get(e, "#94a3b8")
        legend_parts.append(
            f"<span style='display:inline-flex;align-items:center;gap:4px;margin-right:10px;"
            f"font-size:13px;color:#9ca3af'>"
            f"<span style='width:8px;height:8px;background:{color};"
            f"border-radius:1px;display:inline-block;flex-shrink:0'></span>{e}</span>"
        )
    legend = "".join(legend_parts)

    st.markdown(
        f"<div style='margin-bottom:4px;font-size:13px;font-weight:700;color:#9ca3af;"
        f"text-transform:uppercase;letter-spacing:0.8px'>Emotion Distribution by Speaker</div>"
        f"<div style='margin:8px 0'>{rows}</div>"
        f"<div style='display:flex;flex-wrap:wrap;margin-top:6px'>{legend}</div>",
        unsafe_allow_html=True,
    )


def render_requirement(req: dict, id_prefix: str = "FR") -> None:
    border     = req_border_color(req)
    status_l, status_c = req_status_label(req)
    rid        = req.get("id", "?")
    title      = req.get("title", "Untitled")
    priority   = req.get("priority", "medium").upper()
    emotions   = req.get("detected_emotions", [])
    first_time = req.get("first_mentioned", "?")
    speakers   = ", ".join(req.get("stakeholders", [])) or "—"
    criteria   = req.get("acceptance_criteria", [])
    statement  = req.get("statement", "—")
    evidence   = req.get("evidence", "—")

    pri_dot = {"HIGH": "#DC2626", "MEDIUM": "#D97706", "LOW": "#1D9E75"}.get(priority, "#94a3b8")
    emo_str = " · ".join(emotions) if emotions else ""

    ac_html = ""
    if criteria:
        ac_items = "".join(
            f"<li style='margin:5px 0;color:#374151;font-size:15px;line-height:1.55'>{ac}</li>"
            for ac in criteria
        )
        ac_html = (
            f"<div style='margin-top:14px;padding-top:12px;"
            f"border-top:1px solid #f1f5f9'>"
            f"<div style='font-size:13px;font-weight:700;color:#9ca3af;"
            f"text-transform:uppercase;letter-spacing:0.6px;margin-bottom:8px'>"
            f"Acceptance Criteria</div>"
            f"<ul style='margin:0;padding-left:20px'>{ac_items}</ul>"
            f"</div>"
        )

    cat_html = ""
    if id_prefix == "NFR" and req.get("category"):
        cat_html = (
            f"<span style='font-size:13px;color:#9ca3af;margin-left:8px;"
            f"text-transform:uppercase;letter-spacing:0.4px'>{req.get('category','')}</span>"
        )

    # Background tint by status
    bg = {"#1D9E75": "#f0fdf9", "#D97706": "#fffcf0", "#DC2626": "#fff8f8"}.get(border, "#f8fafc")

    col_req, col_ev = st.columns([3, 2])

    with col_req:
        st.markdown(
            f"<div style='border-left:3px solid {border};padding:18px 22px;"
            f"background:{bg};margin-bottom:10px;border-radius:0 6px 6px 0;"
            f"box-shadow:0 1px 3px rgba(0,0,0,0.06)'>"

            # Header row
            f"<div style='display:flex;justify-content:space-between;align-items:flex-start;"
            f"margin-bottom:10px'>"
            f"<div style='flex:1;min-width:0'>"
            f"<span style='font-family:\"JetBrains Mono\",\"Fira Code\",monospace;font-size:14px;"
            f"color:#9ca3af;margin-right:10px'>{rid}</span>"
            f"<span style='font-size:17px;font-weight:600;color:#111827'>{title}</span>"
            f"{cat_html}"
            f"</div>"
            f"<div style='display:flex;gap:10px;align-items:center;flex-shrink:0;margin-left:12px'>"
            f"<span style='font-size:13px;font-weight:700;color:{status_c};"
            f"letter-spacing:0.4px'>{status_l}</span>"
            f"<span style='display:inline-flex;align-items:center;gap:4px;"
            f"font-size:13px;color:#9ca3af'>"
            f"<span style='width:8px;height:8px;background:{pri_dot};border-radius:50%;"
            f"display:inline-block'></span>{priority}</span>"
            f"</div>"
            f"</div>"

            # Statement
            f"<div style='font-size:16px;color:#1e293b;line-height:1.75'>{statement}</div>"

            # Acceptance criteria
            f"{ac_html}"

            f"</div>",
            unsafe_allow_html=True,
        )

    with col_ev:
        with st.expander("Evidence & Analysis"):
            st.markdown(
                f"<div style='font-size:15px;color:#374151;line-height:1.75;padding:6px 0'>"
                f"{evidence}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**First Mentioned**  \n`{first_time}`")
            with c2:
                st.markdown(f"**Conflict**  \n{req.get('conflict_level','none').upper()}")
            c3, c4 = st.columns(2)
            with c3:
                st.markdown(f"**Stakeholders**  \n{speakers}")
            with c4:
                ids = req.get("source_utterances", [])
                st.markdown(f"**Utterances**  \n`{ids}`" if ids else "**Utterances**  \n—")
            if emo_str:
                st.markdown(f"**Emotions**  \n{emo_str}")

    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)


def render_open_issue(oi: dict) -> None:
    level      = oi.get("conflict_level", "low")
    border     = "#DC2626" if level == "high" else "#D97706"
    status_txt = "HIGH CONFLICT" if level == "high" else "LOW CONFLICT"
    status_c   = "#DC2626" if level == "high" else "#D97706"
    desc       = oi.get("description", "")
    raised     = ", ".join(oi.get("raised_by", [])) or "—"
    related    = ", ".join(oi.get("related_requirements", [])) or "—"

    st.markdown(
        f"<div style='border-left:3px solid {border};padding:16px 22px;"
        f"background:#ffffff;margin-bottom:10px;border-radius:0 6px 6px 0;"
        f"box-shadow:0 1px 3px rgba(0,0,0,0.05)'>"
        f"<div style='display:flex;justify-content:space-between;align-items:flex-start;"
        f"margin-bottom:8px'>"
        f"<div style='flex:1;min-width:0'>"
        f"<span style='font-family:\"JetBrains Mono\",\"Fira Code\",monospace;font-size:14px;"
        f"color:#9ca3af;margin-right:10px'>{oi.get('id','?')}</span>"
        f"<span style='font-size:16px;font-weight:600;color:#111827'>{desc}</span>"
        f"</div>"
        f"<span style='font-size:13px;font-weight:700;color:{status_c};"
        f"letter-spacing:0.4px;flex-shrink:0;margin-left:12px'>{status_txt}</span>"
        f"</div>"
        f"<div style='display:flex;gap:18px;flex-wrap:wrap'>"
        f"<span style='font-size:14px;color:#9ca3af'>Raised by: "
        f"<span style='color:#6b7280'>{raised}</span></span>"
        f"<span style='font-size:14px;color:#9ca3af'>Related: "
        f"<span style='color:#6b7280'>{related}</span></span>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Suggested Resolution"):
        st.markdown(
            f"<div style='font-size:15px;color:#374151;line-height:1.75;padding:6px 0'>"
            f"{oi.get('suggested_resolution','—')}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)


def generate_docx(data: dict) -> bytes:
    """Generate an IEEE 29148-style SRS Word document from processed JSON."""
    doc = Document()
    summary = data.get("meeting_summary", {})
    reqs    = data.get("requirements", [])
    nfrs    = data.get("non_functional_requirements", [])
    issues  = data.get("open_issues", [])
    meeting_id = summary.get("meeting_id", "Meeting")

    # ── Title ──────────────────────────────────────────────────────────────
    title = doc.add_heading(f"Software Requirements Specification", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph(f"Meeting: {meeting_id}")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.runs[0].font.size = Pt(12)
    sub.runs[0].font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    doc.add_paragraph()

    # ── 1. Meeting Summary ─────────────────────────────────────────────────
    doc.add_heading("1. Meeting Overview", 1)
    t = doc.add_table(rows=1, cols=2)
    t.style = "Table Grid"
    hdr = t.rows[0].cells
    hdr[0].text = "Attribute"
    hdr[1].text = "Value"
    for k, v in [
        ("Meeting ID",        summary.get("meeting_id", "?")),
        ("Duration",          f"{summary.get('duration_seconds','?')} seconds"),
        ("Speakers",          str(summary.get("num_speakers", "?"))),
        ("Language",          summary.get("language", "?")),
        ("Overall Sentiment", summary.get("overall_sentiment", "?").capitalize()),
        ("Key Themes",        ", ".join(summary.get("key_themes", []))),
        ("Dominant Emotions", ", ".join(summary.get("dominant_emotions", []))),
    ]:
        row = t.add_row().cells
        row[0].text = k
        row[1].text = v
    doc.add_paragraph()
    doc.add_paragraph(summary.get("brief", ""))
    doc.add_paragraph()

    # ── 2. Functional Requirements ─────────────────────────────────────────
    doc.add_heading("2. Functional Requirements", 1)
    for req in reqs:
        rid      = req.get("id", "?")
        title_   = req.get("title", "Untitled")
        priority = req.get("priority", "medium").upper()

        doc.add_heading(f"{rid}  {title_}  [{priority}]", 2)

        p = doc.add_paragraph()
        p.add_run("Statement: ").bold = True
        p.add_run(req.get("statement", "—"))

        criteria = req.get("acceptance_criteria", [])
        if criteria:
            p = doc.add_paragraph()
            p.add_run("Acceptance Criteria:").bold = True
            for ac in criteria:
                doc.add_paragraph(ac, style="List Bullet")

        details = doc.add_paragraph()
        details.add_run(
            f"First mentioned: {req.get('first_mentioned','?')}  |  "
            f"Stakeholders: {', '.join(req.get('stakeholders',[]))}  |  "
            f"Conflict: {req.get('conflict_level','none')}  |  "
            f"Emotions: {', '.join(req.get('detected_emotions',[]))}"
        ).font.color.rgb = RGBColor(0x88, 0x88, 0x88)
        details.runs[0].font.size = Pt(9)

        if req.get("evidence"):
            p = doc.add_paragraph()
            p.add_run("Evidence: ").bold = True
            r = p.add_run(req["evidence"])
            r.font.size = Pt(10)
            r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

        doc.add_paragraph()

    # ── 3. Non-Functional Requirements ────────────────────────────────────
    doc.add_heading("3. Non-Functional Requirements", 1)
    for nfr in nfrs:
        rid      = nfr.get("id", "?")
        title_   = nfr.get("title", "Untitled")
        category = nfr.get("category", "?").capitalize()
        priority = nfr.get("priority", "medium").upper()

        doc.add_heading(f"{rid}  {title_}  [{category} · {priority}]", 2)

        p = doc.add_paragraph()
        p.add_run("Statement: ").bold = True
        p.add_run(nfr.get("statement", "—"))

        criteria = nfr.get("acceptance_criteria", [])
        if criteria:
            p = doc.add_paragraph()
            p.add_run("Acceptance Criteria:").bold = True
            for ac in criteria:
                doc.add_paragraph(ac, style="List Bullet")

        details = doc.add_paragraph()
        details.add_run(
            f"First mentioned: {nfr.get('first_mentioned','?')}  |  "
            f"Emotions: {', '.join(nfr.get('detected_emotions',[]))}"
        ).font.color.rgb = RGBColor(0x88, 0x88, 0x88)
        details.runs[0].font.size = Pt(9)

        if nfr.get("evidence"):
            p = doc.add_paragraph()
            p.add_run("Evidence: ").bold = True
            r = p.add_run(nfr["evidence"])
            r.font.size = Pt(10)
            r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

        doc.add_paragraph()

    # ── 4. Open Issues ─────────────────────────────────────────────────────
    doc.add_heading("4. Open Issues", 1)
    if not issues:
        doc.add_paragraph("No open issues detected.")
    for oi in issues:
        doc.add_heading(f"{oi.get('id','?')}  {oi.get('description','')}", 2)
        p = doc.add_paragraph()
        p.add_run("Raised by: ").bold = True
        p.add_run(", ".join(oi.get("raised_by", [])))

        p = doc.add_paragraph()
        p.add_run("Conflict Level: ").bold = True
        p.add_run(oi.get("conflict_level", "?").upper())

        p = doc.add_paragraph()
        p.add_run("Related Requirements: ").bold = True
        p.add_run(", ".join(oi.get("related_requirements", [])) or "—")

        p = doc.add_paragraph()
        p.add_run("Suggested Resolution: ").bold = True
        p.add_run(oi.get("suggested_resolution", "—"))
        doc.add_paragraph()

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read()


def render_report(data: dict, step1_data: dict | None = None) -> None:
    summary = data.get("meeting_summary", {})
    reqs    = data.get("requirements", [])
    nfrs    = data.get("non_functional_requirements", [])
    issues  = data.get("open_issues", [])

    # ── Page heading (full width) ─────────────────────────────────────────────
    mid = summary.get("meeting_id", "")
    st.markdown(
        f"<div style='display:flex;align-items:baseline;gap:14px;margin-bottom:24px'>"
        f"<span style='font-size:26px;font-weight:700;color:#111827'>Requirements Report</span>"
        f"<span style='font-family:\"JetBrains Mono\",\"Fira Code\",monospace;"
        f"font-size:15px;color:#9ca3af'>{mid}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Two-column layout: left=content  right=export ────────────────────────
    col_main, col_actions = st.columns([3, 2])

    # ── RIGHT: export panel (rendered first so it sticks at top) ─────────────
    with col_actions:
        st.markdown(
            "<div style='font-size:13px;font-weight:700;color:#6b7280;"
            "text-transform:uppercase;letter-spacing:1px;margin-bottom:12px'>"
            "Export</div>",
            unsafe_allow_html=True,
        )
        docx_bytes = generate_docx(data)
        st.download_button(
            label="Download SRS (.docx)",
            data=docx_bytes,
            file_name=f"{summary.get('meeting_id','report')}_SRS.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
            type="primary",
        )
        st.download_button(
            label="Download Raw (.json)",
            data=json.dumps(data, indent=2, ensure_ascii=False),
            file_name=f"{summary.get('meeting_id','report')}_processed.json",
            mime="application/json",
            use_container_width=True,
        )

        # Emotion bars — below export buttons
        if step1_data:
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            speaker_emotion_bars(step1_data)

    # ── LEFT: main content ────────────────────────────────────────────────────
    with col_main:

        # Meeting overview card — brief + stats inside
        dur_s   = summary.get("duration_seconds", 0)
        dur_str = f"{dur_s // 60}m {dur_s % 60}s" if isinstance(dur_s, int) else f"{dur_s}s"
        overall = summary.get("overall_sentiment", "—").capitalize()
        themes_str = " · ".join(summary.get("key_themes", []))

        stats_rows = [
            ("Duration",                    dur_str),
            ("Speakers",                    str(summary.get("num_speakers", "?"))),
            ("Sentiment",                   overall),
            ("Functional Requirements",     str(len(reqs))),
            ("Non-Functional Requirements", str(len(nfrs))),
            ("Open Issues",                 str(len(issues))),
        ]
        stats_html = "".join(
            f"<div style='padding:6px 0;display:flex;justify-content:space-between;"
            f"border-bottom:1px solid #edf2f7'>"
            f"<span style='font-size:14px;color:#6b7280'>{lbl}</span>"
            f"<span style='font-size:14px;font-weight:600;color:#111827;"
            f"font-variant-numeric:tabular-nums'>{val}</span>"
            f"</div>"
            for lbl, val in stats_rows
        )

        st.markdown(
            f"<div style='padding:22px 26px;background:#f8fafc;border-left:3px solid #1D9E75;"
            f"border-radius:0 8px 8px 0;margin-bottom:24px'>"
            f"<div style='font-size:13px;font-weight:700;color:#9ca3af;text-transform:uppercase;"
            f"letter-spacing:1px;margin-bottom:12px'>Meeting Overview</div>"
            f"<div style='font-size:20px;font-weight:600;color:#111827;line-height:1.6;"
            f"margin-bottom:14px'>{summary.get('brief','')}</div>"
            + (f"<div style='font-size:15px;color:#6b7280;margin-bottom:16px'>{themes_str}</div>"
               if themes_str else "")
            + f"<div style='border-top:1px solid #e2e8f0;padding-top:14px;margin-top:4px'>"
            f"{stats_html}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.divider()

        # Functional Requirements
        section_header(f"Functional Requirements  ·  {len(reqs)}")
        status_legend()
        if reqs:
            for req in reqs:
                render_requirement(req, "FR")
        else:
            st.markdown(
                "<div style='color:#9ca3af;font-size:15px;padding:8px 0'>"
                "No functional requirements extracted.</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # Non-Functional Requirements
        section_header(f"Non-Functional Requirements  ·  {len(nfrs)}")
        status_legend()
        if nfrs:
            for nfr in nfrs:
                render_requirement(nfr, "NFR")
        else:
            st.markdown(
                "<div style='color:#9ca3af;font-size:15px;padding:8px 0'>"
                "No non-functional requirements extracted.</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # Open Issues
        section_header(f"Open Issues  ·  {len(issues)}")
        if issues:
            for oi in issues:
                render_open_issue(oi)
        else:
            st.markdown(
                "<div style='color:#9ca3af;font-size:15px;padding:8px 0'>"
                "No open issues detected.</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# Page config & global CSS
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="RE Meeting Analyzer",
    page_icon="📋",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", sans-serif;
    background: #f9fafb;
}

/* ── Main area background ── */
[data-testid="stAppViewContainer"] > .main {
    background: #f9fafb;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111111 !important;
    border-right: 1px solid #1e1e1e !important;
}
[data-testid="stSidebar"],
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label {
    color: #a1a1aa !important;
}

/* ── Sidebar selectbox & text input ── */
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: #1c1c1c !important;
    color: #e4e4e7 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 5px !important;
    font-size: 13px !important;
}
[data-testid="stSidebar"] .stCheckbox label { color: #71717a !important; }

/* ── Sidebar nav buttons ── */
[data-testid="stSidebar"] .stButton > button {
    background: #1c1c1c !important;
    border: 1px solid #2a2a2a !important;
    color: #a1a1aa !important;
    text-align: left !important;
    padding: 10px 14px !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    width: 100% !important;
    margin-bottom: 3px !important;
    transition: all 0.12s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #1D9E75 !important;
    border-color: #1D9E75 !important;
    color: #ffffff !important;
}

/* ── Primary run button in sidebar ── */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #1D9E75 !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    margin-top: 6px !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: #178a65 !important;
}

/* ── Sidebar divider ── */
[data-testid="stSidebar"] hr { border-color: #1e1e1e !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid #e5e7eb !important;
    border-radius: 6px !important;
    background: #ffffff !important;
}
[data-testid="stExpander"] summary {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #6b7280 !important;
    padding: 9px 14px !important;
}

/* ── Divider ── */
hr { border-color: #f1f5f9 !important; }

/* ── Download button (primary) ── */
.stDownloadButton > button[kind="primary"] {
    background: #1D9E75 !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}
.stDownloadButton > button[kind="primary"]:hover {
    background: #178a65 !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] { border-radius: 8px; }

/* ── Report list-item buttons ── */
[data-testid="stSidebar"] .stButton > button {
    text-align: left !important;
    white-space: pre-wrap !important;
    line-height: 1.4 !important;
}

/* ── File uploader in sidebar — dark theme ── */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] > div {
    background: #1c1c1c !important;
    border: 1px dashed #3a3a3a !important;
    border-radius: 6px !important;
    padding: 12px 10px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
    background: #1c1c1c !important;
    border: none !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] p,
[data-testid="stSidebar"] [data-testid="stFileUploader"] span,
[data-testid="stSidebar"] [data-testid="stFileUploader"] small {
    color: #71717a !important;
    font-size: 12px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: #2a2a2a !important;
    border: 1px solid #3a3a3a !important;
    color: #a1a1aa !important;
    border-radius: 5px !important;
    font-size: 12px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
    background: #1D9E75 !important;
    border-color: #1D9E75 !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("page",              "report"),
    ("report_data",       None),
    ("step1_data",        None),
    ("report_meeting_id", None),
    ("qa_selected_id",    None),
    ("qa_report_data",    None),
    ("qa_selected_ids",   set()),
    ("rag_index",         None),   # RequirementIndex instance
    ("rag_index_key",     None),   # frozenset of meeting IDs currently indexed
    ("chat_history",      []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # Brand
    st.markdown(
        "<div style='padding:22px 16px 10px'>"
        "<div style='font-size:16px;font-weight:700;color:#e4e4e7;letter-spacing:-0.3px'>"
        "RE Analyzer</div>"
        "<div style='font-size:11px;color:#52525b;margin-top:2px'>ASE 2026 Demo Track</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='margin:6px 0 10px'>", unsafe_allow_html=True)

    # Navigation
    st.markdown(
        "<div style='font-size:10px;color:#52525b;padding:0 14px 6px;"
        "text-transform:uppercase;letter-spacing:1px'>Navigation</div>",
        unsafe_allow_html=True,
    )
    if st.button("Requirements Report", key="nav_report", use_container_width=True):
        st.session_state.page = "report"
    if st.button("Ask Requirements", key="nav_qa", use_container_width=True):
        st.session_state.page = "qa"

    st.markdown("<hr style='margin:10px 0'>", unsafe_allow_html=True)

    # ── Audio controls (report page only) ─────────────────────────────────
    if st.session_state.page == "report":
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)

        # ── Processed Reports — single select ─────────────────────────────
        processed_files = sorted(
            OUTPUT_DIR.glob("*_processed.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if processed_files:
            st.markdown(
                "<div style='font-size:10px;color:#52525b;padding:0 0 8px;"
                "text-transform:uppercase;letter-spacing:1px'>Processed Reports</div>",
                unsafe_allow_html=True,
            )
            for i, f in enumerate(processed_files, 1):
                mid = f.stem.replace("_processed", "")
                mtime_str = datetime.datetime.fromtimestamp(
                    f.stat().st_mtime
                ).strftime("%b %d")

                label = f"{i}.  {mid}  ·  {mtime_str}"
                if st.button(label, key=f"load_rpt_{mid}", use_container_width=True):
                    st.session_state.report_data       = json.loads(f.read_text(encoding="utf-8"))
                    st.session_state.report_meeting_id = mid
                    step1_f = OUTPUT_DIR / f"{mid}_step1.json"
                    st.session_state.step1_data = (
                        json.loads(step1_f.read_text(encoding="utf-8"))
                        if step1_f.exists() else None
                    )
                    st.rerun()

            st.markdown("<hr style='margin:10px 0'>", unsafe_allow_html=True)

        # ── Upload a new file ──────────────────────────────────────────────
        st.markdown(
            "<div style='font-size:10px;color:#52525b;padding:0 0 6px;"
            "text-transform:uppercase;letter-spacing:1px'>Upload Audio</div>",
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "upload",
            type=list({ext.lstrip(".") for ext in MIME_MAP}),
            label_visibility="collapsed",
        )
        if uploaded is not None:
            save_path = AUDIO_DIR / uploaded.name
            if not save_path.exists():
                save_path.write_bytes(uploaded.read())
                st.success(f"Saved → {uploaded.name}")
            else:
                st.caption(f"{uploaded.name} already exists.")

        st.markdown("<hr style='margin:10px 0'>", unsafe_allow_html=True)

        # ── Select from existing files ─────────────────────────────────────
        st.markdown(
            "<div style='font-size:10px;color:#52525b;padding:0 0 6px;"
            "text-transform:uppercase;letter-spacing:1px'>Select Audio</div>",
            unsafe_allow_html=True,
        )

        audio_files = sorted(
            f for f in AUDIO_DIR.iterdir() if f.suffix.lower() in MIME_MAP
        ) if AUDIO_DIR.exists() else []

        if not audio_files:
            st.warning(f"No audio files in {AUDIO_DIR}/")
            st.stop()

        file_names = [f.name for f in audio_files]
        file_map   = {f.name: f for f in audio_files}

        # Default to uploaded file if just saved
        default_idx = 0
        if uploaded is not None and uploaded.name in file_names:
            default_idx = file_names.index(uploaded.name)

        selected_name = st.selectbox(
            "file", options=file_names, index=default_idx,
            label_visibility="collapsed",
        )
        selected_file = file_map[selected_name]

        # File metadata row
        size_mb   = selected_file.stat().st_size / 1_048_576
        mtime     = datetime.datetime.fromtimestamp(selected_file.stat().st_mtime)
        mtime_str = mtime.strftime("%b %d, %H:%M")
        st.markdown(
            f"<div style='font-size:11px;color:#52525b;padding:2px 2px 10px;"
            f"display:flex;justify-content:space-between'>"
            f"<span>{size_mb:.1f} MB</span><span>{mtime_str}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        meeting_id = st.text_input(
            "Meeting ID", value=selected_file.stem,
            label_visibility="collapsed", placeholder="Meeting ID"
        )
        use_cache = st.checkbox("Use cached results", value=True)
        run_btn   = st.button("Run Analysis", type="primary", use_container_width=True)

    else:
        run_btn       = False
        use_cache     = True
        selected_file = None
        meeting_id    = st.session_state.get("report_meeting_id", "")

        # ── List all processed meetings ────────────────────────────────────
        st.markdown(
            "<div style='font-size:10px;color:#52525b;padding:0 0 8px;"
            "text-transform:uppercase;letter-spacing:1px'>Processed Meetings</div>",
            unsafe_allow_html=True,
        )

        processed_files = sorted(
            OUTPUT_DIR.glob("*_processed.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )

        if not processed_files:
            st.markdown(
                "<div style='color:#52525b;font-size:12px;padding:4px 0'>"
                "No processed meetings yet.<br>Run an analysis first.</div>",
                unsafe_allow_html=True,
            )
        else:
            # Init selected set
            if "qa_selected_ids" not in st.session_state:
                st.session_state.qa_selected_ids = set()

            # Render each meeting as a checkbox row
            for f in processed_files:
                mid = f.stem.replace("_processed", "")
                mtime_str = datetime.datetime.fromtimestamp(
                    f.stat().st_mtime
                ).strftime("%b %d")

                try:
                    d = json.loads(f.read_text(encoding="utf-8"))
                    counts = (f"{len(d.get('requirements',[]))} FR · "
                              f"{len(d.get('non_functional_requirements',[]))} NFR")
                except Exception:
                    counts = "—"

                checked = st.checkbox(
                    mid,
                    value=(mid in st.session_state.qa_selected_ids),
                    key=f"qa_chk_{mid}",
                )
                if checked:
                    st.session_state.qa_selected_ids.add(mid)
                else:
                    st.session_state.qa_selected_ids.discard(mid)

                st.markdown(
                    f"<div style='font-size:11px;color:#52525b;margin:-8px 0 8px 26px;"
                    f"display:flex;justify-content:space-between'>"
                    f"<span>{counts}</span><span>{mtime_str}</span></div>",
                    unsafe_allow_html=True,
                )

            # Summary badge
            n = len(st.session_state.qa_selected_ids)
            badge_c = "#1D9E75" if n > 0 else "#3a3a3a"
            badge_t = "#ffffff" if n > 0 else "#52525b"
            st.markdown(
                f"<div style='margin-top:10px;padding:8px 12px;background:{badge_c}20;"
                f"border:1px solid {badge_c};border-radius:5px;font-size:12px;"
                f"color:{badge_t};text-align:center'>"
                f"{'<b>' + str(n) + '</b> meeting' + ('s' if n != 1 else '') + ' queued for RAG index' if n > 0 else 'No meetings selected'}"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Store selected data for QA page
            st.session_state.qa_report_data = {
                mid: json.loads(
                    (OUTPUT_DIR / f"{mid}_processed.json").read_text(encoding="utf-8")
                )
                for mid in st.session_state.qa_selected_ids
                if (OUTPUT_DIR / f"{mid}_processed.json").exists()
            }

# ══════════════════════════════════════════════════════════════════════════════
# Page: Requirements Report
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.page == "report":

    # Load cached report
    if meeting_id:
        cached_processed = OUTPUT_DIR / f"{meeting_id}_processed.json"
        cached_step1     = OUTPUT_DIR / f"{meeting_id}_step1.json"
        if (
            st.session_state.report_data is None
            and cached_processed.exists()
            and use_cache
        ):
            st.session_state.report_data = json.loads(
                cached_processed.read_text(encoding="utf-8")
            )
            st.session_state.report_meeting_id = meeting_id
            # Load step1 data for emotion bars if available
            if cached_step1.exists() and st.session_state.step1_data is None:
                st.session_state.step1_data = json.loads(
                    cached_step1.read_text(encoding="utf-8")
                )

    # Run pipeline
    if run_btn:
        cached_step1     = OUTPUT_DIR / f"{meeting_id}_step1.json"
        cached_processed = OUTPUT_DIR / f"{meeting_id}_processed.json"
        try:
            with st.status("Running analysis pipeline…", expanded=True) as status:
                if use_cache and cached_step1.exists():
                    status.update(label="Step 1: Loading cached audio analysis…")
                    step1_data = json.loads(cached_step1.read_text(encoding="utf-8"))
                    st.write(f"✅ Step 1 from cache — {len(step1_data.get('utterances',[]))} utterances")
                else:
                    st.write("Step 1: Audio analysis starting…")
                    step1_data = run_step1(selected_file, status)
                    st.write(f"✅ Step 1 complete — {len(step1_data.get('utterances',[]))} utterances")

                if use_cache and cached_processed.exists():
                    status.update(label="Step 2: Loading cached SRS…")
                    processed = json.loads(cached_processed.read_text(encoding="utf-8"))
                    st.write("✅ Step 2 from cache")
                else:
                    st.write("Step 2: Generating requirements document…")
                    processed = run_step2(step1_data, meeting_id, status)
                    st.write(
                        f"✅ Step 2 complete — "
                        f"{len(processed.get('requirements',[]))} FR · "
                        f"{len(processed.get('non_functional_requirements',[]))} NFR · "
                        f"{len(processed.get('open_issues',[]))} issues"
                    )

                save_outputs(step1_data, processed, meeting_id)
                status.update(label="✅ Analysis complete!", state="complete")

            st.session_state.report_data       = processed
            st.session_state.step1_data        = step1_data
            st.session_state.report_meeting_id = meeting_id

        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()

    # Render
    if st.session_state.report_data:
        render_report(st.session_state.report_data, st.session_state.step1_data)
    else:
        st.markdown(
            "<div style='text-align:center;padding:100px 40px'>"
            "<div style='font-size:36px;margin-bottom:16px;color:#d1d5db'>📋</div>"
            "<div style='font-size:18px;font-weight:600;color:#374151;margin-bottom:8px'>"
            "No report yet</div>"
            "<div style='font-size:14px;color:#9ca3af'>Select an audio file and click "
            "<b>Run Analysis</b> to begin.</div>"
            "</div>",
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# Page: Ask Requirements (RAG placeholder)
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "qa":

    qa_data = st.session_state.qa_report_data or {}

    if not qa_data:
        st.markdown(
            "<div style='text-align:center;padding:80px 40px'>"
            "<div style='font-size:36px;margin-bottom:16px;color:#d1d5db'>💬</div>"
            "<div style='font-size:18px;font-weight:600;color:#374151;margin-bottom:8px'>"
            "No meetings selected</div>"
            "<div style='font-size:14px;color:#9ca3af'>Check one or more meetings in the "
            "sidebar to add them to the RAG index.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    meeting_ids = sorted(qa_data.keys())
    total_fr    = sum(len(d.get("requirements", []))               for d in qa_data.values())
    total_nfr   = sum(len(d.get("non_functional_requirements", [])) for d in qa_data.values())
    current_key = frozenset(meeting_ids)

    # ── Build / rebuild index when selection changes ──────────────────────────
    if st.session_state.rag_index_key != current_key:
        st.session_state.rag_index     = RequirementIndex()
        st.session_state.rag_index_key = None
        st.session_state.chat_history  = []

        with st.status(
            f"Building RAG index for {len(qa_data)} meeting(s)…", expanded=True
        ) as idx_status:
            bar = st.progress(0.0)
            st.session_state.rag_index.build(
                qa_data, client,
                progress_cb=lambda v: bar.progress(v),
            )
            n = st.session_state.rag_index.size
            st.session_state.rag_index_key = current_key
            idx_status.update(
                label=f"✅ Index ready — {n} chunks from {len(qa_data)} meeting(s)",
                state="complete",
            )

    index = st.session_state.rag_index

    # ── Terminal-style header ─────────────────────────────────────────────────
    meetings_line = "  ·  ".join(meeting_ids)
    indexed = index.size if index else 0
    st.markdown(
        f"<div style='background:#111111;border-radius:8px;padding:16px 20px;"
        f"margin-bottom:20px;font-family:\"JetBrains Mono\",\"Fira Code\",monospace'>"
        f"<div style='font-size:11px;color:#52525b;margin-bottom:8px'>"
        f"requirements-qa  ·  {len(qa_data)} meeting{'s' if len(qa_data)!=1 else ''}</div>"
        f"<div style='font-size:13px;color:#a1a1aa;margin-bottom:6px'>"
        f"<span style='color:#1D9E75'>●</span>&nbsp; {indexed} chunks indexed — "
        f"{total_fr} FR · {total_nfr} NFR</div>"
        f"<div style='font-size:12px;color:#52525b'>{meetings_line}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Conversation history ──────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(
            msg["role"],
            avatar="🧑" if msg["role"] == "user" else "🤖",
        ):
            st.markdown(msg["content"])

    # ── Suggested questions (only when empty) ─────────────────────────────────
    if not st.session_state.chat_history:
        st.markdown(
            "<div style='font-size:13px;color:#9ca3af;margin:8px 0 16px 0'>"
            "Try asking: &nbsp;"
            "<i>What was this meeting mainly about?</i> &nbsp;·&nbsp; "
            "<i>Which requirements are disputed?</i> &nbsp;·&nbsp; "
            "<i>List all high-priority requirements</i> &nbsp;·&nbsp; "
            "<i>When was [feature] discussed?</i>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Chat input ────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask about the requirements…"):
        # Show user message immediately
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Stream assistant response
        with st.chat_message("assistant", avatar="🤖"):
            reply = st.write_stream(
                rag_stream(
                    prompt,
                    index,
                    st.session_state.chat_history[:-1],
                    client,
                )
            )

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
