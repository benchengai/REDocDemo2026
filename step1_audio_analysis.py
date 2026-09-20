"""
step1_audio_analysis.py
RE Meeting Analysis — Step 1
Audio → Gemini 2.5 Pro → emotion / confidence / transcript / speaker / time
Output: data/outputs/<stem>_step1.json
"""

import json
import os
import sys
import time
from pathlib import Path

import google.genai as genai
import google.genai.types as gtypes
from dotenv import load_dotenv

# ── env ──────────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    sys.exit("ERROR: GEMINI_API_KEY not set in .env")

client = genai.Client(api_key=API_KEY)

MODEL = "gemini-2.5-pro"
AUDIO_DIR = Path("data/audio")
OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIME_MAP = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
}

PROMPT = """\
You are an expert meeting analyst. Listen to this meeting audio carefully.

Return a JSON object with exactly this structure — no markdown fences, no extra text:

{
  "meeting_meta": {
    "duration_seconds": <int>,
    "num_speakers": <int>,
    "language": "<language>"
  },
  "utterances": [
    {
      "id": <int, starting from 1>,
      "speaker": "<Speaker_A | Speaker_B | ...>",
      "start_time": "<MM:SS>",
      "end_time": "<MM:SS>",
      "text": "<verbatim transcript of this speaking turn>",
      "emotion": "<anxious | calm | discouraged | enthusiastic | excited | frightened | furious | satisfied>",
      "confidence": <float 0.0-1.0, how confident you are in this emotion label based on vocal cues>
    }
  ]
}

Rules:
- Speaker labels must be consistent across the whole recording (same voice = same label).
- Emotion must reflect VOCAL/ACOUSTIC signals AND semantic content together, but prioritize acoustic cues when they conflict with word meaning.
- confidence reflects acoustic clarity of the emotion, not semantic certainty.
- start_time and end_time are wall-clock offsets from the start of this audio file in MM:SS format.
- Include every speaking turn, even very short ones.
- Return ONLY the raw JSON object, nothing else.
"""


def pick_audio_file() -> Path:
    """List audio files in AUDIO_DIR and let the user choose one."""
    files = sorted(
        f for f in AUDIO_DIR.iterdir()
        if f.suffix.lower() in MIME_MAP
    )
    if not files:
        sys.exit(f"No audio files found in {AUDIO_DIR}/")

    print("\nAvailable audio files in data/audio/:")
    for i, f in enumerate(files, 1):
        size_mb = f.stat().st_size / 1_048_576
        print(f"  [{i}] {f.name}  ({size_mb:.1f} MB)")

    print()
    while True:
        choice = input("Select a file (number or filename): ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return files[idx]
        else:
            matches = [f for f in files if f.name == choice or f.stem == choice]
            if matches:
                return matches[0]
        print(f"  Please enter a number 1–{len(files)} or a valid filename.")


def upload_audio(path: Path) -> gtypes.File:
    """Upload audio via Files API and wait until ACTIVE."""
    mime = MIME_MAP[path.suffix.lower()]
    size_mb = path.stat().st_size / 1_048_576
    print(f"\nUploading {path.name} ({size_mb:.1f} MB) to Gemini Files API ...")

    with open(path, "rb") as f:
        uploaded = client.files.upload(
            file=f,
            config=gtypes.UploadFileConfig(
                mime_type=mime,
                display_name=path.name,
            ),
        )

    print(f"  File name : {uploaded.name}")
    print(f"  File URI  : {uploaded.uri}")

    # Poll until processing finishes
    dots = 0
    while uploaded.state.name == "PROCESSING":
        dots = (dots + 1) % 4
        print(f"  Processing{'.' * dots}   ", end="\r")
        time.sleep(5)
        uploaded = client.files.get(name=uploaded.name)

    if uploaded.state.name != "ACTIVE":
        sys.exit(f"\nFile processing failed with state: {uploaded.state.name}")

    print(f"\n  Ready: {uploaded.state.name}")
    return uploaded


def analyse(audio_file: gtypes.File) -> tuple[dict, dict]:
    """Call Gemini with the audio file and parse the JSON response.
    Returns (data, token_usage).
    """
    print(f"\nAnalysing with {MODEL} ...")

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            gtypes.Part.from_uri(file_uri=audio_file.uri, mime_type=audio_file.mime_type),
            PROMPT,
        ],
        config=gtypes.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=65536,
        ),
    )

    # Extract token usage
    usage = response.usage_metadata
    token_info = {
        "input_tokens": getattr(usage, "prompt_token_count", 0),
        "output_tokens": getattr(usage, "candidates_token_count", 0),
        "total_tokens": getattr(usage, "total_token_count", 0),
    }

    # Diagnose empty response
    if response.text is None:
        candidate = response.candidates[0] if response.candidates else None
        finish_reason = getattr(candidate, "finish_reason", "unknown") if candidate else "no candidates"
        safety = getattr(candidate, "safety_ratings", []) if candidate else []
        print(f"\n  finish_reason : {finish_reason}")
        print(f"  safety_ratings: {safety}")
        sys.exit("ERROR: Gemini returned no text. See finish_reason above.")

    raw = response.text.strip()

    # Strip accidental markdown fences if model ignores response_mime_type
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw), token_info


def save_result(data: dict, audio_path: Path) -> Path:
    out_path = OUTPUT_DIR / f"{audio_path.stem}_step1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return out_path


def to_seconds(t: str) -> int:
    """Convert MM:SS string to total seconds."""
    try:
        parts = t.split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return 0


def print_summary(data: dict) -> None:
    meta = data.get("meeting_meta", {})
    utterances = data.get("utterances", [])

    print(f"\n{'='*58}")
    print(f"  Duration   : {meta.get('duration_seconds', '?')} s")
    print(f"  Speakers   : {meta.get('num_speakers', '?')}")
    print(f"  Language   : {meta.get('language', '?')}")
    print(f"  Utterances : {len(utterances)}")

    # Emotion distribution
    emotion_counts: dict[str, int] = {}
    for u in utterances:
        e = u.get("emotion", "unknown")
        emotion_counts[e] = emotion_counts.get(e, 0) + 1

    print(f"\n  Emotion distribution:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"    {emotion:<15} {count:>3}  {bar}")

    # Per-speaker talking time
    speaker_time: dict[str, int] = {}
    for u in utterances:
        spk = u.get("speaker", "?")
        dur = to_seconds(u.get("end_time", "0:00")) - to_seconds(u.get("start_time", "0:00"))
        speaker_time[spk] = speaker_time.get(spk, 0) + max(dur, 0)

    print(f"\n  Speaker talking time:")
    total = sum(speaker_time.values()) or 1
    for spk, secs in sorted(speaker_time.items()):
        pct = secs / total * 100
        print(f"    {spk:<12} {secs:>4} s  ({pct:.0f}%)")

    # Average confidence per emotion
    emo_conf: dict[str, list[float]] = {}
    for u in utterances:
        e = u.get("emotion", "unknown")
        c = u.get("confidence", None)
        if c is not None:
            emo_conf.setdefault(e, []).append(float(c))

    print(f"\n  Avg confidence per emotion:")
    for e, vals in sorted(emo_conf.items()):
        avg = sum(vals) / len(vals)
        print(f"    {e:<15} {avg:.2f}")

    print(f"{'='*58}")


def print_token_usage(token_info: dict) -> None:
    inp = token_info.get("input_tokens", 0)
    out = token_info.get("output_tokens", 0)
    tot = token_info.get("total_tokens", 0)
    print(f"\n  Token usage:")
    print(f"    Input  (audio + prompt) : {inp:>8,}")
    print(f"    Output (JSON)           : {out:>8,}  / 65,536 max")
    print(f"    Total                   : {tot:>8,}")
    if out >= 60000:
        print("  WARNING: output is near the 65,536 limit — response may be truncated.")


def main() -> None:
    audio_path = pick_audio_file()
    print(f"\nSelected: {audio_path}")

    audio_file = upload_audio(audio_path)

    try:
        data, token_info = analyse(audio_file)
    finally:
        # Always clean up the remote file to save storage quota
        try:
            client.files.delete(name=audio_file.name)
            print("Remote file deleted from Gemini Files API.")
        except Exception:
            pass

    out_path = save_result(data, audio_path)
    print_summary(data)
    print_token_usage(token_info)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
