"""
generate_srs.py
RE Meeting Analysis — Step 2
JSON (Step 1 output) → Gemini 2.5 Pro → Structured SRS document (JSON)
Output: data/outputs/{meeting_id}_processed.json

Usage:
    python generate_srs.py                          # interactive picker
    python generate_srs.py --file data/outputs/2014a_step1.json --id ES2014a
"""

import argparse
import json
import os
import sys
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

MODEL       = "gemini-2.5-pro"
OUTPUT_DIR  = Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior Requirements Engineer following IEEE 29148 standard.
You are analyzing a stakeholder meeting to extract structured requirements.

The meeting may discuss software systems, hardware products, or services.
Treat all discussable features, constraints, and behaviors as requirements
regardless of the domain.

QUALITY RULES (IEEE 29148):
Every requirement MUST be:
1. Necessary: Removing it would affect the end product
2. Unambiguous: Only one possible interpretation
3. Verifiable: Can be tested or measured objectively
4. Traceable: Linked to specific stakeholder utterances
5. Use SHALL for mandatory, SHOULD for recommended

WRITING RULES:
- Never use vague terms: "fast", "easy", "user-friendly", "etc"
- Never combine two requirements with "and" - split them
- Always describe WHAT the system shall do/be, never HOW
- Always include acceptance criteria
- Short responses like "okay", "yeah", "right", "mhm" are NOT ignored.
  Use surrounding context to determine if they indicate agreement or disagreement.
  Never extract short responses as standalone requirements.
  DO use them as agreement/disagreement signals when attributing stakeholders.

EMOTION CONFIDENCE:
Each utterance has a confidence score for emotion detection:
- confidence >= 0.8: treat emotion as reliable signal
- confidence 0.6-0.8: treat as weak signal, note uncertainty
- confidence < 0.6: ignore emotion label, treat as neutral

THE 8 EMOTION LABELS AND THEIR RE SIGNALS:
Utterances are tagged with exactly one of these 8 emotions:
  enthusiastic → strong stakeholder buy-in, raise requirement priority
  excited      → positive engagement, likely high-value feature
  satisfied    → consensus reached, requirement is stable
  calm         → neutral factual discussion, no strong stance
  anxious      → uncertainty or risk, mark as unresolved if unaddressed
  discouraged  → stakeholder concern or resistance, note as negative signal
  frightened   → strong risk or pressure, flag as open issue
  furious      → active conflict, conflict_level=high, must appear in open_issues

MAPPING EMOTIONS TO REQUIREMENT FIELDS:
- detected_emotions: list the actual emotion labels from source utterances
- sentiment field uses derived signal:
    enthusiastic | excited | satisfied  → "positive"
    calm                                → "neutral"
    anxious | discouraged | frightened | furious → "negative"
- unresolved=true if any source utterance is anxious/frightened/furious
  AND no utterance in the same topic thread is satisfied/enthusiastic
- conflict_level=high if furious appears among speakers on the same topic
- conflict_level=low if discouraged or anxious with no resolution signal

EVIDENCE QUALITY:
evidence field MUST include:
1. Who said what with paraphrase, never exact quote
2. Their emotional state and confidence level
3. Any agreement or disagreement from other speakers
4. Why this utterance supports this requirement

BAD evidence: "Speaker_A mentioned the logo at 12:08"
GOOD evidence: "Speaker_A at 12:08 raised a design constraint (calm, 0.8): product must \
incorporate corporate colors per management instructions. Speaker_B and Speaker_C showed no objection."
"""


def build_user_prompt(meeting_id: str, meta: dict, utterances: list[dict]) -> str:
    """Build the user prompt by injecting the meeting data into the template."""

    # Format utterances as a compact numbered list to save tokens
    utt_lines = []
    for u in utterances:
        emotion_str = f"{u.get('emotion', 'unknown')} (conf={u.get('confidence', '?')})"
        utt_lines.append(
            f"[{u['id']}] {u['speaker']} {u['start_time']}-{u['end_time']} "
            f"| {emotion_str}\n    \"{u['text']}\""
        )
    utterances_block = "\n".join(utt_lines)

    return f"""\
MEETING ID: {meeting_id}
DURATION: {meta.get('duration_seconds', '?')} seconds
SPEAKERS: {meta.get('num_speakers', '?')}
LANGUAGE: {meta.get('language', '?')}

UTTERANCES:
{utterances_block}

─────────────────────────────────────────────────────
TASK: Analyze all utterances above and produce a structured requirements document.

Return a single JSON object with exactly this structure — no markdown, no extra text:

{{
  "meeting_summary": {{
    "meeting_id": "{meeting_id}",
    "duration_seconds": <int>,
    "num_speakers": <int>,
    "language": "<str>",
    "overall_sentiment": "<positive|negative|neutral|mixed>",
    "key_themes": ["<theme>"],
    "decision_count": <int>,
    "unresolved_count": <int>,
    "dominant_emotions": ["<use only the 8 labels: anxious|calm|discouraged|enthusiastic|excited|frightened|furious|satisfied>"],
    "brief": "<2-3 sentence overview of what the meeting was about>"
  }},
  "requirements": [
    {{
      "id": "FR-001",
      "title": "<short noun phrase>",
      "statement": "The system SHALL <what, not how>.",
      "priority": "<high|medium|low>",
      "sentiment": "<positive|negative|neutral>",
      "detected_emotions": ["<one or more of the 8 emotion labels from source utterances>"],
      "unresolved": <true|false>,
      "conflict_level": "<none|low|high>",
      "acceptance_criteria": ["<measurable criterion>"],
      "stakeholders": ["Speaker_X"],
      "evidence": "<rich evidence following EVIDENCE QUALITY rules above>",
      "source_utterances": [<utterance id ints>]
    }}
  ],
  "non_functional_requirements": [
    {{
      "id": "NFR-001",
      "category": "<performance|usability|reliability|security|maintainability|portability|cost|safety>",
      "title": "<short noun phrase>",
      "statement": "The system SHALL <what, not how>.",
      "priority": "<high|medium|low>",
      "detected_emotions": ["<one or more of the 8 emotion labels from source utterances>"],
      "acceptance_criteria": ["<measurable criterion>"],
      "evidence": "<rich evidence following EVIDENCE QUALITY rules above>",
      "source_utterances": [<utterance id ints>]
    }}
  ],
  "open_issues": [
    {{
      "id": "OI-001",
      "description": "<what is unresolved or conflicted>",
      "raised_by": ["Speaker_X"],
      "related_requirements": ["FR-001"],
      "conflict_level": "<low|high>",
      "suggested_resolution": "<actionable next step>"
    }}
  ]
}}

RULES:
- Extract ALL functional requirements discussed, even briefly.
- Split compound requirements (do not use "and" in one statement).
- Every FR and NFR must have at least one acceptance_criteria entry.
- open_issues must capture topics where no clear decision was reached
  OR where speakers disagreed (conflict_level >= low).
- Return ONLY the raw JSON object.
"""


# ── File picker ───────────────────────────────────────────────────────────────

def pick_json_file() -> tuple[Path, str]:
    """List *_step1.json files and let the user choose one."""
    files = sorted(OUTPUT_DIR.glob("*_step1.json"))
    if not files:
        sys.exit(f"No *_step1.json files found in {OUTPUT_DIR}/")

    print("\nAvailable Step 1 JSON files:")
    for i, f in enumerate(files, 1):
        size_kb = f.stat().st_size / 1024
        print(f"  [{i}] {f.name}  ({size_kb:.0f} KB)")

    print()
    while True:
        choice = input("Select a file (number or filename): ").strip()
        selected = None
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                selected = files[idx]
        else:
            matches = [f for f in files if f.name == choice or f.stem == choice]
            if matches:
                selected = matches[0]
        if selected:
            default_id = selected.stem.replace("_step1", "")
            meeting_id = input(f"Meeting ID [{default_id}]: ").strip() or default_id
            return selected, meeting_id
        print(f"  Please enter a number 1–{len(files)} or a valid filename.")


# ── Gemini call ───────────────────────────────────────────────────────────────

def generate_srs(user_prompt: str) -> tuple[dict, dict]:
    """Call Gemini and return (parsed_json, token_usage)."""
    print(f"\nGenerating SRS with {MODEL} ...")

    response = client.models.generate_content(
        model=MODEL,
        contents=user_prompt,
        config=gtypes.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=65536,
        ),
    )

    # Diagnose empty response
    if response.text is None:
        candidate = response.candidates[0] if response.candidates else None
        finish_reason = getattr(candidate, "finish_reason", "unknown") if candidate else "no candidates"
        sys.exit(f"ERROR: Gemini returned no text. finish_reason={finish_reason}")

    usage = response.usage_metadata
    token_info = {
        "input_tokens":  getattr(usage, "prompt_token_count",     0),
        "output_tokens": getattr(usage, "candidates_token_count", 0),
        "total_tokens":  getattr(usage, "total_token_count",      0),
    }

    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw), token_info
    except json.JSONDecodeError as e:
        # Save raw response for debugging
        debug_path = OUTPUT_DIR / "_srs_raw_response.txt"
        debug_path.write_text(raw, encoding="utf-8")
        sys.exit(
            f"ERROR: Failed to parse Gemini response as JSON.\n"
            f"  Reason : {e}\n"
            f"  Raw output saved to {debug_path} for inspection."
        )


# ── Output ────────────────────────────────────────────────────────────────────

def attach_timestamps(data: dict, utterances: list[dict]) -> dict:
    """Post-process: attach the first mention time for each requirement."""
    utt_map = {u["id"]: u for u in utterances}

    def first_mention(ids: list) -> str:
        # Normalise to int (Gemini sometimes returns strings)
        int_ids = sorted({int(i) for i in ids if str(i).isdigit()})
        matched = [utt_map[i] for i in int_ids if i in utt_map]
        return matched[0]["start_time"] if matched else "?"

    for req in data.get("requirements", []):
        req["first_mentioned"] = first_mention(req.get("source_utterances", []))

    for nfr in data.get("non_functional_requirements", []):
        nfr["first_mentioned"] = first_mention(nfr.get("source_utterances", []))

    return data


def save_result(data: dict, meeting_id: str) -> Path:
    out_path = OUTPUT_DIR / f"{meeting_id}_processed.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return out_path


def print_summary(data: dict, token_info: dict) -> None:
    summary = data.get("meeting_summary", {})
    reqs    = data.get("requirements", [])
    nfrs    = data.get("non_functional_requirements", [])
    issues  = data.get("open_issues", [])

    print(f"\n{'='*60}")
    print(f"  Meeting   : {summary.get('meeting_id', '?')}")
    print(f"  Sentiment : {summary.get('overall_sentiment', '?')}")
    print(f"  Themes    : {', '.join(summary.get('key_themes', []))}")
    print(f"  Brief     : {summary.get('brief', '')}")
    print(f"\n  Functional Requirements     : {len(reqs)}")
    print(f"  Non-Functional Requirements : {len(nfrs)}")
    print(f"  Open Issues                 : {len(issues)}")

    if reqs:
        print(f"\n  Requirements by priority:")
        for pri in ("high", "medium", "low"):
            count = sum(1 for r in reqs if r.get("priority") == pri)
            if count:
                print(f"    {pri:<8} {count}")

    if issues:
        print(f"\n  Open Issues:")
        for oi in issues:
            print(f"    [{oi['id']}] {oi['description'][:70]}")

    inp = token_info.get("input_tokens",  0)
    out = token_info.get("output_tokens", 0)
    tot = token_info.get("total_tokens",  0)
    print(f"\n  Token usage:")
    print(f"    Input  : {inp:>8,}")
    print(f"    Output : {out:>8,}  / 65,536 max")
    print(f"    Total  : {tot:>8,}")
    if out >= 60000:
        print("  WARNING: output near token limit — response may be truncated.")
    print(f"{'='*60}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SRS from Step 1 JSON")
    parser.add_argument("--file", help="Path to *_step1.json")
    parser.add_argument("--id",   help="Meeting ID for output filename")
    args = parser.parse_args()

    # Resolve input file and meeting ID
    if args.file:
        json_path  = Path(args.file)
        meeting_id = args.id or json_path.stem.replace("_step1", "")
    else:
        json_path, meeting_id = pick_json_file()

    if not json_path.exists():
        sys.exit(f"File not found: {json_path}")

    print(f"\nInput  : {json_path}")
    print(f"Meeting: {meeting_id}")

    # Load Step 1 JSON
    with open(json_path, encoding="utf-8") as f:
        step1 = json.load(f)

    meta       = step1.get("meeting_meta", {})
    utterances = step1.get("utterances", [])
    print(f"Loaded {len(utterances)} utterances from Step 1.")

    # Build prompt and call Gemini
    user_prompt = build_user_prompt(meeting_id, meta, utterances)
    data, token_info = generate_srs(user_prompt)

    # Attach timestamps from Step 1 utterances (deterministic, no extra tokens)
    data = attach_timestamps(data, utterances)

    # Save and summarize
    out_path = save_result(data, meeting_id)
    print_summary(data, token_info)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
