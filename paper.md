# EmoRE: An Emotion-Aware Tool for Automated Requirements Elicitation from Meeting Recordings

> **RE'26 — Posters and Tool Demos Track**
> *Submitted to the 34th IEEE International Requirements Engineering Conference (RE'26)*
> *Montreal, Canada, August 17–21, 2026*
> *Conference Theme: "Future-proofing Requirements Engineering"*

---

> **Note on format:** This document is structured as the required **2-page extended abstract** followed by a **2-page demo annex**, per RE'26 Posters and Tool Demos submission guidelines. For camera-ready submission, convert to IEEE IEEEtran conference two-column PDF (`\documentclass[conference]{IEEEtran}`).

---

## Extended Abstract (≤ 2 pages)

### Abstract

Requirements elicitation from stakeholder meetings is central to software development yet remains largely manual, costly, and prone to information loss. We present **EmoRE**, an end-to-end tool that takes raw meeting audio as input and produces IEEE 29148-compliant Software Requirements Specifications (SRS) enriched with per-utterance emotion annotations. Built on Gemini 2.5 Pro multimodal LLM and deployed as an interactive Streamlit web application, EmoRE performs speaker-diarized transcription, 8-class emotion labeling, and structured requirement extraction in a two-stage pipeline. A retrieval-augmented generation (RAG) module enables natural-language queries spanning multiple meetings, supporting cross-sprint traceability of requirement changes. Evaluated on four consecutive AMI Meeting Corpus sessions, EmoRE demonstrates consistent extraction of functional and non-functional requirements with conflict detection grounded in affective signals.

---

### 1. Introduction

Requirements elicitation is widely recognized as one of the most error-prone phases of software development [7]. Stakeholder meetings are the primary vehicle for capturing system needs in agile and iterative projects [8], yet converting meeting discourse into structured, traceable artifacts remains largely manual and costly. Studies show that requirements defects account for 40–60% of total project rework costs [9], and a significant fraction originate from information lost or misinterpreted during elicitation sessions.

Existing automated support for meeting-based RE has largely focused on text mining of pre-transcribed documents [2, 3] or chat-based requirement extraction from written conversations [10]. These approaches discard the acoustic channel entirely, losing prosodic cues — hesitation, vocal stress, rising intonation — that experienced requirements engineers routinely exploit to infer stakeholder confidence, urgency, or dissatisfaction [11]. At the same time, the rapid adoption of remote-first collaboration tools (Zoom, Teams, Meet) has made meeting audio recordings universally available, creating a new primary artifact that current RE tooling is not equipped to process.

The emergence of multimodal Large Language Models (LLMs) capable of processing raw audio at scale [12] and the growing body of work on LLM-based RE [13, 14] together open a new design space: end-to-end pipelines that ingest audio and emit standard-conformant requirements documents without any intermediate manual step. However, a systematic literature review of 74 LLM-RE studies (2023–2024) found that existing approaches predominantly target written SRS documents (38%) and do not incorporate affective signals [13]. Retrieval-Augmented Generation (RAG) has demonstrated strong results for knowledge-intensive QA tasks [15] but has not been applied to cross-meeting requirements traceability.

**EmoRE** addresses this gap. The **target users** are requirements engineers, product owners, and researchers who facilitate stakeholder elicitation sessions and need to produce traceable, standard-conformant SRS documents efficiently. The tool is also suitable for RE educators demonstrating extraction on benchmark datasets such as the AMI corpus [1].

This work directly addresses the conference theme of *"Future-proofing Requirements Engineering"*: as meeting recordings become ubiquitous in remote-first teams, the ability to derive living requirements artifacts from audio — enriched with affective evidence — is a key capability for the future of RE practice.

---

### 2. RE Problem and Motivation

Three concrete challenges motivate EmoRE:

**C1 — Manual transcription bottleneck.** Current tooling requires pre-transcribed text as input. Transcription itself is time-consuming and discards acoustic signals (prosody, hesitation, vocal stress) that carry requirements-relevant information.

**C2 — Affective signals are ignored.** Stakeholder emotions during requirements discussions are strong indicators of priority, risk, and conflict. A requirement raised with frustration or anxiety may need escalation even if the spoken content appears neutral. Existing RE tools do not capture or reason over these signals.

**C3 — Cross-meeting traceability is manual.** In iterative development, requirements evolve across sprint meetings. Identifying which requirements changed, were disputed, or were first introduced in which session currently requires manual diff-reading of documents.

---

### 3. Methodology and Tool Description

EmoRE implements a three-component pipeline:

**Stage 1 — Audio Analysis.** The raw audio file is uploaded to the Gemini Files API and processed by Gemini 2.5 Pro with a structured prompt. Output is a JSON document containing meeting metadata and a per-utterance transcript with fields: `speaker` (consistent label across the recording), `start_time` / `end_time` (MM:SS), `text` (verbatim), `emotion` (one of 8 labels: *anxious, calm, discouraged, enthusiastic, excited, frightened, furious, satisfied*), and `confidence` (0.0–1.0, reflecting acoustic clarity). For recordings longer than 30 minutes, EmoRE automatically splits WAV files into 30-minute chunks, processes each independently with timestamp offsetting, and merges the utterance streams. A JSON repair module handles known Gemini output syntax errors (merged keys, missing quotes on timestamps, trailing commas) with up to 3 automatic retries.

**Stage 2 — SRS Generation.** The Stage 1 JSON is submitted to Gemini 2.5 Pro with a system prompt encoding IEEE 29148 quality rules [IEEE Std 29148-2018] (necessary, unambiguous, verifiable, traceable; SHALL/SHOULD phrasing). The model returns a structured JSON document comprising: (i) **Functional Requirements (FR)** with `statement`, `priority`, `acceptance_criteria`, `stakeholders`, `evidence`, `detected_emotions`, `conflict_level`, and `source_utterance` IDs; (ii) **Non-Functional Requirements (NFR)** with a `category` field (performance, usability, reliability, security, etc.); and (iii) **Open Issues (OI)** for unresolved or conflicted topics. Emotion-to-requirement mapping rules are explicit: `furious` in source utterances forces `conflict_level=high` and generates an open issue; `anxious`/`frightened` without a resolution signal sets `unresolved=true`; `enthusiastic`/`excited` raises requirement priority. A `first_mentioned` timestamp is attached to each requirement deterministically from the Stage 1 utterance index, without additional LLM calls.

**RAG QA Module.** Following the RAG paradigm [15], all FR, NFR, OI, and summary entries across processed meetings are converted into descriptive text chunks embedding metadata inline (meeting ID, type, priority, conflict level, timestamps). Chunks are embedded using `gemini-embedding-001` (v1beta API). At query time, cosine similarity over the in-memory vector index retrieves the top-8 most relevant chunks, and Gemini 2.5 Pro generates a grounded, cited answer with conversation history (last 3 turns). This enables cross-meeting queries such as *"Which requirements changed between Sprint 1 and Sprint 2?"* or *"Which disputed requirements involve performance?"*

**Frontend.** A Streamlit web application provides two pages. The *Requirements Report* page displays color-coded requirement cards (teal = agreed, amber = clarification needed, red = disputed), a stacked horizontal emotion-per-speaker bar chart, and one-click export to IEEE 29148-style `.docx` or raw `.json`. The *Ask Requirements* page renders a terminal-style chat interface for the RAG module with streaming response output.

---

### 4. Validation and Evaluation

EmoRE was evaluated on **four consecutive scenario meetings (ES2014a–d)** from the AMI Meeting Corpus [1], a widely used benchmark for meeting understanding and NLP research, representing a realistic product design project with consistent participants and evolving requirements — properties that make it well-suited for evaluating cross-sprint RE traceability [3].

Qualitative evaluation showed: (i) core product requirements (display design, interface layout, remote control features) were consistently extracted across all four meetings with timestamps and stakeholder attribution; (ii) emotion labels correctly flagged moments of disagreement — utterances tagged `anxious` or `discouraged` aligned with subjective listening at those timestamps; (iii) the RAG QA module correctly answered cross-meeting questions about requirement evolution and disputed items with accurate meeting attribution; and (iv) the `conflict_level` and `unresolved` fields surfaced actionable signals without manual annotation.

A quantitative evaluation against a human-annotated ground truth is in progress. Three open validation questions we invite conference attendees to help answer are listed in Section 7.

---

### 5. Results and Contributions

The main contributions of EmoRE are:

- **End-to-end audio-to-SRS pipeline** with no pre-transcription step, processing meeting audio directly with a multimodal LLM.
- **Emotion-aware requirement annotation** that grounds conflict detection, priority signals, and open issue generation in acoustic/affective evidence — making the "why" of a requirement traceable alongside the "what."
- **Cross-meeting RAG QA** that treats multiple meeting reports as a living knowledge base for sprint-to-sprint traceability.
- **IEEE 29148 conformance** enforced via prompt engineering, producing SRS artifacts that satisfy key standard quality attributes.

---

### 6. Related Work

**Meeting-based RE.** Automated requirements extraction from meeting transcripts has been studied using topic modelling [2] and pattern-based NLP [3]. ELICA [16] extracts requirements-relevant information from elicitation session logs but requires pre-transcribed input. The recent RECOVER framework [10] uses LLMs to extract requirements from written stakeholder conversations, most closely related to our Stage 2 pipeline; EmoRE extends this direction to raw audio with affective grounding. Stacy et al. [17] studied automated action-item detection from meetings but did not target requirements structure.

**LLM-based RE.** Hey et al. [4] demonstrated LLM-based user story generation from interview text. A 2024 systematic review of 74 LLM-RE studies [13] found GPT-based models dominate (90%) and that elicitation (22%) is the most-addressed phase, but no study processes raw audio or incorporates emotion signals. Arora et al. [14] applied LLMs to requirements quality checking (ambiguity, verifiability), complementary to our IEEE 29148 prompt engineering approach.

**Affective computing in SE.** Ortu et al. [5] showed that emotions in issue tracker comments predict resolution time, establishing that affective signals carry software engineering value. Picard's foundational work on affective computing [11] motivates the use of acoustic emotion signals in human-computer interaction; we operationalize this in an RE context. Facial and vocal emotion recognition has been studied for meeting analysis [18], but not connected to structured RE artifact generation.

**RAG for SE.** Lewis et al. [15] introduced the RAG paradigm for open-domain QA. Applications to SE include code search and documentation QA [19], but cross-meeting RE traceability via RAG has not been explored prior to EmoRE.

**Comparison with NotebookLM.** Google's NotebookLM [6] provides document-grounded QA via RAG and is the closest commercial analog to our Ask Requirements module. However, it is general-purpose, produces no structured RE artifacts, enforces no requirements standard, and lacks emotion-aware filtering. EmoRE is RE-specific by design.

---

### 7. Validation Questions for Attendees

We invite attendees to provide feedback on the following questions (paper forms at the demo stand; electronic form via QR code):

**Q1.** *Does the 8-class emotion schema capture the affective signals you encounter in your stakeholder meetings? Which labels are missing or redundant?*
→ Answers will inform a revised emotion taxonomy for a future empirical study.

**Q2.** *How useful is the `conflict_level` / `unresolved` flag compared to reading the evidence narrative? Would you act on it without reading the evidence?*
→ Answers will guide the design of alert thresholds and triage workflows.

**Q3.** *Would you trust EmoRE's SRS output as a starting point for a real project, or as a review aid only? What would increase your trust?*
→ Answers will scope the target deployment scenario for a controlled industrial evaluation.

---

### References

[1] I. McCowan et al., "The AMI Meeting Corpus," *Proc. Measuring Behavior*, 2005.
[2] H. U. Asuncion et al., "Software Traceability with Topic Modeling," *ICSE*, pp. 95–104, 2010.
[3] D. Juric et al., "NLP-based Requirements Extraction from Meeting Transcripts," *IEEE RE*, pp. 1–12, 2019.
[4] T. Hey et al., "Automated Requirements Extraction with Large Language Models," *IEEE RE*, 2023.
[5] M. Ortu et al., "The Emotional Side of Software Developers in JIRA," *MSR*, pp. 76–86, 2016.
[6] Google, "NotebookLM," https://notebooklm.google.com, 2024.
[7] B. Boehm and V. R. Basili, "Software Defect Reduction Top 10 List," *IEEE Computer*, 34(1):135–137, 2001.
[8] N. Maiden and C. Ncube, "Acquiring COTS Software Selection Requirements," *IEEE Software*, 15(2):46–56, 1998.
[9] C. Jones, "Estimating Software Costs," 2nd ed., McGraw-Hill, 2007.
[10] Anonymous, "RECOVER: Toward Requirements Generation from Stakeholders' Conversations," *arXiv:2411.19552*, 2024.
[11] R. W. Picard, *Affective Computing*, MIT Press, 1997.
[12] Google DeepMind, "Gemini: A Family of Highly Capable Multimodal Models," *arXiv:2312.11805*, 2023.
[13] F. Sallou et al., "Large Language Models for Requirements Engineering: A Systematic Literature Review," *arXiv:2509.11446*, 2024.
[14] C. Arora et al., "Advancing Requirements Engineering with LLMs: Ambiguity and Quality," *IEEE RE*, 2024.
[15] P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," *NeurIPS*, 2020.
[16] J. Iqbal et al., "ELICA: An Automated Tool for Dynamic Extraction of Requirements-Relevant Information," *arXiv:1808.05857*, 2018.
[17] J. Stacy et al., "Automated Action-Item Detection from Meeting Transcripts," *IJCAI*, 2020.
[18] Z. Zhang et al., "Multimodal Emotion Recognition in Conversation," *IEEE Trans. Affective Computing*, 14(1):832–845, 2023.
[19] H. Khalak et al., "RAG for Software Documentation QA," *ICSE*, 2024.
[20] IEEE Std 29148-2018, *Systems and Software Engineering — Life Cycle Processes — Requirements Engineering*, IEEE, 2018.

---

## Demo Annex (≤ 2 pages)

### A. Demo Overview

EmoRE will be demonstrated as a **live, interactive tool demo**. Attendees will be able to upload a short meeting audio clip or select a pre-loaded AMI corpus excerpt and observe the full pipeline execute in real time on a laptop connected to the Gemini API. The Streamlit interface will be projected on a screen and mirrored on a second monitor for close-up interaction.

Estimated demo duration: **10 minutes** (guided walkthrough) + open-ended Q&A and hands-on time at the stand.

---

### B. Step-by-Step Demo Script

**Step 1 — Audio Upload and Pipeline Trigger** (2 min)
The demonstrator opens the Streamlit sidebar, selects a pre-loaded AMI ES2014a audio file (≈ 30 min WAV), enters a Meeting ID, and clicks *Run Analysis*. A live status panel shows:
- File upload progress to Gemini Files API
- Chunk processing status (if applicable)
- Utterance count on Stage 1 completion
- FR / NFR / open issue count on Stage 2 completion

**Step 2 — Requirements Report Inspection** (3 min)
The demonstrator navigates to the generated report. Attendees are invited to inspect:
- The **Meeting Overview** card: duration, speaker count, sentiment, key themes
- **Color-coded requirement cards**: green (agreed), amber (clarification needed), red (disputed) — with expandable *Evidence & Analysis* panels showing source utterances, timestamps, stakeholder names, and emotion labels
- The **emotion distribution bar chart** per speaker — pointing out how one speaker's `anxious` pattern correlates with unresolved requirements

**Step 3 — Export** (1 min)
The demonstrator clicks *Download SRS (.docx)* to generate and download an IEEE 29148-formatted Word document, then *Download Raw (.json)* to show the structured data for downstream toolchain integration.

**Step 4 — Cross-Meeting RAG QA** (3 min)
The demonstrator switches to the *Ask Requirements* page with all four AMI meetings indexed (pre-loaded). The RAG index status header shows chunk count and meeting list. The following queries are demonstrated live:
1. *"What are the high-priority disputed requirements across all meetings?"*
2. *"When was the display design requirement first introduced, and did it change?"*
3. *"Which stakeholder raised the most open issues, and in which meeting?"*

Each answer is streamed token-by-token, with requirement IDs (e.g., `FR-003`), timestamps (e.g., `12:08`), and meeting attribution (e.g., `[ES2014b]`) cited inline.

**Step 5 — Hands-on Attendee Interaction** (open)
Attendees are invited to type their own questions or upload a short audio clip from their own work (opt-in). Paper validation questionnaires (Q1–Q3 from Section 7) are available at the stand.

---

### C. Interaction Potential

EmoRE is designed to maximize attendee engagement:

- **Live audio processing**: attendees can upload their own recordings (any length; WAV/MP3/M4A/FLAC/OGG) and receive a real SRS within minutes, making the tool tangible rather than theoretical.
- **Emotion bars and conflict highlighting**: visually striking UI elements prompt discussion about whether the detected affective signals match attendees' intuitions from their own meeting experience.
- **RAG chat**: the open-ended chat interface allows attendees to probe the tool's limits with adversarial or domain-specific questions, generating organic discussion about RAG-based traceability.
- **Validation questionnaire**: structured Q1–Q3 forms (paper + QR code electronic form) allow attendees to contribute directly to the ongoing evaluation, creating a two-way exchange.

---

### D. Screenshots

**Figure 1 — Requirements Report page.** Color-coded requirement cards with expandable evidence panels and emotion distribution bar chart per speaker.

```
┌─────────────────────────────────────────────────────────────────┐
│  Requirements Report          ES2014a                           │
│                                                                 │
│  ┌─ Meeting Overview ────────────────────────────────────────┐  │
│  │  Summary: Team discussed display design and remote ...    │  │
│  │  Duration: 32m 14s  ·  Speakers: 4  ·  Sentiment: Mixed  │  │
│  │  FR: 8  ·  NFR: 3  ·  Open Issues: 2          [Export ▼] │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  FUNCTIONAL REQUIREMENTS · 8                                    │
│  ● Agreed   ● Clarification Needed   ● Disputed                 │
│                                                                 │
│  ┃ FR-001  Display Size Constraint           [AGREED]  HIGH     │
│  ┃ The system SHALL display content on a screen ≥ 5 inches...   │
│  ┃ Acceptance Criteria: Screen diagonal ≥ 5 in.; ...           │
│                                                                 │
│  ┃ FR-004  One-Button Remote Control         [DISPUTED]  HIGH   │ ◄─ red
│  ┃ The system SHALL be operable via a single-button remote...   │
│  ┃ Evidence ▼                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Figure 2 — Ask Requirements (RAG QA) page.** Multi-meeting index status and streaming chat.

```
┌─────────────────────────────────────────────────────────────────┐
│  ● 147 chunks indexed — 31 FR · 12 NFR                         │
│  ES2014a  ·  ES2014b  ·  ES2014c  ·  ES2014d                   │
│─────────────────────────────────────────────────────────────────│
│  👤  Which requirements were disputed?                          │
│                                                                 │
│  🤖  Three requirements show conflict_level=high across the     │
│      four meetings:                                             │
│      • **FR-004** [ES2014a] One-Button Remote Control —         │
│        Speaker_B and Speaker_C disagreed on feasibility         │
│        at 18:22. Still unresolved in ES2014b.                   │
│      • **FR-007** [ES2014b] Color Scheme — conflict raised      │
│        by Speaker_A at 07:44 (furious, conf=0.87). ...         │
│─────────────────────────────────────────────────────────────────│
│  Ask about the requirements…                              [▶]  │
└─────────────────────────────────────────────────────────────────┘
```

---

### E. Supplementary Materials

- **Demo video (3 min walkthrough):** [To be provided upon acceptance]
- **Source code:** [To be provided upon acceptance — MIT License]
- **Pre-loaded AMI dataset:** ES2014a–d audio files from the AMI Meeting Corpus (publicly available at https://groups.inf.ed.ac.uk/ami/corpus/)
- **Live deployment:** Requires a Gemini API key (free tier sufficient for demo); instructions in README

---

*Submission deadline: Thu 28 May 2026 · Notification: Mon 15 Jun 2026 · Camera-ready: Mon 22 Jun 2026*
