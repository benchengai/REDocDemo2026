"""
test_gemini.py — 单文件会议分析
  Pass 1 (音频 → Gemini) : 转录 + speaker分离 + 声学情绪  [多模态]
  跨段speaker匹配         : resemblyzer 声纹统一标签
  Pass 2 (文本 → Gemini) : RE需求信号提取                 [纯文字]
  合并输出完整JSON

用法:
    python test_gemini.py data/audio/meeting.wav
    python test_gemini.py          # 自动找 data/audio/ 中第一个音频
    python test_gemini.py --no_reid   # 跳过声纹匹配（调试）
"""

import argparse, json, os, re, sys, shutil, tempfile, time
from pathlib import Path

import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

# ── 声纹依赖 ────────────────────────────────────────────────────────────────────
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    from pydub import AudioSegment
    REID_AVAILABLE = True
    PYDUB_AVAILABLE = True
except ImportError:
    REID_AVAILABLE = False
    try:
        from pydub import AudioSegment
        PYDUB_AVAILABLE = True
    except ImportError:
        PYDUB_AVAILABLE = False

# ── 环境 ────────────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    sys.exit("ERROR: GEMINI_API_KEY not found in .env")
genai.configure(api_key=API_KEY)

# ── 常量 ────────────────────────────────────────────────────────────────────────
MIME_MAP = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".mp4": "video/mp4",
    ".m4a": "audio/mp4", ".flac": "audio/flac", ".ogg": "audio/ogg",
}
CHUNK_MIN            = 30      # 超过此时长才分段（分钟）
SIMILARITY_THRESHOLD = 0.78
MIN_CLIP_MS          = 2000
PASS2_BATCH          = 60      # Pass2 每批处理句数


# ══════════════════════════════════════════════════════════════════════════════
# Prompt 定义
# ══════════════════════════════════════════════════════════════════════════════

def prompt_pass1(known_speakers: list[str]) -> str:
    """Pass 1：音频转录 + speaker + 声学情绪（紧凑格式）"""
    ctx = ""
    if known_speakers:
        ctx = (f"IMPORTANT: continuation of same meeting. "
               f"Previous speakers: {', '.join(known_speakers)}. "
               f"Reuse same labels for same voices.\n")
    return f"""{ctx}You are a meeting transcriptionist and emotion analyst.
Analyze this audio. Return a JSON array ONLY — no markdown, no extra text.

Each element must be exactly:
{{"id":<n>,"speaker":"Speaker_A","start":"MM:SS","end":"MM:SS","text":"<transcript>",
  "emotion":"<neutral|positive|negative|frustrated|excited|uncertain|satisfied>",
  "emo_conf":<0.0-1.0>,
  "emo_cue":"<max 8 words: acoustic cues — tone/pitch/pace/tremor>"}}

Rules:
- Consistent speaker labels (Speaker_A, Speaker_B ...) throughout.
- emotion MUST reflect vocal/acoustic signals, not just word meaning.
- Include every speaking turn.
- Return ONLY the JSON array.
"""

PROMPT_PASS2_TMPL = """You are a requirements engineer analyzing a meeting transcript.
Analyze the utterances below and extract RE signals.
Return a JSON array ONLY — no markdown, no extra text.

Only include utterances that have a meaningful requirement (skip trivial ones).
Each element:
{{"id":<same id>,"req_type":"<functional|non-functional|constraint|stakeholder_need>",
  "priority":"<high|medium|low>",
  "requirement":"<The system shall ... — IEEE 830 style>",
  "topics":["<tag>"]}}

UTTERANCES:
{text}
"""


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def get_model() -> str:
    candidates = ["gemini-2.5-flash","gemini-2.5-flash-preview-04-17",
                  "gemini-2.5-pro","gemini-1.5-pro","gemini-1.5-flash"]
    available = {m.name.replace("models/","")
                 for m in genai.list_models()
                 if "generateContent" in m.supported_generation_methods}
    for c in candidates:
        if c in available:
            return c
    sys.exit(f"No available model. Found: {sorted(available)}")


def call_gemini(model_name: str, contents: list, label: str) -> str:
    try:
        gen_cfg = genai.types.GenerationConfig(
            temperature=0.1, top_p=0.95, max_output_tokens=65536,
            thinking_config=genai.types.ThinkingConfig(thinking_budget=0))
    except Exception:
        gen_cfg = genai.types.GenerationConfig(
            temperature=0.1, top_p=0.95, max_output_tokens=65536)

    model = genai.GenerativeModel(model_name=model_name, generation_config=gen_cfg)

    for attempt in range(4):
        try:
            return model.generate_content(contents).text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err and attempt < 3:
                m = re.search(r"retry in (\d+)", err)
                wait = int(m.group(1)) + 5 if m else 60
                print(f"  ⚠ [{label}] 限流，等待 {wait}s [{attempt+1}/3] ...")
                time.sleep(wait)
            else:
                raise


def parse_json_safe(raw: str) -> any:
    """解析JSON，失败时用 json_repair 修复。"""
    # 清洗 markdown
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  ⚠ JSON问题({e})，尝试修复 ...")
        try:
            from json_repair import repair_json
            result = repair_json(raw, return_objects=True)
            if result:
                print(f"  → 修复成功")
                return result
        except ImportError:
            print("  ⚠ 请安装: pip install json-repair")
        raise


def ts_to_sec(ts: str) -> int:
    p = ts.split(":")
    if len(p) == 2: return int(p[0])*60 + int(p[1])
    return int(p[0])*3600 + int(p[1])*60 + int(p[2])

def sec_to_ts(s: int) -> str:
    m, s = divmod(int(s), 60)
    return f"{m:02d}:{s:02d}"

def ts_to_ms(ts: str) -> int:
    return ts_to_sec(ts) * 1000

def offset_ts(ts: str, offset: int) -> str:
    return sec_to_ts(ts_to_sec(ts) + offset)


# ══════════════════════════════════════════════════════════════════════════════
# 音频处理
# ══════════════════════════════════════════════════════════════════════════════

def duration_min(audio_path: Path) -> float:
    audio = AudioSegment.from_file(str(audio_path))
    return len(audio) / 60000

def split_audio(audio_path: Path) -> list[tuple[Path, int]]:
    """切分音频，返回 [(chunk_path, offset_sec)]。"""
    audio   = AudioSegment.from_file(str(audio_path))
    total   = len(audio)
    chunk_ms = CHUNK_MIN * 60000
    n_chunks = (total + chunk_ms - 1) // chunk_ms
    print(f"  [分段] {total/60000:.1f}分钟 → {n_chunks}段（每段≤{CHUNK_MIN}分钟）")

    tmp_dir = Path(tempfile.mkdtemp(prefix="re_"))
    chunks  = []
    for i in range(n_chunks):
        s = i * chunk_ms
        e = min((i+1)*chunk_ms, total)
        path = tmp_dir / f"chunk_{i+1:02d}.wav"
        audio[s:e].export(str(path), format="wav")
        offset = s // 1000
        print(f"  [分段] 段{i+1}: {sec_to_ts(offset)} → {sec_to_ts(e//1000)}")
        chunks.append((path, offset))
    return chunks


def upload_file(path: Path) -> genai.types.File:
    mime = MIME_MAP.get(path.suffix.lower(), "audio/wav")
    mb   = path.stat().st_size / 1024 / 1024
    print(f"  [上传] {path.name} ({mb:.1f}MB) ...")
    f = genai.upload_file(str(path), mime_type=mime)
    for _ in range(30):
        if genai.get_file(f.name).state.name == "ACTIVE":
            return f
        time.sleep(3)
    sys.exit("ERROR: 上传超时")


# ══════════════════════════════════════════════════════════════════════════════
# Pass 1：音频分析（转录 + 声学情绪）
# ══════════════════════════════════════════════════════════════════════════════

def pass1_single(audio_path: Path, model: str) -> list[dict]:
    """单段：直接上传原始文件。"""
    print(f"  [Pass1] 单段直传")
    f = upload_file(audio_path)
    try:
        raw  = call_gemini(model, [prompt_pass1([]), f], "Pass1")
        data = parse_json_safe(raw)
        print(f"  [Pass1] 完成 → {len(data)} 条")
        return data
    finally:
        try: genai.delete_file(f.name)
        except: pass


def pass1_chunks(audio_path: Path, model: str) -> list[dict]:
    """多段：切分后逐段分析，合并时加时间偏移。"""
    chunks      = split_audio(audio_path)
    all_utts    = []
    known_spks  = []
    global_id   = 1

    for i, (chunk_path, offset) in enumerate(chunks):
        print(f"\n  [Pass1] 段{i+1}/{len(chunks)} ...")
        f = upload_file(chunk_path)
        try:
            raw  = call_gemini(model, [prompt_pass1(known_spks), f], f"P1-段{i+1}")
            data = parse_json_safe(raw)
        finally:
            try: genai.delete_file(f.name)
            except: pass

        for u in data:
            u = dict(u)
            u["id"]    = global_id
            u["start"] = offset_ts(u.get("start","00:00"), offset)
            u["end"]   = offset_ts(u.get("end","00:00"),   offset)
            # 兼容 start_time/end_time 字段名
            u.pop("start_time", None); u.pop("end_time", None)
            all_utts.append(u)
            global_id += 1
            if u.get("speaker") and u["speaker"] not in known_spks:
                known_spks.append(u["speaker"])

        print(f"  [Pass1] 段{i+1} 完成 → {len(data)} 条，累计 {len(all_utts)} 条")
        if i < len(chunks)-1:
            time.sleep(2)

    # 清理临时文件
    tmp_dir = chunks[0][0].parent
    shutil.rmtree(str(tmp_dir), ignore_errors=True)
    return all_utts


# ══════════════════════════════════════════════════════════════════════════════
# 跨段 Speaker 匹配（resemblyzer）
# ══════════════════════════════════════════════════════════════════════════════

def match_speakers_across_chunks(audio_path: Path, utterances: list[dict]) -> list[dict]:
    """
    从原始音频提取每个(chunk_内局部speaker)的声纹，
    做跨段匹配，统一为 Person_1/2/3... 标签。
    只有多段时才需要；单段时 Speaker_A/B/C 本身已一致。
    """
    if not REID_AVAILABLE:
        print("  ⚠ resemblyzer 未安装，跳过跨段匹配")
        return utterances

    print(f"  [声纹] 加载 VoiceEncoder ...")
    encoder = VoiceEncoder()
    audio   = AudioSegment.from_file(str(audio_path))

    # 找出所有不同的局部 speaker 标签
    local_speakers = sorted(set(u.get("speaker","") for u in utterances if u.get("speaker")))
    print(f"  [声纹] 发现局部标签: {local_speakers}")

    # 提取每个 speaker 的声纹
    emb_map: dict[str, np.ndarray] = {}
    for spk in local_speakers:
        clips = sorted(
            [(ts_to_ms(u["end"]) - ts_to_ms(u["start"]),
              ts_to_ms(u["start"]), ts_to_ms(u["end"]))
             for u in utterances
             if u.get("speaker") == spk
             and (ts_to_ms(u.get("end","00:00")) - ts_to_ms(u.get("start","00:00"))) >= MIN_CLIP_MS],
            reverse=True
        )[:8]

        if not clips:
            print(f"  ⚠ {spk} 无足够长片段")
            continue

        embs = []
        with tempfile.TemporaryDirectory() as tmp:
            for j, (_, s, e) in enumerate(clips):
                p = Path(tmp)/f"{spk}_{j}.wav"
                audio[s:e].set_frame_rate(16000).set_channels(1).export(str(p), format="wav")
                try:
                    embs.append(encoder.embed_utterance(preprocess_wav(str(p))))
                except: pass
        if embs:
            emb_map[spk] = np.mean(embs, axis=0)
            print(f"  [声纹] {spk}: {len(embs)}段提取完成")

    if not emb_map:
        print("  ⚠ 无声纹数据，保留原始标签")
        return utterances

    # 贪心匹配：相似度最高且超阈值的合并为同一 Person
    person_embs: dict[str, np.ndarray] = {}   # Person_N → embedding
    local_to_person: dict[str, str]    = {}

    for spk, emb in emb_map.items():
        best_pid, best_sim = None, 0.0
        for pid, pemb in person_embs.items():
            sim = float(np.dot(emb, pemb) /
                        (np.linalg.norm(emb)*np.linalg.norm(pemb) + 1e-9))
            if sim > best_sim:
                best_sim, best_pid = sim, pid

        if best_sim >= SIMILARITY_THRESHOLD:
            # 合并到已有 Person
            local_to_person[spk] = best_pid
            # 更新 Person embedding（滑动平均）
            person_embs[best_pid] = (person_embs[best_pid] + emb) / 2
            print(f"  [声纹] {spk} → {best_pid} (相似度 {best_sim:.2f}，同一人)")
        else:
            # 新建 Person
            pid = f"Person_{len(person_embs)+1}"
            person_embs[pid] = emb
            local_to_person[spk] = pid
            print(f"  [声纹] {spk} → {pid} (新说话人，最高相似度 {best_sim:.2f})")

    # 更新 utterances 里的 speaker
    for u in utterances:
        orig = u.get("speaker","")
        u["speaker_local"] = orig
        u["speaker"]       = local_to_person.get(orig, orig)

    return utterances


# ══════════════════════════════════════════════════════════════════════════════
# Pass 2：文本需求分析
# ══════════════════════════════════════════════════════════════════════════════

def pass2_re_signals(utterances: list[dict], model: str) -> dict[int, dict]:
    """
    对完整转录文本做需求提取。
    - 只发送 text 长度 >= 15 的句子
    - 每批 PASS2_BATCH 句
    - 只返回有需求的 utterance（节省token）
    - 返回 {id → {req_type, priority, requirement, topics}}
    """
    meaningful = [u for u in utterances if len(u.get("text","")) >= 15]
    print(f"  [Pass2] 共 {len(meaningful)} 条有效句，分 "
          f"{(len(meaningful)+PASS2_BATCH-1)//PASS2_BATCH} 批分析")

    req_map: dict[int, dict] = {}
    batches = [meaningful[i:i+PASS2_BATCH] for i in range(0, len(meaningful), PASS2_BATCH)]

    for bi, batch in enumerate(batches):
        lines = [f"[ID:{u['id']}][{u['speaker']}]: {u['text']}" for u in batch]
        prompt = PROMPT_PASS2_TMPL.format(text="\n".join(lines))
        print(f"  [Pass2] 批次 {bi+1}/{len(batches)} ({len(batch)}句) ...")

        try:
            raw     = call_gemini(model, [prompt], f"P2-b{bi+1}")
            results = parse_json_safe(raw)
            if isinstance(results, list):
                for item in results:
                    uid = item.get("id")
                    if uid:
                        req_map[uid] = {
                            "req_type":    item.get("req_type","none"),
                            "priority":    item.get("priority","none"),
                            "requirement": item.get("requirement"),
                            "topics":      item.get("topics",[]),
                        }
        except Exception as e:
            print(f"  ⚠ 批次{bi+1}失败: {e}")

        if bi < len(batches)-1:
            time.sleep(2)

    print(f"  [Pass2] 完成，提取到 {len(req_map)} 条需求信号")
    return req_map


# ══════════════════════════════════════════════════════════════════════════════
# 合并 & 输出
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_RE = {"req_type":"none","priority":"none","requirement":None,"topics":[]}

def build_final(utterances: list[dict], req_map: dict[int,dict]) -> dict:
    """将 Pass1 和 Pass2 结果合并成最终 JSON 结构。"""
    final_utts = []
    for u in utterances:
        re = req_map.get(u["id"], DEFAULT_RE)
        final_utts.append({
            "id":             u["id"],
            "speaker":        u.get("speaker",""),
            "speaker_local":  u.get("speaker_local", u.get("speaker","")),
            "start_time":     u.get("start",""),
            "end_time":       u.get("end",""),
            "text":           u.get("text",""),
            "emotion": {
                "primary":      u.get("emotion","neutral"),
                "confidence":   u.get("emo_conf", 0.5),
                "cues":         u.get("emo_cue",""),
                "source":       "audio",   # 明确标注情绪来自音频
            },
            "re_signals": {
                "requirement_type":      re["req_type"],
                "priority_hint":         re["priority"],
                "extracted_requirement": re["requirement"],
                "topics":                re["topics"],
            }
        })

    # 情绪分布
    emo_count: dict[str,int] = {}
    for u in final_utts:
        e = u["emotion"]["primary"]
        emo_count[e] = emo_count.get(e,0) + 1
    total = max(len(final_utts),1)
    emo_dist = {k: round(v/total,3) for k,v in emo_count.items()}

    # 需求汇总
    reqs = [u for u in final_utts if u["re_signals"]["extracted_requirement"]]

    return {
        "meeting_meta": {
            "num_speakers":  len(set(u["speaker"] for u in final_utts)),
            "num_utterances": len(final_utts),
            "num_requirements": len(reqs),
            "speakers": sorted(set(u["speaker"] for u in final_utts)),
        },
        "utterances": final_utts,
        "emotion_distribution": emo_dist,
        "requirements_summary": [
            {"speaker": u["speaker"],
             "start_time": u["start_time"],
             "type": u["re_signals"]["requirement_type"],
             "priority": u["re_signals"]["priority_hint"],
             "requirement": u["re_signals"]["extracted_requirement"]}
            for u in reqs
        ],
    }


def print_summary(result: dict, audio_name: str):
    meta  = result["meeting_meta"]
    dist  = result.get("emotion_distribution",{})
    reqs  = result.get("requirements_summary",[])
    spks  = meta.get("speakers",[])

    print(f"\n{'='*60}")
    print(f"分析完成: {audio_name}")
    print(f"{'='*60}")
    print(f"说话人  : {', '.join(spks)}")
    print(f"发言轮次: {meta['num_utterances']}")
    print(f"提取需求: {meta['num_requirements']} 条")

    if dist:
        top = max(dist.items(), key=lambda x: x[1])
        print(f"主情绪  : {top[0]} ({top[1]:.0%})")
        print(f"情绪分布: " + "  ".join(f"{k}:{v:.0%}" for k,v in
              sorted(dist.items(), key=lambda x:-x[1]) if v>0.03))

    if reqs:
        print(f"\n需求清单:")
        for r in reqs[:8]:
            print(f"  [{r['type'].upper()}][{r['priority']}] {r['requirement']}")
        if len(reqs) > 8:
            print(f"  ... 还有 {len(reqs)-8} 条，见 JSON")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="?", help="音频文件路径（不填则自动找 data/audio/）")
    parser.add_argument("--output_dir", default="data/outputs")
    parser.add_argument("--no_reid",    action="store_true", help="跳过声纹匹配")
    args = parser.parse_args()

    # 确定音频文件
    if args.audio:
        audio_path = Path(args.audio)
    else:
        audio_dir = Path("data/audio")
        files = sorted([f for f in audio_dir.iterdir()
                        if f.suffix.lower() in MIME_MAP])
        if not files:
            sys.exit("ERROR: data/audio/ 中没有音频文件")
        audio_path = files[0]
        print(f"自动选择: {audio_path.name}")

    if not audio_path.exists():
        sys.exit(f"ERROR: 文件不存在: {audio_path}")

    output_dir  = Path(args.output_dir)
    output_path = output_dir / f"{audio_path.stem}_analysis.json"

    # 依赖检查
    if not PYDUB_AVAILABLE:
        sys.exit("ERROR: 请安装 pydub + ffmpeg\n"
                 "  pip install pydub\n"
                 "  conda install -c conda-forge ffmpeg")

    use_reid = not args.no_reid and REID_AVAILABLE
    if not args.no_reid and not REID_AVAILABLE:
        print("⚠ resemblyzer 未安装，跳过声纹匹配（pip install resemblyzer）\n")

    model = get_model()
    print(f"模型: {model}")
    print(f"文件: {audio_path.name}\n")

    # ── Pass 1：音频分析 ──────────────────────────────────────────────────────
    dur = duration_min(audio_path)
    print(f"[Step 1] 音频转录 + 情绪分析（{dur:.1f} 分钟）")
    if dur <= CHUNK_MIN:
        utterances = pass1_single(audio_path, model)
    else:
        utterances = pass1_chunks(audio_path, model)

    # ── 跨段 Speaker 匹配 ─────────────────────────────────────────────────────
    if dur > CHUNK_MIN and use_reid:
        print(f"\n[Step 2] 跨段 Speaker 声纹匹配")
        utterances = match_speakers_across_chunks(audio_path, utterances)
    else:
        # 单段无需匹配，Speaker_A/B/C 已一致
        for u in utterances:
            u["speaker_local"] = u.get("speaker","")

    # ── Pass 2：需求信号提取 ──────────────────────────────────────────────────
    print(f"\n[Step 3] 需求信号提取（文本分析）")
    req_map = pass2_re_signals(utterances, model)

    # ── 合并 & 保存 ───────────────────────────────────────────────────────────
    print(f"\n[Step 4] 合并结果")
    result = build_final(utterances, req_map)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print_summary(result, audio_path.name)
    print(f"\n✓ 保存至: {output_path}")


if __name__ == "__main__":
    main()
