#!/usr/bin/env python3
"""Analyze a local video with Qwen multimodal and render a study webpage."""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 .* doesn't match a supported version!",
)

import requests

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.7-plus"
DEFAULT_API_KEY_ENV = "DASHSCOPE_API_KEY"

# Platform-aware default paths for ffmpeg/ffprobe
if os.name == "nt":
    _default_ffmpeg_candidates = [
        Path("C:/Users/lenovo/bin/ffmpeg/ffmpeg-8.1.2-essentials_build/bin/ffmpeg.exe"),
        Path("C:/Users/lenovo/bin/ffmpeg") / "ffmpeg-8.1.2-essentials_build" / "bin" / "ffmpeg.exe",
    ]
    _found_ffmpeg = next((p for p in _default_ffmpeg_candidates if p.exists()), None)
    DEFAULT_FFMPEG = _found_ffmpeg or Path("ffmpeg.exe")
    DEFAULT_FFPROBE = (_found_ffmpeg.parent / "ffprobe.exe") if _found_ffmpeg else Path("ffprobe.exe")
else:
    DEFAULT_FFMPEG = Path("/opt/homebrew/bin/ffmpeg")
    DEFAULT_FFPROBE = Path("/opt/homebrew/bin/ffprobe")
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{12,}"),
    re.compile(r"DashScope\s+[A-Za-z0-9._-]{12,}", re.I),
]


@dataclass
class Frame:
    frame_id: str
    timestamp: float
    time_label: str
    path: Path
    segment_index: int


def scrub_secret(text: str) -> str:
    value = text or ""
    for pattern in SECRET_PATTERNS:
        value = pattern.sub("[REDACTED]", value)
    return value


def ensure_tool(path: Path, fallback_name: str) -> Path:
    if path.exists() and os.access(path, os.X_OK):
        return path
    found = shutil.which(fallback_name)
    if found:
        return Path(found)
    raise RuntimeError(f"{fallback_name}-not-found")


def run_command(cmd: list[str], *, timeout: int = 3600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def seconds_to_label(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hour = seconds // 3600
    minute = (seconds % 3600) // 60
    second = seconds % 60
    if hour:
        return f"{hour:02d}:{minute:02d}:{second:02d}"
    return f"{minute:02d}:{second:02d}"


def ffprobe_info(video: Path) -> dict[str, Any]:
    ffprobe = ensure_tool(DEFAULT_FFPROBE, "ffprobe")
    cmd = [
        str(ffprobe),
        "-hide_banner",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size,format_name,bit_rate",
        "-show_entries",
        "stream=index,codec_type,codec_name,width,height,r_frame_rate,duration",
        "-of",
        "json",
        str(video),
    ]
    proc = run_command(cmd, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(scrub_secret(proc.stderr or proc.stdout))
    return json.loads(proc.stdout)


def duration_from_info(info: dict[str, Any]) -> float:
    try:
        return float(info.get("format", {}).get("duration") or 0)
    except (TypeError, ValueError):
        return 0.0


def make_out_dir(video: Path, out_dir: str | None) -> Path:
    if out_dir:
        target = Path(out_dir).expanduser()
    else:
        target = video.parent / "study-qwen"
    target.mkdir(parents=True, exist_ok=True)
    (target / "assets").mkdir(parents=True, exist_ok=True)
    return target


def extract_frames(
    video: Path,
    out_dir: Path,
    *,
    duration: float,
    frame_count: int,
) -> list[Frame]:
    ffmpeg = ensure_tool(DEFAULT_FFMPEG, "ffmpeg")
    assets_dir = out_dir / "assets"
    if duration <= 0:
        duration = 60.0
    frame_count = max(4, min(frame_count, 36))
    # For very short or low-fps videos, cap frame_count to avoid seeking past EOF
    if duration > 0:
        max_reasonable = max(4, int(duration // 0.5))
        frame_count = min(frame_count, max_reasonable)
    # Avoid exact first/last frames; interview videos often open and close with fades.
    timestamps = [
        max(0.0, min(duration - 0.5, duration * (index + 0.5) / frame_count))
        for index in range(frame_count)
    ]
    frames: list[Frame] = []
    for index, timestamp in enumerate(timestamps, start=1):
        target = assets_dir / f"frame-{index:03d}.jpg"
        cmd = [
            str(ffmpeg),
            "-y",
            "-i",
            str(video),
            "-ss",
            f"{timestamp:.3f}",
            "-frames:v",
            "1",
            "-pix_fmt",
            "yuvj420p",
            "-q:v",
            "3",
            str(target),
        ]
        proc = run_command(cmd, timeout=120)
        if proc.returncode != 0 or not target.exists():
            raise RuntimeError(scrub_secret(proc.stderr or proc.stdout))
        frames.append(
            Frame(
                frame_id=f"frame-{index:03d}",
                timestamp=timestamp,
                time_label=seconds_to_label(timestamp),
                path=target,
                segment_index=index - 1,
            )
        )
    storyboard = [
        {
            "frame_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "time_label": frame.time_label,
            "path": str(frame.path.relative_to(out_dir)),
            "segment_index": frame.segment_index,
        }
        for frame in frames
    ]
    (out_dir / "storyboard.json").write_text(
        json.dumps({"frames": storyboard}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return frames


def extract_audio(video: Path, out_dir: Path) -> Path:
    ffmpeg = ensure_tool(DEFAULT_FFMPEG, "ffmpeg")
    target = out_dir / "audio.wav"
    cmd = [
        str(ffmpeg),
        "-y",
        "-i",
        str(video),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(target),
    ]
    proc = run_command(cmd, timeout=600)
    if proc.returncode != 0 or not target.exists():
        raise RuntimeError(scrub_secret(proc.stderr or proc.stdout))
    return target


def transcribe_with_whisper(audio: Path, out_dir: Path, *, model_name: str, language: str) -> dict[str, Any]:
    try:
        import whisper  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "skipped",
            "reason": f"openai-whisper-unavailable:{exc}",
            "segments": [],
            "text": "",
        }
    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio), language=language)
    segments = [
        {
            "start": float(seg.get("start", 0)),
            "end": float(seg.get("end", 0)),
            "text": str(seg.get("text", "")).strip(),
        }
        for seg in result.get("segments", [])
    ]
    text = str(result.get("text", "")).strip()
    (out_dir / "transcript.txt").write_text(text + "\n", encoding="utf-8")
    (out_dir / "transcript.json").write_text(
        json.dumps({"segments": segments, "text": text}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"status": "ok", "segments": segments, "text": text}


def load_transcript(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            return {
                "status": "ok",
                "segments": data.get("segments") or [],
                "text": data.get("text") or text,
            }
    return {"status": "ok", "segments": [], "text": text.strip()}


def transcript_for_window(transcript: dict[str, Any], start: float, end: float, fallback_chars: int = 1400) -> str:
    segments = transcript.get("segments") or []
    selected = [
        seg.get("text", "").strip()
        for seg in segments
        if float(seg.get("end", 0) or 0) >= start and float(seg.get("start", 0) or 0) <= end
    ]
    joined = "\n".join(part for part in selected if part)
    if joined:
        return joined[:4000]
    text = str(transcript.get("text") or "")
    if not text:
        return ""
    # Plain text fallback: approximate by proportional position.
    duration = max(end, 1.0)
    ratio = max(0.0, min(1.0, start / duration))
    cursor = int(len(text) * ratio)
    return text[cursor : cursor + fallback_chars]


def image_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def extract_json_block(text: str) -> Any:
    raw = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S | re.I)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start_candidates = [pos for pos in [raw.find("{"), raw.find("[")] if pos >= 0]
        if not start_candidates:
            raise
        start = min(start_candidates)
        end = max(raw.rfind("}"), raw.rfind("]"))
        if end <= start:
            raise
        return json.loads(raw[start : end + 1])


def openai_chat_completion(
    *,
    base_url: str,
    model: str,
    api_key: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.2,
    timeout: int = 180,
) -> str:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    response = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
        },
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"api-error-{response.status_code}:{scrub_secret(response.text[:500])}")
    data = response.json()
    return data["choices"][0]["message"]["content"]


def build_segment_messages(
    *,
    video_title: str,
    segment_index: int,
    start: float,
    end: float,
    frames: list[Frame],
    transcript_text: str,
) -> list[dict[str, Any]]:
    frame_lines = "\n".join(
        f"- {frame.frame_id}: {frame.time_label}, image={frame.path.name}" for frame in frames
    )
    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": textwrap.dedent(
                f"""
                请分析这个视频片段，输出严格 JSON。

                视频标题：{video_title}
                片段编号：{segment_index + 1}
                时间范围：{seconds_to_label(start)} - {seconds_to_label(end)}

                可用画面帧：
                {frame_lines}

                该时间段转录：
                {transcript_text or "（没有可用转录，请根据画面谨慎判断，并把不确定内容写入待核验。）"}

                输出 JSON schema：
                {{
                  "segment_title": "这一段讲什么",
                  "segment_summary": "2-4 句摘要",
                  "key_points": [
                    {{
                      "title": "关键知识点标题",
                      "explanation": "具体内容，说明它为什么重要",
                      "evidence": "来自转录或画面的证据",
                      "frame_id": "必须从可用画面帧中选一个",
                      "confidence": "high|medium|low"
                    }}
                  ],
                  "timeline_item": {{
                    "time": "MM:SS",
                    "title": "时间线标题",
                    "summary": "一句话说明"
                  }},
                  "uncertain": ["需要进一步核验的信息"]
                }}
                """
            ).strip(),
        }
    ]
    for frame in frames:
        user_content.append({"type": "image_url", "image_url": {"url": image_to_data_url(frame.path)}})
    return [
        {
            "role": "system",
            "content": (
                "你是严谨的视频学习研究员。你的任务是把视频片段转成可复习的学习内容。"
                "所有关键知识点都要绑定到真实画面 frame_id。只输出 JSON。"
            ),
        },
        {"role": "user", "content": user_content},
    ]


def build_final_messages(
    *,
    video_title: str,
    duration: float,
    frame_ids: list[str],
    segment_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "你是学习网页总编辑。请把分段视频分析合成为结构化学习网页内容。"
                "每个关键知识点必须绑定一个存在的 frame_id。只输出 JSON。"
            ),
        },
        {
            "role": "user",
            "content": textwrap.dedent(
                f"""
                视频标题：{video_title}
                时长：{seconds_to_label(duration)}
                可用 frame_id：{", ".join(frame_ids)}

                分段分析 JSON：
                {json.dumps(segment_results, ensure_ascii=False)}

                请输出 JSON schema：
                {{
                  "page_title": "网页主标题",
                  "kicker": "短标签",
                  "subtitle": "一段话说明视频学习价值",
                  "meta": ["视频ID或来源", "时长", "核心人物/主题"],
                  "overview": {{
                    "heading": "30 秒总览标题",
                    "lead": "总览说明",
                    "cards": [{{"title": "卡片标题", "body": "具体内容", "accent": "blue|ochre|green|red"}}]
                  }},
                  "knowledge_points": [
                    {{
                      "title": "关键知识点",
                      "body": "具体学习内容，必须来自视频",
                      "why_it_matters": "为什么值得学",
                      "frame_id": "存在的 frame_id",
                      "confidence": "high|medium|low"
                    }}
                  ],
                  "timeline": [{{"time": "MM:SS", "title": "节点标题", "body": "节点说明", "frame_id": "存在的 frame_id"}}],
                  "matrix": [{{"title": "张力/框架", "body": "说明", "accent": "blue|ochre|green|red"}}],
                  "study_questions": [{{"title": "复盘问题", "body": "为什么问这个"}}],
                  "actions": ["看完后可以做的行动"],
                  "caveats": ["待核验或局限"]
                }}
                """
            ).strip(),
        },
    ]


def normalize_frame_id(value: str, frame_ids: list[str], fallback: str) -> str:
    if value in frame_ids:
        return value
    match = re.search(r"frame[-_ ]?(\d+)", value or "", re.I)
    if match:
        candidate = f"frame-{int(match.group(1)):03d}"
        if candidate in frame_ids:
            return candidate
    return fallback


def normalize_analysis(analysis: dict[str, Any], frames: list[Frame]) -> dict[str, Any]:
    frame_ids = [frame.frame_id for frame in frames]
    fallback = frame_ids[0] if frame_ids else ""
    fixes: list[dict[str, str]] = []
    for collection_name in ["knowledge_points", "timeline"]:
        for item in analysis.get(collection_name, []) or []:
            original = str(item.get("frame_id") or "")
            normalized = normalize_frame_id(original, frame_ids, fallback)
            if normalized != original:
                fixes.append({"where": collection_name, "from": original, "to": normalized})
            item["frame_id"] = normalized
    if fixes:
        analysis.setdefault("normalization_notes", []).extend(fixes)
    return analysis


def mock_analysis(video_title: str, frames: list[Frame], duration: float) -> dict[str, Any]:
    def frame(index: int) -> str:
        return frames[min(index, len(frames) - 1)].frame_id if frames else ""

    return {
        "page_title": f"{video_title}：完整学习总结",
        "kicker": "Qwen 多模态视频学习",
        "subtitle": "这份网页由脚本抽取关键帧并按学习结构组织，真实运行时会由 Qwen 结合转录与画面生成具体内容。",
        "meta": [f"时长：{seconds_to_label(duration)}", f"关键帧：{len(frames)} 张", "模式：mock-analysis"],
        "overview": {
            "heading": "30 秒总览",
            "lead": "这是模板预览，用于验证网页结构、媒体引用和知识点画面绑定。",
            "cards": [
                {"title": "先看全局", "body": "把视频拆成可复习的主题，而非只保留碎片截图。", "accent": "blue"},
                {"title": "再看证据", "body": "每个知识点绑定到视频画面，方便回到原片核对。", "accent": "green"},
                {"title": "最后行动", "body": "把观点转成复盘问题和行动清单。", "accent": "ochre"},
            ],
        },
        "knowledge_points": [
            {
                "title": "知识点需要画面证据",
                "body": "真正的学习网页要让读者看到观点来自哪一段视频。",
                "why_it_matters": "这能减少泛泛总结，提升可核验性。",
                "frame_id": frame(0),
                "confidence": "medium",
            },
            {
                "title": "转录和画面要交叉使用",
                "body": "转录给出语义，画面给出上下文和视觉证据。",
                "why_it_matters": "只看文字会漏掉人物、产品、图示和场景。",
                "frame_id": frame(1),
                "confidence": "medium",
            },
        ],
        "timeline": [
            {"time": "00:00", "title": "开场", "body": "建立主题和人物。", "frame_id": frame(0)},
            {"time": seconds_to_label(duration / 2), "title": "中段", "body": "展开核心观点。", "frame_id": frame(len(frames) // 2)},
        ],
        "matrix": [
            {"title": "能力 × 风险", "body": "越强的工具越需要验证边界。", "accent": "red"},
            {"title": "效率 × 学习", "body": "总结要服务复盘，而非只追求快。", "accent": "green"},
        ],
        "study_questions": [
            {"title": "这个观点来自哪里？", "body": "回到绑定画面和原视频时间点核验。"},
            {"title": "哪些内容还需要查证？", "body": "把低置信度内容列入 caveats。"},
        ],
        "actions": ["回看关键帧对应时间点", "把高价值知识点整理成自己的卡片"],
        "caveats": ["当前为 mock 模式，真实内容需要调用 Qwen 生成。"],
    }


def e(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def rel_path(path: Path, root: Path) -> str:
    return quote(str(path.relative_to(root)).replace("\\", "/"), safe="/")


def render_cards(cards: list[dict[str, Any]]) -> str:
    parts = []
    for card in cards:
        accent = card.get("accent") if card.get("accent") in {"blue", "ochre", "green", "red"} else "blue"
        parts.append(
            f"""
            <article class="card accent-{accent}">
              <h3>{e(card.get('title'))}</h3>
              <p>{e(card.get('body'))}</p>
            </article>
            """
        )
    return "\n".join(parts)


def render_html(
    *,
    analysis: dict[str, Any],
    frames: list[Frame],
    video: Path,
    out_dir: Path,
    info: dict[str, Any],
    transcript_status: str,
) -> Path:
    frame_map = {frame.frame_id: frame for frame in frames}
    poster = frames[0].path if frames else video
    video_src = quote(video.name)
    duration = duration_from_info(info)
    overview = analysis.get("overview") or {}
    knowledge_html = []
    for item in analysis.get("knowledge_points", []) or []:
        frame = frame_map.get(item.get("frame_id")) or (frames[0] if frames else None)
        img = rel_path(frame.path, out_dir) if frame else ""
        time_label = frame.time_label if frame else ""
        knowledge_html.append(
            f"""
            <article class="knowledge-card">
              <img src="{img}" alt="{e(item.get('title'))}">
              <div>
                <div class="time-chip">{e(time_label)} · {e(item.get('confidence', ''))}</div>
                <h3>{e(item.get('title'))}</h3>
                <p>{e(item.get('body'))}</p>
                <p class="why"><strong>为什么重要：</strong>{e(item.get('why_it_matters'))}</p>
              </div>
            </article>
            """
        )

    timeline_html = []
    for item in analysis.get("timeline", []) or []:
        frame = frame_map.get(item.get("frame_id")) or (frames[0] if frames else None)
        img = rel_path(frame.path, out_dir) if frame else ""
        timeline_html.append(
            f"""
            <article class="moment rich-moment">
              <div class="time">{e(item.get('time'))}</div>
              <img src="{img}" alt="{e(item.get('title'))}">
              <div>
                <h3>{e(item.get('title'))}</h3>
                <p>{e(item.get('body'))}</p>
              </div>
            </article>
            """
        )

    matrix_html = render_cards(analysis.get("matrix") or [])
    questions_html = "\n".join(
        f"""
        <article class="card">
          <h3>{index}. {e(item.get('title'))}</h3>
          <p>{e(item.get('body'))}</p>
        </article>
        """
        for index, item in enumerate(analysis.get("study_questions") or [], start=1)
    )
    actions_html = "\n".join(f"<li>{e(item)}</li>" for item in analysis.get("actions") or [])
    caveats_html = "\n".join(f"<li>{e(item)}</li>" for item in analysis.get("caveats") or [])
    meta = analysis.get("meta") or []
    meta_html = "\n".join(f'<span class="pill">{e(item)}</span>' for item in meta)
    if not meta_html:
        meta_html = f'<span class="pill">时长：{e(seconds_to_label(duration))}</span>'

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{e(analysis.get('page_title') or video.stem)}</title>
  <style>
    :root {{
      --ink: #171511;
      --muted: #6a645c;
      --paper: #f7f2ea;
      --panel: #fffaf2;
      --line: #d8cab6;
      --coal: #23201b;
      --ochre: #c4862f;
      --blue: #2f6f8f;
      --green: #607d3b;
      --red: #a44f42;
      --shadow: 0 18px 50px rgba(34, 30, 24, .12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        linear-gradient(90deg, rgba(23, 21, 17, .035) 1px, transparent 1px),
        linear-gradient(180deg, rgba(23, 21, 17, .03) 1px, transparent 1px),
        var(--paper);
      background-size: 48px 48px;
      font-family: "Songti SC", "STSong", "Noto Serif CJK SC", Georgia, serif;
      line-height: 1.7;
    }}
    a {{ color: inherit; }}
    .page {{ width: min(1180px, calc(100vw - 36px)); margin: 0 auto; }}
    .hero {{
      min-height: 88vh;
      display: grid;
      grid-template-columns: minmax(0, 1.02fr) minmax(320px, .98fr);
      gap: 42px;
      align-items: center;
      padding: 42px 0 28px;
    }}
    .kicker {{
      display: inline-flex;
      font: 700 12px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
      letter-spacing: .08em;
      text-transform: uppercase;
      color: var(--blue);
      border-bottom: 2px solid var(--blue);
      padding-bottom: 8px;
      margin-bottom: 24px;
    }}
    h1 {{ margin: 0; font-size: clamp(42px, 6.4vw, 88px); line-height: .92; letter-spacing: 0; }}
    h2 {{ margin: 0; font-size: clamp(28px, 4vw, 54px); line-height: 1; letter-spacing: 0; }}
    .subtitle {{ max-width: 760px; margin: 28px 0 0; color: var(--muted); font-size: clamp(18px, 2vw, 23px); }}
    .meta-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 28px; }}
    .pill, .time-chip {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(255, 250, 242, .78);
      font: 700 13px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
      color: var(--coal);
    }}
    .hero-media {{ padding: 16px; background: var(--panel); border: 1px solid var(--line); box-shadow: var(--shadow); transform: rotate(1.2deg); }}
    .hero-media img, .hero-media video {{ width: 100%; aspect-ratio: 16 / 9; display: block; object-fit: cover; background: #080806; }}
    .hero-media video {{ margin-top: 12px; }}
    .caption {{ margin-top: 12px; color: var(--muted); font-size: 13px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .band {{ padding: 72px 0; border-top: 1px solid rgba(216, 202, 182, .75); }}
    .band-title {{ display: grid; grid-template-columns: 220px 1fr; gap: 28px; align-items: start; margin-bottom: 28px; }}
    .section-number {{ color: var(--ochre); font: 800 14px/1 ui-monospace, SFMono-Regular, Menlo, monospace; text-transform: uppercase; }}
    .lead {{ margin: 16px 0 0; color: var(--muted); font-size: 18px; max-width: 850px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }}
    .matrix {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    .card {{ background: rgba(255, 250, 242, .88); border: 1px solid var(--line); border-radius: 8px; padding: 22px; box-shadow: 0 10px 30px rgba(34, 30, 24, .06); }}
    .card h3, .knowledge-card h3, .moment h3 {{ margin: 0 0 12px; font-size: 22px; line-height: 1.15; }}
    .card p, .card li, .knowledge-card p, .moment p {{ color: var(--muted); margin: 0; font-size: 15px; }}
    .accent-blue {{ border-top: 5px solid var(--blue); }}
    .accent-ochre {{ border-top: 5px solid var(--ochre); }}
    .accent-green {{ border-top: 5px solid var(--green); }}
    .accent-red {{ border-top: 5px solid var(--red); }}
    .knowledge-grid {{ display: grid; gap: 16px; }}
    .knowledge-card {{
      display: grid;
      grid-template-columns: minmax(220px, .42fr) 1fr;
      gap: 18px;
      padding: 16px;
      background: rgba(255, 250, 242, .9);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 10px 30px rgba(34, 30, 24, .06);
    }}
    .knowledge-card img, .rich-moment img {{ width: 100%; aspect-ratio: 16 / 9; object-fit: cover; border: 1px solid var(--line); }}
    .why {{ margin-top: 10px !important; color: var(--coal) !important; }}
    .timeline {{ display: grid; gap: 14px; }}
    .moment {{ display: grid; grid-template-columns: 96px 220px 1fr; gap: 18px; align-items: start; padding: 18px 0; border-bottom: 1px solid rgba(216, 202, 182, .82); }}
    .time {{ color: var(--blue); font: 800 18px/1 ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .note-list {{ columns: 2; color: var(--muted); }}
    .footer {{ padding: 36px 0 64px; color: var(--muted); font-size: 14px; }}
    @media (max-width: 880px) {{
      .hero, .band-title, .grid, .matrix, .knowledge-card, .moment {{ grid-template-columns: 1fr; }}
      .hero {{ min-height: auto; gap: 26px; }}
      .hero-media {{ transform: none; }}
      .note-list {{ columns: 1; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero" aria-labelledby="page-title">
      <div>
        <div class="kicker">{e(analysis.get('kicker') or 'Qwen 多模态视频学习')}</div>
        <h1 id="page-title">{e(analysis.get('page_title') or video.stem)}</h1>
        <p class="subtitle">{e(analysis.get('subtitle'))}</p>
        <div class="meta-row">{meta_html}<span class="pill">转录：{e(transcript_status)}</span></div>
      </div>
      <figure class="hero-media">
        <img src="{rel_path(poster, out_dir)}" alt="视频关键画面">
        <video controls preload="metadata" poster="{rel_path(poster, out_dir)}">
          <source src="{video_src}" type="video/mp4">
        </video>
        <figcaption class="caption">本地视频与 Qwen 多模态抽帧学习结果。每个关键知识点都绑定到具体画面。</figcaption>
      </figure>
    </section>

    <section class="band" aria-labelledby="overview-title">
      <div class="band-title">
        <div class="section-number">01 · 30 秒总览</div>
        <div>
          <h2 id="overview-title">{e(overview.get('heading') or '30 秒总览')}</h2>
          <p class="lead">{e(overview.get('lead'))}</p>
        </div>
      </div>
      <div class="grid">{render_cards(overview.get('cards') or [])}</div>
    </section>

    <section class="band" aria-labelledby="knowledge-title">
      <div class="band-title">
        <div class="section-number">02 · 知识点 × 画面</div>
        <div>
          <h2 id="knowledge-title">关键知识点匹配正确画面</h2>
          <p class="lead">这一段是网页的核心：每条学习内容都能回到视频中的某个画面，方便复查上下文。</p>
        </div>
      </div>
      <div class="knowledge-grid">{''.join(knowledge_html)}</div>
    </section>

    <section class="band" aria-labelledby="timeline-title">
      <div class="band-title">
        <div class="section-number">03 · 时间线</div>
        <div>
          <h2 id="timeline-title">按视频结构复盘</h2>
          <p class="lead">时间线把知识点放回原片顺序，避免只剩结论。</p>
        </div>
      </div>
      <div class="timeline">{''.join(timeline_html)}</div>
    </section>

    <section class="band" aria-labelledby="matrix-title">
      <div class="band-title">
        <div class="section-number">04 · 框架</div>
        <div>
          <h2 id="matrix-title">风险、机会与张力</h2>
          <p class="lead">把视频里的观点提炼成可迁移的分析框架。</p>
        </div>
      </div>
      <div class="matrix">{matrix_html}</div>
    </section>

    <section class="band" aria-labelledby="questions-title">
      <div class="band-title">
        <div class="section-number">05 · 复盘</div>
        <div>
          <h2 id="questions-title">复盘问题与行动清单</h2>
          <p class="lead">用问题把视频内容转成自己的理解。</p>
        </div>
      </div>
      <div class="grid">{questions_html}</div>
      <article class="card" style="margin-top:16px">
        <h3>看完后可以做什么</h3>
        <ul>{actions_html}</ul>
      </article>
      <article class="card" style="margin-top:16px">
        <h3>局限与待核验</h3>
        <ul>{caveats_html}</ul>
      </article>
    </section>

    <footer class="footer">
      Generated on {e(dt.datetime.now().isoformat(timespec='seconds'))}. Report: qwen-analysis.json / storyboard.json / run-report.json
    </footer>
  </main>
</body>
</html>
"""
    target = out_dir / "study-summary-qwen.html"
    target.write_text(html_text, encoding="utf-8")
    return target


def validate_html_media(html_path: Path, root: Path) -> list[str]:
    from html.parser import HTMLParser
    from urllib.parse import unquote

    class Parser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.refs: list[str] = []

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            data = dict(attrs)
            for attr in ["src", "poster"]:
                ref = data.get(attr)
                if ref and not ref.startswith(("http://", "https://", "data:")):
                    self.refs.append(ref)

    parser = Parser()
    parser.feed(html_path.read_text(encoding="utf-8"))
    missing = []
    for ref in parser.refs:
        candidate = root / unquote(ref)
        if not candidate.exists():
            missing.append(ref)
    return missing


def assert_no_secret_in_outputs(out_dir: Path) -> list[str]:
    leaked: list[str] = []
    for path in out_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".txt", ".html", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                leaked.append(str(path))
                break
    return leaked


def segment_frames(frames: list[Frame], duration: float) -> list[tuple[float, float, list[Frame]]]:
    if not frames:
        return []
    boundaries = []
    for index, frame in enumerate(frames):
        start = 0.0 if index == 0 else (frames[index - 1].timestamp + frame.timestamp) / 2
        end = duration if index == len(frames) - 1 else (frame.timestamp + frames[index + 1].timestamp) / 2
        boundaries.append((start, end, [frame]))
    return boundaries


def analyze_with_qwen(
    *,
    api_key: str,
    base_url: str,
    model: str,
    video_title: str,
    duration: float,
    frames: list[Frame],
    transcript: dict[str, Any],
    max_segments: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    segment_results: list[dict[str, Any]] = []
    segments = segment_frames(frames, duration)[:max_segments]
    for index, (start, end, seg_frames) in enumerate(segments):
        text = transcript_for_window(transcript, start, end)
        messages = build_segment_messages(
            video_title=video_title,
            segment_index=index,
            start=start,
            end=end,
            frames=seg_frames,
            transcript_text=text,
        )
        raw = openai_chat_completion(
            base_url=base_url,
            model=model,
            api_key=api_key,
            messages=messages,
        )
        segment_results.append(extract_json_block(raw))
        time.sleep(0.2)
    final_raw = openai_chat_completion(
        base_url=base_url,
        model=model,
        api_key=api_key,
        messages=build_final_messages(
            video_title=video_title,
            duration=duration,
            frame_ids=[frame.frame_id for frame in frames],
            segment_results=segment_results,
        ),
    )
    final = extract_json_block(final_raw)
    if not isinstance(final, dict):
        raise RuntimeError("final-analysis-not-object")
    return segment_results, final


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Qwen multimodal video study webpage.")
    parser.add_argument("--video", required=True, help="Local MP4 path")
    parser.add_argument("--title", default="", help="Study title")
    parser.add_argument("--out-dir", default="", help="Output directory")
    parser.add_argument("--transcript", default="", help="Existing transcript .txt or .json")
    parser.add_argument("--frame-count", type=int, default=12, help="Number of key frames to extract")
    parser.add_argument("--max-segments", type=int, default=12, help="Max multimodal segments sent to Qwen")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Qwen multimodal model name")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV, help="Environment variable containing API key")
    parser.add_argument("--whisper-model", default="base", help="Local Whisper model for transcript")
    parser.add_argument("--language", default="zh", help="Whisper language code")
    parser.add_argument("--skip-transcribe", action="store_true", help="Skip local transcription")
    parser.add_argument("--mock-analysis", action="store_true", help="Render with mock analysis without API call")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    video = Path(args.video).expanduser().resolve()
    if not video.exists():
        print(f"video-not-found:{video}", file=sys.stderr)
        return 1
    out_dir = make_out_dir(video, args.out_dir or None)
    info = ffprobe_info(video)
    duration = duration_from_info(info)
    video_title = args.title or video.stem

    # Copy video into output folder if needed, so HTML can use a local relative path.
    local_video = out_dir / video.name
    if video.resolve() != local_video.resolve():
        if not local_video.exists() or local_video.stat().st_size != video.stat().st_size:
            shutil.copy2(video, local_video)
    else:
        local_video = video

    frames = extract_frames(local_video, out_dir, duration=duration, frame_count=args.frame_count)

    transcript: dict[str, Any]
    if args.transcript:
        transcript = load_transcript(Path(args.transcript).expanduser())
        transcript_status = "existing"
        (out_dir / "transcript.txt").write_text(str(transcript.get("text") or ""), encoding="utf-8")
    elif args.skip_transcribe:
        transcript = {"status": "skipped", "segments": [], "text": ""}
        transcript_status = "skipped"
    else:
        try:
            audio = extract_audio(local_video, out_dir)
            transcript = transcribe_with_whisper(audio, out_dir, model_name=args.whisper_model, language=args.language)
            transcript_status = str(transcript.get("status") or "unknown")
        except Exception as exc:  # noqa: BLE001
            transcript = {"status": "failed", "reason": scrub_secret(str(exc)), "segments": [], "text": ""}
            transcript_status = "failed"

    segment_results: list[dict[str, Any]]
    if args.mock_analysis:
        segment_results = []
        analysis = mock_analysis(video_title, frames, duration)
    else:
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            print(f"missing-api-key-env:{args.api_key_env}", file=sys.stderr)
            return 2
        try:
            segment_results, analysis = analyze_with_qwen(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                video_title=video_title,
                duration=duration,
                frames=frames,
                transcript=transcript,
                max_segments=args.max_segments,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"qwen-analysis-failed:{scrub_secret(str(exc))}", file=sys.stderr)
            return 3

    analysis = normalize_analysis(analysis, frames)
    (out_dir / "qwen-segments.json").write_text(
        json.dumps({"segments": segment_results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "qwen-analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    html_path = render_html(
        analysis=analysis,
        frames=frames,
        video=local_video,
        out_dir=out_dir,
        info=info,
        transcript_status=transcript_status,
    )
    missing_media = validate_html_media(html_path, out_dir)
    leaked = assert_no_secret_in_outputs(out_dir)
    report = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "video": str(local_video),
        "html": str(html_path),
        "model": args.model,
        "base_url": args.base_url,
        "duration": duration,
        "frames": len(frames),
        "transcript_status": transcript_status,
        "missing_media": missing_media,
        "secret_leaks": leaked,
    }
    (out_dir / "run-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if missing_media:
        print(f"missing-media:{missing_media}", file=sys.stderr)
        return 4
    if leaked:
        print(f"secret-leak-detected:{leaked}", file=sys.stderr)
        return 5
    print(html_path)
    print(f"frames={len(frames)}")
    print(f"transcript={transcript_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

