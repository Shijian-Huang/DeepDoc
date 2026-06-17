import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont


VIDEO_DIR = Path(__file__).resolve().parent / "data" / "videos"
WORK_DIR = VIDEO_DIR / "work"
DEFAULT_PIPER_MODEL = Path(__file__).resolve().parent / "data" / "voices" / "en_US-lessac-medium.onnx"
DEFAULT_TTS_VOICE = os.getenv("DEEPDOC_TTS_VOICE", "Samantha")
DEFAULT_TTS_RATE = os.getenv("DEEPDOC_TTS_RATE", "150")
SCENE_PAUSE_SECONDS = float(os.getenv("DEEPDOC_SCENE_PAUSE_SECONDS", "1.05"))
TTS_PROVIDER = os.getenv("DEEPDOC_TTS_PROVIDER", "piper").strip().lower()
OPENAI_TTS_MODEL = os.getenv("DEEPDOC_OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.getenv("DEEPDOC_OPENAI_TTS_VOICE", "marin")
OPENAI_TTS_INSTRUCTIONS = os.getenv(
    "DEEPDOC_OPENAI_TTS_INSTRUCTIONS",
    (
        "Read like a calm, clear research explainer. Use a warm natural tone, "
        "moderate pace, and brief pauses between clauses. Avoid sounding like a robot."
    ),
)
PIPER_BIN = os.getenv("DEEPDOC_PIPER_BIN", "piper")
PIPER_MODEL = os.getenv("DEEPDOC_PIPER_MODEL", str(DEFAULT_PIPER_MODEL))
PIPER_LENGTH_SCALE = os.getenv("DEEPDOC_PIPER_LENGTH_SCALE", "1.08")
PIPER_NOISE_SCALE = os.getenv("DEEPDOC_PIPER_NOISE_SCALE", "")
PIPER_NOISE_W = os.getenv("DEEPDOC_PIPER_NOISE_W", "")

INK = "#17202A"
PAPER = "#F3F6F8"
PANEL = "#FFFFFF"
LINE = "#D8E0E7"
TEAL = "#1F7A6D"
BLUE = "#2F6F9F"
WARM = "#B85C38"
SOFT_TEAL = "#E7F2EF"
SOFT_BLUE = "#E8F1F7"
SOFT_WARM = "#F6EAE4"


class VideoGenerationError(RuntimeError):
    pass


def _ensure_tools() -> None:
    if TTS_PROVIDER == "say" and not shutil.which("say"):
        raise VideoGenerationError("macOS say command was not found.")
    if not shutil.which("ffmpeg"):
        raise VideoGenerationError(
            "ffmpeg was not found. Install ffmpeg, then retry video generation."
        )


def _safe_text(value: str) -> str:
    return value.replace("\x00", " ").strip()


def _load_font(size: int, bold: bool = False) -> Any:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_width(text: str, font: Any) -> int:
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0]


def _truncate_to_width(text: str, font: Any, max_width: int) -> str:
    cleaned = _safe_text(text)
    if _text_width(cleaned, font) <= max_width:
        return cleaned

    suffix = "..."
    while cleaned and _text_width(cleaned + suffix, font) > max_width:
        cleaned = cleaned[:-1]
    return cleaned.rstrip() + suffix


def _wrap_limited_lines(text: str, font: Any, max_width: int, max_lines: int) -> list[str]:
    lines = _wrap_lines(text, font, max_width)
    if len(lines) <= max_lines:
        return lines
    limited = lines[:max_lines]
    limited[-1] = _truncate_to_width(" ".join(lines[max_lines - 1:]), font, max_width)
    return limited


def _wrap_lines(text: str, font: Any, max_width: int) -> list[str]:
    words = _safe_text(text).split()
    lines: list[str] = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = font.getbbox(candidate)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word

    if current:
        lines.append(current)
    return lines or [""]


def _scene_bullets(scene: dict) -> list[str]:
    bullets = scene.get("bullets")
    if not isinstance(bullets, list):
        return []
    return [str(bullet) for bullet in bullets]


def _voiceover_text(scene: dict) -> str:
    voiceover = str(scene.get("voiceover") or "")
    if voiceover.strip():
        return voiceover

    heading = str(scene.get("heading") or "")
    return ". ".join([heading, *_scene_bullets(scene)]).strip()


def _estimate_duration_seconds(text: str) -> float:
    words = len(text.split())
    return max(6.0, min(22.0, words / 2.1 + 2.5))


def _say_command(audio_path: Path, text: str) -> list[str]:
    narration = _narration_text(text)
    command = ["say"]
    if DEFAULT_TTS_VOICE:
        command.extend(["-v", DEFAULT_TTS_VOICE])
    if DEFAULT_TTS_RATE:
        command.extend(["-r", DEFAULT_TTS_RATE])
    command.extend(["-o", str(audio_path), narration])
    return command


def _tts_audio_extension() -> str:
    if TTS_PROVIDER == "openai":
        return "mp3"
    if TTS_PROVIDER == "piper":
        return "wav"
    return "aiff"


def _generate_tts_audio(audio_path: Path, text: str) -> None:
    if TTS_PROVIDER == "openai":
        _generate_openai_tts(audio_path, text)
        return
    if TTS_PROVIDER == "piper":
        _generate_piper_tts(audio_path, text)
        return

    _run(_say_command(audio_path, text))


def _generate_piper_tts(audio_path: Path, text: str) -> None:
    piper_path = shutil.which(PIPER_BIN)
    if not piper_path:
        venv_piper = Path(sys.executable).parent / "piper"
        if venv_piper.exists():
            piper_path = str(venv_piper)
    if not piper_path:
        raise VideoGenerationError(
            "Piper was not found. Install it with `venv/bin/python -m pip install piper-tts`, "
            "or set DEEPDOC_TTS_PROVIDER=say."
        )
    if not PIPER_MODEL:
        raise VideoGenerationError(
            "DEEPDOC_PIPER_MODEL must point to a Piper .onnx voice model."
        )

    model_path = Path(PIPER_MODEL).expanduser()
    if not model_path.exists():
        raise VideoGenerationError(f"Piper model was not found: {model_path}")

    command = [
        piper_path,
        "--model", str(model_path),
        "--output_file", str(audio_path),
    ]
    if PIPER_LENGTH_SCALE:
        command.extend(["--length_scale", PIPER_LENGTH_SCALE])
    if PIPER_NOISE_SCALE:
        command.extend(["--noise_scale", PIPER_NOISE_SCALE])
    if PIPER_NOISE_W:
        command.extend(["--noise_w", PIPER_NOISE_W])

    result = subprocess.run(
        command,
        input=_narration_text(text),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "Piper TTS failed."
        raise VideoGenerationError(message)


def _generate_openai_tts(audio_path: Path, text: str) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise VideoGenerationError(
            "OPENAI_API_KEY is required when DEEPDOC_TTS_PROVIDER=openai."
        )

    payload = {
        "model": OPENAI_TTS_MODEL,
        "voice": OPENAI_TTS_VOICE,
        "input": _narration_text(text),
        "instructions": OPENAI_TTS_INSTRUCTIONS,
        "response_format": "mp3",
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/audio/speech",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            audio_path.write_bytes(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise VideoGenerationError(f"OpenAI TTS failed: {detail}") from error
    except urllib.error.URLError as error:
        raise VideoGenerationError(f"OpenAI TTS failed: {error.reason}") from error


def _narration_text(text: str) -> str:
    cleaned = _safe_text(text)
    cleaned = cleaned.replace("—", ", ")
    cleaned = cleaned.replace("–", ", ")
    cleaned = re.sub(r"\(([^)]{1,80})\)", r", \1,", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([.!?])\s+", r"\1  ", cleaned)
    return cleaned.strip()


def _probe_duration_seconds(path: Path) -> Optional[float]:
    if not shutil.which("ffprobe"):
        return None

    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def _draw_text_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    xy: tuple[int, int],
    font: Any,
    fill: str,
    line_gap: int,
) -> int:
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = font.getbbox(line or " ")
        y += bbox[3] - bbox[1] + line_gap
    return y


def _scene_kind(scene: dict, index: int, total: int) -> str:
    role = _safe_text(str(scene.get("role") or "")).lower()
    if role in {
        "hook", "surprising_finding", "core_insight", "why_it_matters",
        "mechanism", "technical_insight", "evidence", "risk", "takeaway",
    }:
        if role in {"hook", "surprising_finding", "evidence"}:
            return "finding"
        if role in {"core_insight", "why_it_matters", "mechanism", "technical_insight"}:
            return "method"
        if role == "risk":
            return "problem"
        return "takeaway"

    heading = _safe_text(scene.get("heading") or "").lower()
    bullets = " ".join(str(item).lower() for item in scene.get("bullets", []))
    text = f"{heading} {bullets}"
    tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text))

    def has_any(words: list[str]) -> bool:
        return any(word in tokens for word in words)

    def has_phrase(phrases: list[str]) -> bool:
        return any(phrase in text for phrase in phrases)

    if index == total or has_any(["future", "takeaway", "conclusion"]):
        return "takeaway"
    if has_any(["study", "participants", "participant", "measure", "measures", "performance", "comparison"]):
        return "study"
    if has_any(["result", "results", "finding", "findings", "gain", "gains", "completed", "advantage"]):
        return "finding"
    if has_any(["challenge", "challenges", "difficult", "manual", "searching", "switching", "navigate", "navigation"]) or has_phrase(["current workflow", "context switching"]):
        return "problem"
    if has_any([
        "method", "introducing", "gilt", "context", "integrated", "interface",
        "feature", "features", "api", "query", "queries", "usage", "explains",
        "explain", "support", "automated",
    ]) or has_phrase(["code segment", "domain-specific", "open-ended"]):
        return "method"
    return "problem"


def _kind_style(kind: str) -> tuple[str, str, str]:
    styles = {
        "problem": (WARM, SOFT_WARM, "Problem"),
        "method": (TEAL, SOFT_TEAL, "Method"),
        "study": (BLUE, SOFT_BLUE, "Study"),
        "finding": (WARM, SOFT_WARM, "Finding"),
        "takeaway": (TEAL, SOFT_TEAL, "Takeaway"),
    }
    return styles.get(kind, (TEAL, SOFT_TEAL, "Scene"))


def _draw_header(
    draw: ImageDraw.ImageDraw,
    index: int,
    total: int,
    kind: str,
    accent: str,
    label: str,
    video_title: str,
) -> None:
    brand_font = _load_font(19, bold=True)
    title_font = _load_font(22, bold=True)
    small_font = _load_font(17, bold=True)
    draw.rectangle((0, 0, 1280, 132), fill=INK)
    draw.text((72, 17), f"DeepDoc · Scene {index}/{total}", font=brand_font, fill="#DDE9EF")
    title_lines = _wrap_limited_lines(video_title or "Research explainer", title_font, 840, 2)
    y = 48
    for line in title_lines:
        draw.text((72, y), line, font=title_font, fill="#FFFFFF")
        bbox = title_font.getbbox(line or " ")
        y += bbox[3] - bbox[1] + 5

    pill_x0, pill_y0, pill_x1, pill_y1 = 990, 44, 1208, 84
    draw.rounded_rectangle((pill_x0, pill_y0, pill_x1, pill_y1), radius=18, fill=accent)
    label_text = label.upper()
    label_width = _text_width(label_text, small_font)
    draw.text((pill_x0 + (pill_x1 - pill_x0 - label_width) / 2, 55), label_text, font=small_font, fill="#FFFFFF")

    progress_width = int(1280 * index / total)
    draw.rectangle((0, 129, 1280, 134), fill="#263444")
    draw.rectangle((0, 129, progress_width, 134), fill=accent)


def _draw_title(draw: ImageDraw.ImageDraw, heading: str, x: int, y: int, max_width: int) -> int:
    title_font = _load_font(42, bold=True)
    return _draw_text_lines(
        draw,
        _wrap_lines(heading, title_font, max_width)[:3],
        (x, y),
        title_font,
        INK,
        12,
    )


def _draw_bullets(
    draw: ImageDraw.ImageDraw,
    bullets: list[str],
    x: int,
    y: int,
    max_width: int,
    accent: str,
    font_size: int = 30,
    limit: int = 4,
) -> int:
    bullet_font = _load_font(font_size)
    for bullet in bullets[:limit]:
        wrapped = _wrap_lines(str(bullet), bullet_font, max_width)
        if y > 540:
            break
        draw.ellipse((x, y + 11, x + 14, y + 25), fill=accent)
        y = _draw_text_lines(draw, wrapped, (x + 34, y), bullet_font, "#253240", 8) + 12
    return y


def _draw_subtitle(draw: ImageDraw.ImageDraw, voiceover: str, accent: str) -> None:
    subtitle = _safe_text(voiceover)
    if not subtitle:
        return

    x0, y0, x1, y1 = 72, 548, 1208, 704
    max_width = x1 - x0 - 88
    max_lines = 6
    line_gap = 5
    caption_font = _load_font(21)
    lines = _wrap_lines(subtitle, caption_font, max_width)

    for size in [21, 20, 19, 18, 17, 16, 15, 14]:
        candidate_font = _load_font(size)
        candidate_lines = _wrap_lines(subtitle, candidate_font, max_width)
        line_height = candidate_font.getbbox("Ag")[3] - candidate_font.getbbox("Ag")[1]
        total_height = min(len(candidate_lines), max_lines) * line_height + (min(len(candidate_lines), max_lines) - 1) * line_gap
        if len(candidate_lines) <= max_lines and total_height <= y1 - y0 - 32:
            caption_font = candidate_font
            lines = candidate_lines
            break

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = _truncate_to_width(lines[-1], caption_font, max_width)

    draw.rounded_rectangle((x0, y0, x1, y1), radius=12, fill="#FFFFFF", outline=LINE, width=2)
    draw.rectangle((x0, y0, x0 + 18, y1), fill=accent)
    _draw_text_lines(draw, lines, (x0 + 40, y0 + 22), caption_font, "#253240", line_gap)


def _write_template_slide_image(path: Path, scene: dict, index: int, total: int, video_title: str) -> None:
    kind = _scene_kind(scene, index, total)
    accent, soft, label = _kind_style(kind)
    image = Image.new("RGB", (1280, 720), PAPER)
    draw = ImageDraw.Draw(image)
    _draw_header(draw, index, total, kind, accent, label, video_title)

    heading = _safe_text(scene.get("heading") or "Untitled scene")
    bullets = _scene_bullets(scene)

    draw.rounded_rectangle((72, 160, 1208, 540), radius=18, fill=PANEL, outline=LINE, width=2)
    draw.rectangle((72, 160, 90, 540), fill=soft)
    y = _draw_title(draw, heading, 118, 198, 980) + 30
    _draw_bullets(draw, bullets, 124, y, 980, accent, font_size=31, limit=4)

    _draw_subtitle(draw, _voiceover_text(scene), accent)
    image.save(path)


def _write_ai_fallback_slide_image(path: Path, scene: dict, index: int, total: int, video_title: str) -> None:
    image = Image.new("RGB", (1280, 720), "#F7F3EA")
    draw = ImageDraw.Draw(image)
    accent = "#3949AB"
    _draw_header(draw, index, total, "scene", accent, "Scene", video_title)

    heading = _safe_text(scene.get("heading") or "Untitled scene")
    bullets = _scene_bullets(scene)
    draw.rounded_rectangle((72, 160, 1208, 540), radius=18, fill="#FFFFFF", outline=LINE, width=2)
    y = _draw_title(draw, heading, 116, 198, 990) + 28
    _draw_bullets(draw, bullets, 122, y, 940, accent, font_size=30, limit=5)
    _draw_subtitle(draw, _voiceover_text(scene), accent)
    image.save(path)


def _write_slide_image(
    path: Path,
    scene: dict,
    index: int,
    total: int,
    video_title: str,
) -> str:
    try:
        _write_template_slide_image(path, scene, index, total, video_title)
        return "text_only"
    except Exception:
        _write_ai_fallback_slide_image(path, scene, index, total, video_title)
        return "ai_fallback"


def _run(command: list[str], cwd: Optional[Path] = None) -> None:
    result = subprocess.run(command, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "Command failed."
        raise VideoGenerationError(message)


def generate_video_from_script(analysis_id: str, video_script: dict, display_title: Optional[str] = None) -> dict:
    _ensure_tools()

    scenes = video_script.get("scenes", [])
    if not isinstance(scenes, list) or not scenes:
        raise VideoGenerationError("No video script scenes were found.")

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = WORK_DIR / analysis_id
    work_dir.mkdir(parents=True, exist_ok=True)

    title = video_script.get("paper_title") or display_title or video_script.get("title") or "Research explainer"

    segment_paths: list[Path] = []
    visual_sources: list[dict] = []
    for index, scene in enumerate(scenes, start=1):
        voiceover = _voiceover_text(scene)
        audio_path = work_dir / f"scene_{index:02d}.{_tts_audio_extension()}"
        slide_path = work_dir / f"scene_{index:02d}.png"
        segment_path = work_dir / f"scene_{index:02d}.mp4"

        _generate_tts_audio(audio_path, voiceover)
        audio_duration = _probe_duration_seconds(audio_path)
        duration = (audio_duration or _estimate_duration_seconds(voiceover)) + SCENE_PAUSE_SECONDS
        visual_source = _write_slide_image(slide_path, scene, index, len(scenes), title)
        visual_sources.append({
            "scene": index,
            "source": visual_source,
            "page": None,
        })
        _run([
            "ffmpeg", "-y",
            "-loop", "1",
            "-t", f"{duration:.2f}",
            "-i", slide_path.name,
            "-i", audio_path.name,
            "-af", f"apad=pad_dur={SCENE_PAUSE_SECONDS:.2f}",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            segment_path.name,
        ], cwd=work_dir)
        segment_paths.append(segment_path)

    concat_path = work_dir / "concat.txt"
    concat_path.write_text(
        "".join(f"file '{path}'\n" for path in segment_paths),
        encoding="utf-8",
    )

    output_path = VIDEO_DIR / f"{analysis_id}.mp4"
    _run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_path),
        "-c", "copy",
        str(output_path),
    ])

    return {
        "video_path": str(output_path),
        "video_url": f"/analyses/{analysis_id}/video/download",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scene_count": len(segment_paths),
        "visual_sources": visual_sources,
        "tts_provider": TTS_PROVIDER,
        "tts_voice": OPENAI_TTS_VOICE if TTS_PROVIDER == "openai" else DEFAULT_TTS_VOICE,
        "tts_model": (
            OPENAI_TTS_MODEL
            if TTS_PROVIDER == "openai"
            else PIPER_MODEL
            if TTS_PROVIDER == "piper"
            else "macos-say"
        ),
        "tts_rate": None if TTS_PROVIDER in {"openai", "piper"} else DEFAULT_TTS_RATE,
        "tts_length_scale": PIPER_LENGTH_SCALE if TTS_PROVIDER == "piper" else None,
        "scene_pause_seconds": SCENE_PAUSE_SECONDS,
    }
