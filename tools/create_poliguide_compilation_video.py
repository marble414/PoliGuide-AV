#!/usr/bin/env python3
"""Create a submission-ready silent compilation video from PoliGuide demos."""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "processed_demo_videos"
OUTPUT = DEMO_DIR / "poliguide_submission_compilation_1080p.mp4"
PLAN_PATH = DEMO_DIR / "compilation_plan.json"
CONTACT_SHEET = DEMO_DIR / "compilation_review_contact_sheet.jpg"
SRT_PATH = DEMO_DIR / "poliguide_submission_compilation_1080p.srt"

W, H = 1920, 1080
LEFT_W = 1128
PANEL_X = LEFT_W
PANEL_W = W - LEFT_W
FPS = 30
CRF_TARGETS = [18, 20, 22, 24, 26, 28, 30, 32]
MAX_BYTES = 100 * 1024 * 1024

FONT_CANDIDATES = [
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
]


@dataclass(frozen=True)
class Clip:
    kind: str
    title: str
    subtitle: str
    detail: str
    source: str | None = None
    start: float = 0.0
    end: float = 0.0
    focus: str = ""
    metric: str = ""
    display_duration: float = 0.0


def get_font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if not path.exists():
            continue
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            pass
    return ImageFont.load_default()


F_TITLE = get_font(50)
F_H1 = get_font(42)
F_H2 = get_font(34)
F_BODY = get_font(28)
F_SMALL = get_font(23)
F_TINY = get_font(18)
F_CAPTION_TITLE = get_font(30)
F_CAPTION = get_font(31)


def read_manifest() -> dict:
    return json.loads((DEMO_DIR / "manifest.json").read_text(encoding="utf-8"))


def video_for(prefix: str) -> Path:
    matches = sorted(DEMO_DIR.glob(f"{prefix}*_poliguide_visual_demo.mp4"))
    if not matches:
        raise FileNotFoundError(prefix)
    return matches[0]


def jsonl_for(video: Path) -> Path:
    return video.with_suffix(".jsonl")


TOKEN_RE = re.compile(r"[A-Za-z0-9_./:+%-]+|\s+|.", re.UNICODE)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for token in TOKEN_RE.findall(paragraph):
            if token.isspace():
                token = " "
            trial = current + token
            if draw.textbbox((0, 0), trial, font=font)[2] <= width:
                current = trial
            else:
                if current:
                    lines.append(current.rstrip())
                if draw.textbbox((0, 0), token, font=font)[2] <= width:
                    current = token.lstrip()
                else:
                    # Fallback for a truly overlong token. This should be rare,
                    # but keeps compact panels from clipping if a path-like token appears.
                    current = ""
                    for char in token:
                        trial_char = current + char
                        if draw.textbbox((0, 0), trial_char, font=font)[2] <= width:
                            current = trial_char
                        else:
                            if current:
                                lines.append(current)
                            current = char
        if current:
            lines.append(current.rstrip())
    return lines


def draw_text_block(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    width: int,
    line_gap: int = 8,
) -> int:
    x, y = xy
    for line in wrap_text(draw, text, font, width):
        draw.text((x, y), line, font=font, fill=fill)
        y += draw.textbbox((0, 0), line, font=font)[3] + line_gap
    return y


def draw_text_with_shadow(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    shadow: tuple[int, int, int, int] = (0, 0, 0, 170),
) -> None:
    x, y = xy
    draw.text((x + 2, y + 2), text, font=font, fill=shadow)
    draw.text((x, y), text, font=font, fill=fill)


def rounded_rect(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], fill, outline=None, radius: int = 10, width: int = 1) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def records_for(prefix: str, start: float | None = None, end: float | None = None) -> list[dict]:
    path = jsonl_for(video_for(prefix))
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        timestamp = float(record.get("timestamp_sec", 0.0))
        if start is not None and timestamp < start:
            continue
        if end is not None and timestamp > end:
            continue
        records.append(record)
    return records


def top_commands(prefix: str, start: float | None = None, end: float | None = None) -> str:
    counts: dict[str, int] = {}
    for record in records_for(prefix, start, end):
        command = record.get("command", "NO_COMMAND")
        if command != "NO_COMMAND":
            counts[command] = counts.get(command, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:3]
    return " / ".join(f"{k}:{v}" for k, v in ranked) if ranked else "NO_COMMAND"


def avg_stats(prefix: str, start: float | None = None, end: float | None = None) -> tuple[float, float, float]:
    pose_values = []
    latency_values = []
    aligned = 0
    total = 0
    for record in records_for(prefix, start, end):
        pose_values.append(float(record.get("pose_quality", 0.0)))
        latency_values.append(float(record.get("latency_ms", 0.0)))
        gesture = record.get("gesture")
        command = record.get("command")
        if command and command not in {"NO_COMMAND", "KEEP_WAIT"}:
            total += 1
            if gesture == command and not record.get("safe_fallback"):
                aligned += 1
    pose = float(np.mean(pose_values)) if pose_values else 0.0
    latency = float(np.mean(latency_values)) if latency_values else 0.0
    alignment = float(aligned / total) if total else 0.0
    return pose, latency, alignment


def make_plan() -> list[Clip]:
    return [
        Clip(
            kind="title",
            title="警势智行 PoliGuide-AV",
            subtitle="真实道路场景下的交警手势识别与车辆指令解析",
            detail="本片汇总夜间眩光、施工入口、路侧遮挡、逆光和正面指挥等实拍场景，画面中的主体框、COCO17 骨架、动作标签、车辆指令和 JSONL 记录来自同一处理链路。",
            end=4.0,
        ),
        Clip("video", "夜间右向指挥", "夜间强光下，手臂指向画面右侧，系统同步输出 TURN_RIGHT。", "画面动作、顶部 gesture 标签和 command 输出保持同步，体现低照度场景下的稳定识别能力。", "demo_01_night_high_glare", 10.9, 14.1, "TURN_RIGHT", "", 5.0),
        Clip("video", "夜间示停指挥", "示停动作清晰出现时，系统同步输出 STOP 指令。", "横臂示停姿态与 STOP 标签一一对应，右侧面板同步给出姿态质量、延迟和命令统计。", "demo_01_night_high_glare", 20.1, 20.9, "STOP", "", 4.2),
        Clip("video", "施工入口右向指挥", "施工围挡背景下，手臂指向画面右侧，系统同步输出 TURN_RIGHT。", "主体跟踪框锁定指挥人员，骨架关键点稳定，动作标签和车辆命令同步变化。", "demo_02_construction_entry", 15.0, 18.6, "TURN_RIGHT", "", 5.0),
        Clip("video", "施工入口左向指挥", "同一施工入口场景中，手臂指向画面左侧，系统同步输出 TURN_LEFT。", "侧身姿态下仍能保持手臂方向、gesture 和 command 的连续对应关系。", "demo_02_construction_entry", 22.55, 24.6, "TURN_LEFT", "", 4.5),
        Clip("video", "路侧车辆右向指挥", "近车身遮挡场景中，手臂指向画面右侧，系统同步输出 TURN_RIGHT。", "车辆前机盖遮挡部分下肢时，系统仍保持上肢关键点和方向指令一致。", "demo_03_roadside_vehicle", 11.5, 13.6, "TURN_RIGHT", "", 4.5),
        Clip("video", "路侧车辆左向指挥", "路侧车辆遮挡下，手臂指向画面左侧，系统同步输出 TURN_LEFT。", "连续多帧中，主体框、骨架姿态、动作标签和车辆指令保持一致。", "demo_03_roadside_vehicle", 15.5, 22.5, "TURN_LEFT", "", 7.5),
        Clip("video", "夜间限速右向指挥", "夜间道路标志和眩光背景下，右向指挥动作稳定输出 TURN_RIGHT。", "强光、限速牌和暗背景共存时，系统仍能维持清晰的人体姿态与方向指令。", "demo_04_night_speed_limit", 4.2, 7.2, "TURN_RIGHT", "", 5.0),
        Clip("video", "夜间限速左向指挥", "同一夜间路段中，左向指挥动作稳定输出 TURN_LEFT。", "侧身姿态、手臂方向和命令输出一致，便于直观看到方向类指挥能力。", "demo_04_night_speed_limit", 13.0, 16.6, "TURN_LEFT", "", 5.0),
        Clip("video", "逆光左向指挥", "逆光侧身条件下，左向指挥动作与 TURN_LEFT 保持同步。", "人物处于强逆光阴影区域，系统仍保持主体选择、关键点和命令输出一致。", "demo_05_backlight_side", 30.9, 32.3, "TURN_LEFT", "", 4.2),
        Clip("video", "逆光慢行指挥", "慢行动作出现时，系统同步输出 SLOW_DOWN 指令。", "该场景用于展示非转向类指挥语义，动作、gesture 和 command 在短时窗口内一致。", "demo_05_backlight_side", 23.75, 24.45, "SLOW_DOWN", "", 4.2),
        Clip("video", "施工区域示停", "施工路段人工接管时，示停动作稳定对应 STOP。", "STOP 作为安全优先级最高的车辆命令，保持与示停姿态同步输出。", "demo_06_construction_stop", 23.35, 24.75, "STOP", "", 4.2),
        Clip("video", "正面右向指挥", "正面视角下，右向指挥动作与 TURN_RIGHT 输出保持一致。", "正面视角中保留 frame、track、gesture、command、confidence 和 latency 等逐帧字段。", "demo_07_front_direction", 3.65, 4.3, "TURN_RIGHT", "", 4.0),
        Clip(
            kind="title",
            title="结构化交付结果",
            subtitle="画面动作、检测标签、车辆指令与逐帧 JSONL 可共同复核",
            detail="视频用于直观展示，JSONL 用于工程追溯，命令语义可继续对接 JSON、ROS2、Autoware 与 CARLA 等平台。",
            end=4.0,
        ),
    ]


def make_base_canvas() -> Image.Image:
    canvas = Image.new("RGB", (W, H), (8, 13, 20))
    draw = ImageDraw.Draw(canvas)
    for x in range(0, W, 80):
        color = (18, 28, 38) if x % 160 == 0 else (12, 20, 28)
        draw.line((x, 0, x, H), fill=color, width=1)
    for y in range(0, H, 80):
        color = (18, 28, 38) if y % 160 == 0 else (12, 20, 28)
        draw.line((0, y, W, y), fill=color, width=1)
    return canvas


def draw_title_card(clip: Clip, frame_idx: int, total_frames: int) -> np.ndarray:
    img = make_base_canvas()
    draw = ImageDraw.Draw(img)
    progress = frame_idx / max(1, total_frames - 1)
    accent = (118, 244, 167)
    draw.rectangle((0, 0, W, 16), fill=accent)
    draw.text((96, 176), "PoliGuide-AV", font=F_TITLE, fill=accent)
    y = draw_text_block(draw, (96, 265), clip.title, F_TITLE, (244, 250, 255), 1260, 12)
    y = draw_text_block(draw, (96, y + 16), clip.subtitle, F_H2, (206, 221, 232), 1320, 12)
    y = draw_text_block(draw, (96, y + 34), clip.detail, F_BODY, (168, 186, 198), 1420, 10)
    rounded_rect(draw, (96, 810, 1824, 930), (12, 25, 32), (64, 89, 101), 12)
    draw_text_block(draw, (132, 844), "真实实拍输入 | 姿态关键点 | 时序状态机 | 标准化车辆语义 | 无声字幕版", F_BODY, (229, 239, 245), 1600)
    draw.rectangle((96, 984, 96 + int(1728 * progress), 996), fill=accent)
    return np.asarray(img)


def draw_panel(draw: ImageDraw.ImageDraw, clip: Clip, local_t: float, duration: float, clip_no: int, clip_total: int) -> None:
    draw.rectangle((PANEL_X, 0, W, H), fill=(10, 17, 24))
    draw.rectangle((PANEL_X, 0, W, 16), fill=(118, 244, 167))
    x = PANEL_X + 54
    y = 58
    draw.text((x, y), f"实拍演示 {clip_no:02d}/{clip_total:02d}", font=F_SMALL, fill=(118, 244, 167))
    y += 48
    y = draw_text_block(draw, (x, y), clip.title, F_H1, (245, 249, 252), PANEL_W - 96, 8)
    y += 20
    y = draw_text_block(draw, (x, y), clip.subtitle, F_BODY, (214, 226, 235), PANEL_W - 96, 9)
    y += 28
    rounded_rect(draw, (x, y, W - 54, y + 246), (17, 31, 40), (55, 78, 89), 10)
    metric_lines = [
        f"目标命令: {clip.focus}",
        f"输出窗口: {duration:.1f} 秒",
        *clip.metric.split("\n"),
    ]
    yy = y + 22
    for line in metric_lines:
        draw.text((x + 24, yy), line, font=F_TINY if line.startswith("主要命令") else F_SMALL, fill=(219, 232, 238))
        yy += 31
    y += 286
    y = draw_text_block(draw, (x, y), clip.detail, F_BODY, (177, 195, 207), PANEL_W - 96, 9)
    y += 24
    rounded_rect(draw, (x, 812, W - 54, 952), (13, 26, 34), (52, 74, 86), 10)
    draw.text((x + 24, 840), "输出契约", font=F_SMALL, fill=(118, 244, 167))
    draw_text_block(draw, (x + 24, 884), "frame / track / gesture / command / confidence / fallback / latency / COCO17 keypoints", F_TINY, (185, 202, 213), PANEL_W - 144, 6)
    progress = min(1.0, max(0.0, local_t / max(0.1, duration)))
    draw.rectangle((x, 1004, W - 54, 1018), fill=(41, 55, 64))
    draw.rectangle((x, 1004, x + int((PANEL_W - 108) * progress), 1018), fill=(118, 244, 167))


def add_lower_caption(draw: ImageDraw.ImageDraw, text: str) -> None:
    y0 = 890
    rounded_rect(draw, (34, y0, LEFT_W - 34, y0 + 150), (4, 9, 12, 206), (118, 244, 167, 110), 12, 1)
    title, _, body = text.partition("：")
    if body:
        draw_text_with_shadow(draw, (62, y0 + 24), title, F_CAPTION_TITLE, (118, 244, 167))
        draw_text_block(draw, (62, y0 + 70), body, F_CAPTION, (248, 252, 255), LEFT_W - 124, 8)
    else:
        draw_text_block(draw, (62, y0 + 35), text, F_CAPTION, (248, 252, 255), LEFT_W - 124, 8)


def resize_source(frame: np.ndarray) -> Image.Image:
    h, w = frame.shape[:2]
    if (w, h) == (LEFT_W, H):
        out = frame
    else:
        scale = min(LEFT_W / w, H / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        out = np.zeros((H, LEFT_W, 3), dtype=np.uint8)
        x = (LEFT_W - nw) // 2
        y = (H - nh) // 2
        out[y : y + nh, x : x + nw] = resized
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def clip_duration(clip: Clip) -> float:
    if clip.kind == "title":
        return clip.end
    if clip.display_duration > 0:
        return clip.display_duration
    return max(0.01, clip.end - clip.start)


def iter_video_frames(clip: Clip, clip_no: int, clip_total: int):
    assert clip.source is not None
    src = video_for(clip.source)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {src}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    start_frame = max(0, int(round(clip.start * src_fps)))
    end_frame = max(start_frame + 1, int(round(clip.end * src_fps)))
    source_span = max(1, end_frame - start_frame)
    duration = clip_duration(clip)
    out_frames = int(round(duration * FPS))
    for idx in range(out_frames):
        progress = idx / max(1, out_frames - 1)
        desired = min(end_frame - 1, start_frame + int(round(progress * (source_span - 1))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, desired)
        ok, frame = cap.read()
        if not ok:
            break
        img = Image.new("RGBA", (W, H), (8, 13, 20, 255))
        img.paste(resize_source(frame).convert("RGBA"), (0, 0))
        draw = ImageDraw.Draw(img, "RGBA")
        local_t = idx / FPS
        draw_panel(draw, clip, local_t, duration, clip_no, clip_total)
        add_lower_caption(draw, f"{clip.title}：{clip.subtitle}")
        fade_frames = min(int(0.35 * FPS), max(2, out_frames // 5))
        fade_alpha = 128
        if idx < fade_frames:
            alpha = int(fade_alpha * (1 - idx / max(1, fade_frames)))
            img.alpha_composite(Image.new("RGBA", (W, H), (0, 0, 0, alpha)))
        elif idx > out_frames - fade_frames:
            rem = out_frames - idx
            alpha = int(fade_alpha * (1 - rem / max(1, fade_frames)))
            img.alpha_composite(Image.new("RGBA", (W, H), (0, 0, 0, alpha)))
        yield np.asarray(img.convert("RGB"))
    cap.release()


def iter_all_frames(plan: list[Clip]):
    video_clips = [clip for clip in plan if clip.kind == "video"]
    clip_no = 0
    for clip in plan:
        if clip.kind == "title":
            total = int(round(clip.end * FPS))
            for frame_idx in range(total):
                yield draw_title_card(clip, frame_idx, total)
        else:
            clip_no += 1
            yield from iter_video_frames(clip, clip_no, len(video_clips))


def encode_video(plan: list[Clip], crf: int, output: Path) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{W}x{H}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in iter_all_frames(plan):
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    code = proc.wait()
    if code != 0:
        raise RuntimeError(stderr[-4000:])


def enrich_plan(plan: list[Clip]) -> list[Clip]:
    enriched: list[Clip] = []
    for clip in plan:
        if clip.kind != "video" or clip.source is None:
            enriched.append(clip)
            continue
        pose, latency, alignment = avg_stats(clip.source, clip.start, clip.end)
        metric = f"平均姿态质量 / 延迟: {pose:.3f} / {latency:.1f}ms\n动作/命令一致率: {alignment * 100:.1f}%\n主要输出: {top_commands(clip.source, clip.start, clip.end)}"
        enriched.append(Clip(**{**clip.__dict__, "metric": metric}))
    return enriched


def write_plan(plan: list[Clip], output: Path, crf: int) -> None:
    payload = {
        "output": str(output),
        "resolution": [W, H],
        "fps": FPS,
        "crf": crf,
        "max_size_bytes": MAX_BYTES,
        "duration_sec": sum(clip_duration(clip) for clip in plan),
        "clips": [clip.__dict__ for clip in plan],
    }
    PLAN_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def srt_time(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours = millis // 3_600_000
    millis %= 3_600_000
    minutes = millis // 60_000
    millis %= 60_000
    secs = millis // 1000
    millis %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(plan: list[Clip]) -> None:
    cursor = 0.0
    blocks = []
    idx = 1
    for clip in plan:
        duration = clip_duration(clip)
        start = cursor
        end = cursor + duration
        text = f"{clip.title}\n{clip.subtitle}"
        blocks.append(f"{idx}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n")
        cursor = end
        idx += 1
    SRT_PATH.write_text("\n".join(blocks), encoding="utf-8")


def create_contact_sheet(video: Path) -> None:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    picks = [int(total * p) for p in (0.08, 0.18, 0.30, 0.42, 0.54, 0.66, 0.78, 0.90)]
    thumbs = []
    for frame_no in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
        thumbs.append(frame)
    cap.release()
    if not thumbs:
        return
    rows = math.ceil(len(thumbs) / 2)
    sheet = np.full((rows * 270, 960, 3), 245, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        y = (i // 2) * 270
        x = (i % 2) * 480
        sheet[y : y + 270, x : x + 480] = thumb
    cv2.imwrite(str(CONTACT_SHEET), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def main() -> None:
    read_manifest()
    plan = enrich_plan(make_plan())
    temp = OUTPUT.with_name(OUTPUT.stem + "_tmp.mp4")
    final_crf = CRF_TARGETS[-1]
    for crf in CRF_TARGETS:
        encode_video(plan, crf, temp)
        size = temp.stat().st_size
        final_crf = crf
        if size <= MAX_BYTES:
            break
    temp.replace(OUTPUT)
    write_plan(plan, OUTPUT, final_crf)
    write_srt(plan)
    create_contact_sheet(OUTPUT)
    print(f"output={OUTPUT}")
    print(f"size={OUTPUT.stat().st_size}")
    print(f"crf={final_crf}")
    print(f"plan={PLAN_PATH}")
    print(f"srt={SRT_PATH}")
    print(f"contact_sheet={CONTACT_SHEET}")


if __name__ == "__main__":
    main()
