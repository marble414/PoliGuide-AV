from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from tpgr.cli.common import configure_runtime
from tpgr.config import ensure_dir, load_config
from tpgr.io.json_io import JsonlWriter
from tpgr.io.video_io import get_video_writer, iter_video
from tpgr.io.visualization import draw_detections, draw_result_panel
from tpgr.pipeline.system import TrafficPoliceGestureSystem


def main() -> None:
    parser = argparse.ArgumentParser(description="视频推理")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-video", default="outputs/infer.mp4")
    parser.add_argument("--output-jsonl", default="outputs/infer.jsonl")
    parser.add_argument("--display", dest="display", action="store_true")
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument("--debug", action="store_true", help="输出右侧 debug 仪表盘，并在 jsonl 中保留完整中间状态")
    parser.add_argument("--debug-topk", type=int, default=5, help="debug 面板中显示的 top-k 分类分数")
    parser.add_argument("--debug-panel-width", type=int, default=440, help="debug 仪表盘宽度")
    parser.set_defaults(display=None)
    args = parser.parse_args()

    configure_runtime(1)
    cfg = load_config(args.config)
    system = TrafficPoliceGestureSystem(cfg)
    writer = None
    display = bool(args.debug) if args.display is None else bool(args.display)

    with JsonlWriter(args.output_jsonl) as jsonl_writer:
        for frame_index, frame, fps in tqdm(iter_video(args.input), desc="infer"):
            result, detections = system.process_frame(frame, fps=fps)
            vis = frame.copy()
            vis = draw_detections(vis, detections, result.active_track_id, debug=args.debug)
            vis = draw_result_panel(
                vis,
                result,
                debug=args.debug,
                topk=args.debug_topk,
                panel_width=args.debug_panel_width,
            )

            if writer is None:
                writer = get_video_writer(args.output_video, fps, (vis.shape[1], vis.shape[0]))
            writer.write(vis)
            jsonl_writer.write({
                **result.to_dict(),
                "detections": detections,
            })

            if display:
                cv2.imshow("tpgr", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    if writer is not None:
        writer.release()
    if display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
