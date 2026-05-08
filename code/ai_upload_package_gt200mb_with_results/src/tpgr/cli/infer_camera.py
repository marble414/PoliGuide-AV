from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from tpgr.cli.common import configure_runtime
from tpgr.config import load_config
from tpgr.io.json_io import JsonlWriter
from tpgr.io.video_io import get_video_writer
from tpgr.io.visualization import draw_detections, draw_result_panel
from tpgr.pipeline.system import TrafficPoliceGestureSystem


def main() -> None:
    parser = argparse.ArgumentParser(description="摄像头实时推理")
    parser.add_argument("--config", required=True)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--output-video", default="outputs/camera_demo.mp4")
    parser.add_argument("--output-jsonl", default="outputs/camera_demo.jsonl")
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument("--debug", action="store_true", help="输出右侧 debug 仪表盘，并在 jsonl 中保留完整中间状态")
    parser.add_argument("--debug-topk", type=int, default=5, help="debug 面板中显示的 top-k 分类分数")
    parser.add_argument("--debug-panel-width", type=int, default=440, help="debug 仪表盘宽度")
    parser.set_defaults(display=True)
    args = parser.parse_args()

    configure_runtime(1)
    cfg = load_config(args.config)
    system = TrafficPoliceGestureSystem(cfg)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {args.camera_id}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps > 1e-3 else 25.0

    writer = None
    with JsonlWriter(args.output_jsonl) as jsonl_writer:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
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
            jsonl_writer.write({**result.to_dict(), "detections": detections})

            if args.display:
                cv2.imshow("tpgr-camera", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    if writer is not None:
        writer.release()
    cap.release()
    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
