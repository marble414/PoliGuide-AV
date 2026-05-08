#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

CTPGESTURE_V1_FILE_ID = "1QT88DwKyhJ4-hEk81YEpGvikKDS_uqjj"
CTPGESTURE_V2_FILE_ID = "1ItPsIYY828LPkoal1y9TEfrg-_IDchj3"


def maybe_gdown(file_id: str, output: Path) -> None:
    gdown = shutil.which("gdown")
    if gdown:
        cmd = [gdown, f"https://drive.google.com/uc?id={file_id}", "-O", str(output)]
    else:
        try:
            import gdown  # noqa: F401
        except ImportError as e:
            raise RuntimeError("未找到 gdown，请先 `pip install gdown`。") from e
        cmd = [sys.executable, "-m", "gdown", f"https://drive.google.com/uc?id={file_id}", "-O", str(output)]
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="下载公开可访问的交警手势相关数据入口")
    parser.add_argument("--dataset", required=True, choices=["ctpgesture_v1", "ctpgesture_v2", "tcg"])
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--download", action="store_true", help="若提供则尝试自动下载；否则仅打印说明")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "ctpgesture_v1":
        target = out_dir / "ctpgesture_v1.zip"
        print("CTPGesture v1: 公开 Google Drive 文件，建议人工核验数据许可边界后使用。")
        print(f"建议保存路径: {target}")
        if args.download:
            maybe_gdown(CTPGESTURE_V1_FILE_ID, target)
        else:
            print("自动下载命令:")
            print(f"python tools/download_datasets.py --dataset ctpgesture_v1 --output-dir {out_dir} --download")
        return

    if args.dataset == "ctpgesture_v2":
        target = out_dir / "ctpgesture_v2.zip"
        print("CTPGesture v2: 公开 Google Drive 文件，含方向增强标注。建议人工核验许可边界。")
        print(f"建议保存路径: {target}")
        if args.download:
            maybe_gdown(CTPGESTURE_V2_FILE_ID, target)
        else:
            print("自动下载命令:")
            print(f"python tools/download_datasets.py --dataset ctpgesture_v2 --output-dir {out_dir} --download")
        return

    if args.dataset == "tcg":
        print("TCG 数据集需邮件申请，无法自动下载。")
        print("请参考 tools/request_tcg_email_template.md")
        return


if __name__ == "__main__":
    main()
