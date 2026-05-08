#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tpgr.cli.common import configure_runtime
from tpgr.config import load_config
from tpgr.data.sequence_dataset import SequenceDataset, collate_batch
from tpgr.io.video_io import iter_video
from tpgr.models.builder import build_model
from tpgr.pipeline.system import TrafficPoliceGestureSystem


def main() -> None:
    configure_runtime(1)
    root = Path.cwd()
    cfg = load_config(root / 'configs' / 'toy_tcn.yaml')
    ds = SequenceDataset(
        manifest_path=cfg['dataset']['manifest_path'],
        split='train',
        clip_len=int(cfg['dataset']['clip_len']),
        representation=cfg['dataset']['representation'],
        training=True,
        dataset_name='toy',
    )
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_batch)
    batch = next(iter(loader))

    model = build_model(cfg['model'])
    x = batch['x'].float()
    y = batch['y']
    logits = model(x)
    loss = torch.nn.CrossEntropyLoss()(logits, y)
    loss.backward()

    demo_cfg = load_config(root / 'configs' / 'deploy_precomputed_demo.yaml')
    system = TrafficPoliceGestureSystem(demo_cfg)
    demo_video = root / 'data' / 'processed' / 'toy' / 'demo' / 'toy_demo.mp4'

    commands = []
    for idx, frame, fps in iter_video(str(demo_video)):
        result, _ = system.process_frame(frame, fps=fps)
        commands.append(result.command)
        if idx >= 24:
            break

    summary = {
        'dataset_size': len(ds),
        'batch_shape': list(x.shape),
        'logits_shape': list(logits.shape),
        'loss': float(loss.item()),
        'first_25_commands': commands,
        'non_no_command_frames': sum(int(cmd != 'NO_COMMAND') for cmd in commands),
    }
    out_path = root / 'runs' / 'smoke_check.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
