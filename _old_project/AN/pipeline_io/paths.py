from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PipelinePaths:
    input_path: str
    basename: str
    data_dir: str
    video_dir: str
    output_video_dir: str
    output_minicourt_dir: str
    video_json_path: str
    world_json_path: str


def prepare_paths(
    input_path: str,
    output_name: Optional[str] = None,
    *,
    data_dir: str,
    video_dir: str,
) -> PipelinePaths:
    """Resolve absolute paths & create output directories.

    This is the拆分自原本 pipeline/main.py 的「路徑準備」段落。
    """
    input_abs = os.path.abspath(input_path)
    if not os.path.exists(input_abs):
        raise FileNotFoundError(f"影片不存在: {input_abs}")

    basename = output_name or os.path.splitext(os.path.basename(input_abs))[0]

    output_video_dir = os.path.join(video_dir, "output_videos")
    output_minicourt_dir = os.path.join(video_dir, "output_minicourt")
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_minicourt_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    video_json_path = os.path.join(data_dir, f"video_info_{basename}.json")
    world_json_path = os.path.join(data_dir, f"world_info_{basename}.json")

    return PipelinePaths(
        input_path=input_abs,
        basename=basename,
        data_dir=data_dir,
        video_dir=video_dir,
        output_video_dir=output_video_dir,
        output_minicourt_dir=output_minicourt_dir,
        video_json_path=video_json_path,
        world_json_path=world_json_path,
    )
