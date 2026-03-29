"""
FFmpeg 影片 I/O 管線 (Video I/O Pipeline)

封裝 FFmpeg decode + encode 子程序管線，支援 context manager。
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Generator, Optional

import numpy as np


class VideoPipe:
    """FFmpeg decode + encode 管線。

    用法::

        with VideoPipe(video_path, out_path, w, h, fps) as pipe:
            for frame in pipe.frames():
                # frame: 可寫的 (h, w, 3) uint8 ndarray
                ...
                pipe.write(frame)
    """

    def __init__(
        self,
        video_path: Path,
        out_path: Path,
        width: int,
        height: int,
        fps: float,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_bytes = width * height * 3

        self._dec = self._start_decoder(video_path, width, height)
        self._enc = self._start_encoder(out_path, width, height, fps)

    # ── public ────────────────────────────────────────────────────────────────

    def frames(self) -> Generator[np.ndarray, None, None]:
        """逐幀 decode，yield 可寫的 numpy array (h, w, 3)。"""
        while True:
            raw = self._dec.stdout.read(self.frame_bytes) if self._dec.stdout else b""
            if not raw or len(raw) < self.frame_bytes:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.height, self.width, 3)
            ).copy()

    def write(self, frame: np.ndarray) -> None:
        """寫一幀到 encoder pipe。"""
        if self._enc and self._enc.stdin:
            self._enc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        """關閉管線、等待子程序結束。"""
        if self._dec.stdout:
            self._dec.stdout.close()
        self._dec.wait()
        if self._enc:
            if self._enc.stdin:
                self._enc.stdin.close()
            self._enc.wait()

    def __enter__(self) -> VideoPipe:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _start_decoder(path: Path, w: int, h: int) -> subprocess.Popen:
        cmd = [
            "/usr/bin/ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(path), "-an",
            "-vf", f"scale={w}:{h}",
            "-pix_fmt", "bgr24", "-f", "rawvideo", "-vsync", "0", "pipe:1",
        ]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    @staticmethod
    def _start_encoder(path: Path, w: int, h: int, fps: float) -> subprocess.Popen:
        cmd = [
            "/usr/bin/ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
            "-r", str(fps), "-i", "pipe:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ]
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
