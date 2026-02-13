# tennis_analysis_v3.0.0/backend/utils/video_utils.py
import cv2
import numpy as np
from typing import Iterator, Tuple, Optional, Dict


# --- Basic video read/write utilities ---

def read_video(video_path: str) -> list[np.ndarray]:
    """一次性讀取整部影片，回傳所有影格 (frames)。"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video_unified(frames, output_path, fps=24.0, verbose=True):
    """
    儲存影片為 MP4 格式。

    frames: list[np.ndarray]
    output_path: 輸出路徑
    fps: 幀率
    回傳: 實際儲存檔案路徑
    """
    import os

    if not frames:
        raise ValueError("沒有影片幀可以保存")

    if not hasattr(frames, '__len__'):
        frames = list(frames)

    h, w = frames[0].shape[:2]

    def _log(*args):
        if verbose:
            print(*args)

    # 定義嘗試的編碼與對應副檔名
    # 優先嘗試 H.264 (avc1) -> .mp4
    # 其次嘗試 VP9 (vp09) -> .webm (瀏覽器支援度高)
    # 其次嘗試 VP8 (vp80) -> .webm
    # 最後嘗試 MPEG-4 (mp4v) -> .mp4 (瀏覽器可能不支援，但至少能產出檔案)
    codecs_to_try = [
        ('mp4v', '.mp4'),
        ('avc1', '.mp4'),
        ('vp09', '.webm'),
        ('vp80', '.webm'),
        
    ]
    
    out = None
    used_codec = None
    final_out_path = None

    for codec, ext in codecs_to_try:
        try:
            # 根據編碼調整副檔名
            current_out_path = os.path.splitext(output_path)[0] + ext
            fourcc = cv2.VideoWriter_fourcc(*codec)
            temp_out = cv2.VideoWriter(current_out_path, fourcc, float(fps), (w, h), True)
            
            if temp_out.isOpened():
                out = temp_out
                used_codec = codec
                final_out_path = current_out_path
                _log(f"成功初始化 VideoWriter，使用 codec: {codec}, 輸出: {final_out_path}")
                break
            else:
                _log(f"無法使用 codec {codec} 初始化 VideoWriter，嘗試下一個...")
        except Exception as e:
            _log(f"嘗試 codec {codec} 時發生錯誤: {e}")

    if out is None or not out.isOpened():
        raise Exception(f"VideoWriter 初始化失敗，已嘗試 codecs: {codecs_to_try}")
    
    # 更新 out_path 為最終成功的路徑
    out_path = final_out_path

    for i, frame in enumerate(frames):
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)

        if frame.dtype != np.uint8:
            maxv = float(frame.max()) if frame.size else 1.0
            frame = (frame * 255).astype(np.uint8) if maxv <= 1.0 else frame.astype(np.uint8)

        if frame.ndim == 2:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            frame_bgr = frame
        else:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        out.write(frame_bgr)

        if (i + 1) % 100 == 0:
            _log(f"已寫入 {i + 1}/{len(frames)} 幀")

    out.release()

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        _log(f"影片保存成功: {out_path} ({os.path.getsize(out_path)} bytes)")
        return out_path
    else:
        raise Exception(f"影片檔案創建失敗或為空: {out_path}")


def save_video(frames, output_path, fps=24.0, verbose=True):
    """簡易封裝，預設使用 mp4v"""
    return save_video_unified(frames, output_path, fps=fps, verbose=verbose)


def save_video_mp4(frames, output_path, fps=24.0, verbose=True):
    """強制 mp4 格式封裝"""
    return save_video_unified(frames, output_path, fps=fps, verbose=verbose)


def get_video_fps(video_path, strict=False, default=24.0) -> float:
    """取得影片 FPS，若無法讀取則回傳預設值。"""
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()

    if fps and fps > 0:
        return float(fps)
    if strict:
        raise ValueError(f"無法從影片讀取 FPS: {video_path}")
    return float(default)


def get_video_metadata(video_path: str) -> Dict[str, Optional[float]]:
    """取得影片基本資訊（寬、高、FPS、幀數、估計時長）。"""
    meta = {
        'width': None,
        'height': None,
        'fps': None,
        'frame_count': None,
        'duration_seconds': None,
    }

    cap = cv2.VideoCapture(video_path)
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()

    meta['width'] = width or None
    meta['height'] = height or None
    meta['fps'] = float(fps) if fps and fps > 0 else None
    meta['frame_count'] = frame_count or None
    meta['duration_seconds'] = (
        meta['frame_count'] / meta['fps'] if meta['fps'] and meta['frame_count'] else None
    )

    return meta


# --- New: Iterative video reader with frame index and timestamp ---

class VideoReader:
    """
    可逐幀讀取影片，並回傳 (frame_index, frame, timestamp)。
    """

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"無法開啟影片: {video_path}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0) or 24.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._index = 0

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray, float]]:
        """每次產生 (frame_index, frame, timestamp_seconds)"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            ts = self._index / self.fps if self.fps else 0.0
            yield self._index, frame, ts
            self._index += 1

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def read_video_iter(video_path: str) -> Iterator[Tuple[int, np.ndarray, float]]:
    """封裝版生成器，用於簡化逐幀讀取流程。"""
    vr = VideoReader(video_path)
    try:
        for item in vr:
            yield item
    finally:
        vr.close()


# --- CLI 測試 ---
if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    import os

    parser = argparse.ArgumentParser(description="Video utils CLI: iterate frames and print metadata")
    parser.add_argument("video", help="Path to input video file")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise SystemExit(f"找不到影片檔案: {args.video}")

    meta = get_video_metadata(args.video)
    fps = meta.get("fps") or 24.0
    width = meta.get("width") or 0
    height = meta.get("height") or 0
    frame_count = meta.get("frame_count") or None

    print(f"影片資訊 - fps: {fps}, size: {width}x{height}, frames: {frame_count}")

    total = frame_count if frame_count and frame_count > 0 else None
    iter_gen = read_video_iter(args.video)

    for frame_index, frame, ts in tqdm(iter_gen, total=total, desc="Reading frames"):
        pass

    print("讀取完成。")
