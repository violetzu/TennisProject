"""工具函式：提供影片場景切割功能。"""

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.stats_manager import StatsManager

def scene_detect(path_video):
    """
    Split video to disjoint fragments based on color histograms

    根據顏色直方圖變化切分影片，取得各段場景範圍。
    """
    video = open_video(path_video)               # 取代 VideoManager
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector())

    # 直接把 video 丟進去偵測（不再使用 set_downscale_factor / start）
    scene_manager.detect_scenes(video=video)

    # v0.6+ 不再需要 base_timecode 參數
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        # 若場景偵測失敗，退回整部影片視為單段
        start_tc = getattr(video, 'base_timecode', None)
        end_tc = getattr(video, 'duration', None)
        # 安全回退：若屬性不存在就用 0 ~ 0
        if start_tc is None or end_tc is None:
            return [[0, 0]]
        return [[start_tc.get_frames(), end_tc.get_frames()]]

    # 將 Timecode 轉成 frame index（v0.6+ 用 get_frames()）
    scenes = [[start.get_frames(), end.get_frames()] for start, end in scene_list]
    return scenes
