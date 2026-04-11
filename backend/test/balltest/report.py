from __future__ import annotations

from .paths import ABLATION_RESULT_PATH, METHODS_RESULT_PATH, REPORT_PATH, TRACKERS_RESULT_PATH
from .utils import json_load


def _load_rows(path, label: str) -> list[dict]:
    rows = json_load(path, default=[])
    if not isinstance(rows, list):
        raise RuntimeError(f"{label} must be a JSON list: {path}")
    return rows


def _is_tracker_system_method(method: str) -> bool:
    return method == "yolo_balltracker" or method.endswith("_balltracker")


def _render_methods_table(rows: list[dict]) -> list[str]:
    if not rows:
        return ["(尚未產生方法比較結果)"]
    lines = [
        "| Method | Precision | Recall | F1 | Mean Err | Median Err | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {mean_error_px:.2f} | "
            "{median_error_px:.2f} | {throughput_fps:.2f} |".format(**row)
        )
    return lines


def _render_ablation_table(rows: list[dict]) -> list[str]:
    if not rows:
        return ["(尚未產生消融結果)"]
    lines = [
        "| Mode | Precision | Recall | F1 | Mean Err | Median Err | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {mode} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {mean_error_px:.2f} | "
            "{median_error_px:.2f} | {throughput_fps:.2f} |".format(**row)
        )
    return lines


def _render_tracker_table(rows: list[dict], mode_key: str) -> list[str]:
    if not rows:
        return ["(尚未產生追蹤器結果)"]
    lines = [
        f"| {mode_key} | Gap Frame Recall | Gap Rate | FP Suppression | Noise Rejection | Jitter |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {natural_gap_recovery_frame_recall:.4f} | {natural_gap_recovery_gap_rate:.4f} | "
            "{natural_fp_suppression_rate:.4f} | {synthetic_noise_rejection_rate:.4f} | {jitter_px:.2f} |".format(
                label=row[mode_key],
                **row,
            )
        )
    return lines


def _render_trackers_table(rows: list[dict]) -> list[str]:
    if not rows:
        return ["(尚未產生追蹤器比較結果)"]
    lines = [
        "| Method | Precision | Recall | F1 | Mean Err | Median Err | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {mean_error_px:.2f} | "
            "{median_error_px:.2f} | {throughput_fps:.2f} |".format(**row)
        )
    return lines


def report_command(args) -> None:
    del args
    method_rows = [row for row in _load_rows(METHODS_RESULT_PATH, "methods result") if not _is_tracker_system_method(str(row.get("method", "")))]
    ablation_rows = _load_rows(ABLATION_RESULT_PATH, "ablation result")
    tracker_rows = _load_rows(TRACKERS_RESULT_PATH, "trackers result")

    lines = [
        "# 球追蹤實驗報告",
        "",
        "這份報告只整理結果與必要的最小實驗設定。完整流程、資料協議、訓練方式與操作步驟請見 `backend/test/balltest/README.md`。",
        "",
        "## 最小實驗設定",
        "",
        "- 正式結果只使用 `test` split。",
        "- 模型訓練使用 `train`，checkpoint selection 使用 `select`。",
        "- 追蹤器比較與所有消融都直接重播同一份 YOLO raw detection cache。",
        "- `median_error_px` 為全域 TP error 中位數。",
        "- `FPS` 以端到端 frame pipeline 計時：讀圖、resize / 前處理、模型推論與座標後處理都算入。",
        "",
        "## 論文基準",
        "",
        "- `TrackNet` 依 Huang et al. (2019) 重建：3-frame / 9-channel 輸入、`640x360`、256-class heatmap、threshold + Hough circle 後處理。",
        "- `TrackNetV4` 依 Raj et al. (2025) 重建：`512x288`、motion prompt layer、multiple-input multiple-output、motion-aware fusion。",
        "- `TrackNetV5` 依 Tang et al. (2025) 重建：`512x288`、13-channel MDD 輸入、V2-style backbone、R-STR 殘差 refinement 與 TSATTHead。",
        "- 三個 TrackNet 家族方法的輸入解析度不完全相同；本實驗保留各論文設定，不另外強行統一解析度。",
        "",
        "## 方法比較",
        "",
    ]
    lines.extend(_render_methods_table(method_rows))
    lines.extend(
        [
            "",
            "- 這一節只比較 raw detector / raw sequence model 的最終輸出。",
            "- `BallTracker`、SORT、ByteTrack 等後處理系統不納入主方法表，改列在後面的追蹤器比較。",
            "",
            "## 追蹤器消融（yolo_botsort_finalized 各元件移除）",
            "",
        ]
    )
    lines.extend(_render_ablation_table(ablation_rows))
    lines.extend(
        [
            "",
            "### 消融補幀與過濾指標",
            "",
        ]
    )
    ablation_tracker_rows = [

        {
            "mode": row.get("mode", row.get("tracker_mode", "unknown")),
            "natural_gap_recovery_frame_recall": row.get("natural_gap_recovery_frame_recall", 0.0),
            "natural_gap_recovery_gap_rate": row.get("natural_gap_recovery_gap_rate", 0.0),
            "natural_fp_suppression_rate": row.get("natural_fp_suppression_rate", 0.0),
            "synthetic_noise_rejection_rate": row.get("synthetic_noise_rejection_rate", 0.0),
            "jitter_px": row.get("jitter_px", 0.0),
        }
        for row in ablation_rows
    ]
    lines.extend(_render_tracker_table(ablation_tracker_rows, "mode"))
    lines.extend(
        [
            "",
            "## 追蹤器比較",
            "",
            "- 所有方法共用同一份 YOLO detection cache，finalize 後處理完全相同。",
            "- 比較項目：`yolo_balltracker`（我們）vs SORT / DeepSORT / ByteTrack / ByteTrack+finalize / BoT-SORT / BoT-SORT+finalize。",
            "",
        ]
    )
    lines.extend(_render_trackers_table(tracker_rows))
    tracker_gap_rows = [
        {
            "method": row["method"],
            "natural_gap_recovery_frame_recall": row.get("natural_gap_recovery_frame_recall", 0.0),
            "natural_gap_recovery_gap_rate": row.get("natural_gap_recovery_gap_rate", 0.0),
            "natural_fp_suppression_rate": row.get("natural_fp_suppression_rate", 0.0),
            "synthetic_noise_rejection_rate": row.get("synthetic_noise_rejection_rate", 0.0),
            "jitter_px": row.get("jitter_px", 0.0),
        }
        for row in tracker_rows
    ]
    lines.extend(["", "### 追蹤補幀與過濾指標", ""])
    lines.extend(_render_tracker_table(tracker_gap_rows, "method"))
    lines.extend(
        [
            "",
            "## 主要輸出",
            "",
            "- `backend/test/balltest/artifacts/results/methods.json`",
            "- `backend/test/balltest/artifacts/results/ablation.json`",
            "- `backend/test/balltest/artifacts/results/trackers.json`",
            "- `backend/test/balltest/實驗報告.md`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] wrote {REPORT_PATH}")
