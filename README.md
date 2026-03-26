# TennisProject

Upload a tennis match video, and the system automatically detects players, tracks the ball, identifies every shot and rally, then generates an annotated video and detailed match statistics — all in one click.

## What It Does

### Annotated Video Output

The system overlays the original video with:

- **Court lines** — detected via keypoint model, projected onto the frame
- **Player skeletons** — top player in green, bottom player in yellow
- **Ball trail** — color follows the player who last hit the ball, with smooth gradient transitions at each contact point

### Rally & Shot Statistics

One-click analysis produces a full match breakdown:

| Tab | What You See |
|---|---|
| **Rally** | Each rally listed with shot count, duration, server, and per-shot details (player, type, speed). Click any shot to jump to that moment in the video. |
| **Player** | Per-player stats: total shots, serves, winners, shot type distribution |
| **Depth** | Court position distribution — how often each player was at the net, service line, or baseline |
| **Speed** | Ball speed stats (avg/max/min) for all shots, serves, and rally balls |
| **Court** | SVG court diagram with shot landing heatmap and winner landing markers |

### AI Chat

Ask questions about the match in natural language. The Vision-Language Model (Qwen) has access to both the video and the full analysis JSON, enabling answers like:

- *"Who had better court coverage?"*
- *"Summarize the longest rally"*
- *"Which player served faster on average?"*

### Account & History

- **Registered users** — manage multiple videos, revisit past analyses, re-analyze with updated models
- **Guest mode** — no login required; upload and analyze immediately (auto-expires after 7 days)

## How It Works

```
Upload video → Court Detection → Pose Estimation → Ball Tracking
  → Sliding Window (outlier filtering + interpolation)
  → Rally-Aware Rendering (buffer rally frames → detect events → color ball trail → encode)
  → Event Detection → Rally Segmentation → VLM Winner Judgment
  → Annotated MP4 + Analysis JSON
```

Key technical highlights:

- **Single-pass pipeline** — detection, filtering, rendering, and encoding happen in one streaming pass (FFmpeg decode → process → FFmpeg encode)
- **Rally-aware buffering** — frames during active rallies are buffered in memory; when a rally ends, `detect_events` runs on the segment to determine precise ball ownership before drawing the trail
- **Sliding window** — 30-frame delay ensures ball positions are filtered and interpolated before being finalized (outlier removal, gap interpolation with direction checking)
- **No duplicate computation** — per-rally event results are accumulated and reused for the final analysis, avoiding a redundant full-video pass

## System Requirements

### GPU (VRAM)

| Component | VRAM | Note |
|---|---|---|
| YOLO models (ball + pose + court) | ~2 GB | Always required. Three small models run on the same GPU. |
| vLLM — Qwen3-VL-**2B** | ~6 GB | Minimum for AI chat |
| vLLM — Qwen3-VL-**8B** | ~20 GB | Recommended for quality |
| vLLM — Qwen3.5-**27B**-FP8 | ~30 GB | Best quality, needs A100/4090+ |

YOLO and vLLM share the same GPU. For a 24 GB card (e.g. RTX 4090), use the 8B model with `gpu-memory-utilization 0.85`.

**No local GPU for VLM?** The system doesn't require a built-in vLLM instance. You can point `APP_VLLM_URL` in `.env` to any OpenAI-compatible API endpoint (e.g. a remote server, cloud GPU, or third-party API). Rally winner judgment and AI chat both go through this endpoint. Without any VLM configured, analysis still runs — but rally win/loss judgment and AI chat will be unavailable.

### RAM

| Source | Memory | Note |
|---|---|---|
| Base (backend + FFmpeg) | ~2 GB | Python process + decode/encode pipes |
| Rally frame buffer | **~6 MB × rally length (frames)** | 1080p frame = 1920×1080×3 ≈ 5.93 MB. A 10-second rally at 60fps (600 frames) ≈ **3.6 GB**. Freed after each rally flush. |
| Position arrays | ~50 MB per 10k frames | `all_ball_positions`, `all_player_*`, `all_wrist_*`, `all_ball_owner` |

**Recommendation: 16 GB+ RAM.** Long rallies (15s+) at 1080p60 can peak at 5+ GB for the frame buffer alone. Lower resolution reduces memory proportionally (720p ≈ 2.7 MB/frame).

### Disk

- Uploaded videos + annotated output (2× original size)
- vLLM model cache (`data/huggingface/`): 5–60 GB depending on model
- MySQL data: minimal

## Quick Start

### 1. Clone & download models

```sh
git clone https://github.com/violetzu/TennisProject.git
cd TennisProject/
bash ./backend/models/download.sh
```

Trained model weights (ball, court) should be placed under `backend/models/`. Model paths are configured in `backend/config.py`. See [Model Training](#model-training) for dataset sources and training scripts.

### 2. Configure

```sh
cp .env.example .env
```

| Variable | Note |
|---|---|
| `APP_VLLM_MODEL` | VLM model name. `Qwen/Qwen3-VL-8B-Instruct` needs ~30 GB VRAM. Try 2B/4B for smaller GPUs. |
| `FRONTEND_DEV_MODE` | `true` = `npm run dev` with hot reload |
| `BACKEND_DEV_MODE` | `true` = uvicorn `--reload` |
| `CLOUDFLARE_TUNNEL_TOKEN` | Leave empty if not exposing externally |

### 3. Run

```sh
# With VLM (GPU required)
docker compose --profile vllm up -d --build

# Without VLM (analysis works, AI chat disabled)
docker compose up -d --build
```

Open **http://localhost:3000**

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 16, React 19, Tailwind CSS v4 |
| Backend | FastAPI, SQLAlchemy, MySQL 8.0 |
| Detection | YOLOv8 (ball), YOLO26n-pose (player & court) |
| VLM | Qwen3-VL via vLLM |
| Video | FFmpeg pipe (decode/encode), OpenCV |
| Deploy | Docker Compose, Cloudflare Tunnel |

## Model Training

All three detection models were custom-trained. Training scripts, dataset download scripts, and conversion tools are in `backend/models/`.

### Ball Detection

| Item | Detail |
|---|---|
| Base model | YOLOv26s |
| Dataset | [TrackNet](https://github.com/yastrebksv/TrackNet?tab=readme-ov-file#dataset) (44,747 frames, 1280×720, 30fps) — converted to YOLO format via `tracknet_to_yolo.py` (center point → 16px bbox) |
| Training | 300 epochs, imgsz 1280, batch 48, rect, mixup 0.1 |
| Result | **mAP50 0.956** |
| Scripts | `tracknet_dataset.sh` (download), `tracknet_to_yolo.py` (convert), `tracknet_ball_train.sh` (train), `ball_val.sh` (validate) |

### Court Keypoint Detection

| Item | Detail |
|---|---|
| Base model | YOLOv26n-pose |
| Dataset | [Roboflow Tennis Court](https://universe.roboflow.com/ds/zINRxAuTHq?key=r4cjK6OqYT) — 14 keypoints (4 corners + service lines + center line) |
| Training | 100 epochs, imgsz 320, rect, mixup 0.1, augmentation (degrees 5, scale 0.3, shear 2, perspective 0.0001) |
| Result | **mAP50 0.994** |
| Scripts | `court_dataset.sh` (download), `court_train.sh` (train), `court_val.sh` (validate) |

### Player Pose Estimation

| Item | Detail |
|---|---|
| Model | YOLOv26n-pose (pretrained COCO 17-keypoint) |
| Training | None — used as-is |
| Scripts | `download.sh` |

## Documentation

- **[SPEC.md](SPEC.md)** — Full system specification in Chinese: architecture, database schema, backend/frontend details, analysis pipeline internals, JSON format, API reference

## Links

- [Google Drive — Models & Test Videos](https://drive.google.com/drive/folders/1ttI0QDaQ6rkU-6uh9F-09ewdqgxi_HqU?usp=drive_link)
- [(Guide) Ubuntu Docker + NVIDIA Container Toolkit](https://github.com/violetzu/knowledge/blob/01ecf7828174c0a082418e4410d5e8081abc7799/docker%20install.md)
