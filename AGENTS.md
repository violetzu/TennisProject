# tennis

## 專案概述

網球影片分析系統：使用者上傳影片後，後端以 YOLO / TrackNet 模型偵測球路、球員姿態、場地，並透過 vLLM 提供 AI 對話分析功能。支援訪客模式（無登入）與個人帳號模式。

## 架構

§C：FastAPI 主導 + Next.js 薄殼，DB 由 FastAPI 管（SQLAlchemy + PostgreSQL）；FastAPI 負責 CV 推理、影片分析、Auth（自訂 JWT）

專案名（DB user / DB name）：`tennis`
部署網址：`https://your-domain.example.com`

## Auth

Auth-0：Next.js 無 NextAuth；Auth 完全由 FastAPI JWT 處理（個人帳號 + 訪客 token 皆在 fastapi-service）

## 常用指令

```bash
make setup       # 第一次：複製 .env.example → .env，填入 secrets 後再啟動
make up          # 啟動 production stack
make dev-up      # 啟動 dev stack（frontend HMR + fastapi --reload）
make dev-logs    # tail 所有 container log
make db-backup   # 備份 DB → backup.dump
make up-vllm     # 啟動含 vLLM profile 的 production stack

# 進入 DB
docker compose exec db psql -U tennis -d tennis
```

## 開發規則

- 本機不安裝任何 Node / Python 套件，所有指令透過 Docker 執行
- Next.js 16（node:22-bookworm-slim）；fastapi-service Python（pytorch/pytorch CUDA image）
- 前端呼叫 fastapi 一律走 Next.js rewrites（`/api/*` → `http://fastapi:8000/api/*`），不直接打 `FASTAPI_SERVICE_URL`
- Auth（登入、JWT）由 FastAPI 管；Next.js 側無 NextAuth
- FastAPI 直接用 SQLAlchemy 管理 DB（`Base.metadata.create_all`），目前無 Alembic migration；schema 變更需手動處理或重建 DB
- Server Actions 預設；有 streaming（SSE chat）才用 API Route
- vLLM 和 vllm-embedding 透過 Docker profile 控制，`make up` 不啟動，`make up-vllm` 才啟動

## 安全規則

- production compose 所有 service 不暴露 ports；對外只走 Cloudflare Tunnel
- dev compose 可暴露 `app:3000`，其他 service 不暴露
- .env 不 commit；secret 欄位用 `openssl rand -base64 32` 生成
- security headers 已在 next.config.ts 啟用（production only）
- FastAPI CORS 目前設定為 `allow_origins=["*"]`，production 前應收緊

## 環境

- Node 22，Python（pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime）
- PostgreSQL 16
- GPU：nvidia，用於 fastapi-service（CV 推理）及 vllm / vllm-embedding（LLM 推理）
