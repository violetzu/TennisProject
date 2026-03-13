# ---- Build Stage ----
FROM node:20-bookworm-slim AS builder
WORKDIR /app

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ .
RUN npm run build

# ---- Runtime Stage ----
FROM node:20-bookworm-slim AS runner
WORKDIR /app

ENV NODE_ENV=production

# 只複製必要檔案，縮小最終映像
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000
CMD ["npm", "run", "start", "--", "--hostname", "0.0.0.0", "--port", "3000"]