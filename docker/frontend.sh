#!/bin/sh
set -eu

if [ "${FRONTEND_DEV_MODE:-true}" = "true" ]; then
  npm ci --include=dev
  npm run dev -- --hostname 0.0.0.0 --port 3000
else
  npm ci --include=dev
  npm run build
  npm run start -- --hostname 0.0.0.0 --port 3000
fi
