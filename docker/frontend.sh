#!/bin/sh
set -eu

if [ "${FRONTEND_DEV_MODE:-true}" = "true" ]; then
  npm ci
  npm run dev -- --hostname 0.0.0.0 --port 3000
else
  npm ci
  npm run build
  cp -r .next/static .next/standalone/.next/static
  cp -r public .next/standalone/public
  node .next/standalone/server.js
fi
