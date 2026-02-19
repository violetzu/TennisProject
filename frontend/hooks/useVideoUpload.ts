// hooks/useVideoUpload.ts
"use client";

import { apiFetch } from "@/lib/apiFetch";
import { getToken } from "@/lib/auth";

export type UploadMeta = {
  width?: number;
  height?: number;
  fps?: number;
  frame_count?: number;
  duration?: number;
};

export type UploadCompleteResp = {
  ok: boolean;
  session_id: string;
  analysis_record_id: number;
  guest_token: string | null;
  filename: string;
  meta: UploadMeta;
  video_url: string;
  mode: "user" | "guest";
  error?: string;
};

function uploadChunkXHR(args: {
  uploadId: string;
  index: number;
  totalChunks: number;
  blob: Blob;
  onProgress?: (loaded: number, total: number) => void;
}) {
  const { uploadId, index, totalChunks, blob, onProgress } = args;

  return new Promise<void>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const url =
      `/api/upload_chunk?upload_id=${encodeURIComponent(uploadId)}` +
      `&index=${index}&total=${totalChunks}`;

    xhr.open("POST", url, true);

    const token = getToken();
    if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);

    xhr.upload.onprogress = (evt) => {
      const total = evt.lengthComputable ? evt.total : blob.size;
      onProgress?.(Math.min(evt.loaded, total), total);
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve();
      else reject(new Error(xhr.responseText || `chunk ${index} 上傳失敗（HTTP ${xhr.status}）`));
    };

    xhr.onerror = () => reject(new Error(`chunk ${index} 網路錯誤`));

    const fd = new FormData();
    fd.append("chunk", blob);
    xhr.send(fd);
  });
}

export async function uploadInChunksSmooth(
  file: File,
  opts: { concurrency?: number; chunkSize?: number } = {},
  onProgress?: (pct: number) => void
): Promise<UploadCompleteResp> {
  const concurrency = Math.max(1, Math.min(opts.concurrency ?? 3, 12));
  const chunkSize = opts.chunkSize ?? 10 * 1024 * 1024;

  const totalSize = file.size;
  const totalChunks = Math.ceil(totalSize / chunkSize);

  const uploadId =
    typeof crypto?.randomUUID === "function"
      ? crypto.randomUUID()
      : `${Date.now()}_${Math.random()}`;

  const uploadedByChunk = new Array<number>(totalChunks).fill(0);

  function reportProgress() {
    const total = uploadedByChunk.reduce((a, b) => a + b, 0);
    onProgress?.(Math.round((total / totalSize) * 100));
  }

  const tasks = Array.from({ length: totalChunks }, (_, index) => {
    const start = index * chunkSize;
    const blob = file.slice(start, Math.min(totalSize, start + chunkSize));

    return async () => {
      await uploadChunkXHR({
        uploadId,
        index,
        totalChunks,
        blob,
        onProgress: (loaded) => {
          uploadedByChunk[index] = Math.min(loaded, blob.size);
          reportProgress();
        },
      });
      uploadedByChunk[index] = blob.size;
      reportProgress();
    };
  });

  // 並發 worker pool
  let cursor = 0;
  async function worker() {
    while (cursor < tasks.length) {
      await tasks[cursor++]();
    }
  }
  await Promise.all(
    Array.from({ length: Math.min(concurrency, tasks.length) }, () => worker())
  );

  // 上傳完成 → 通知後端合併
  const res = await apiFetch("/api/upload_complete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ upload_id: uploadId, filename: file.name }),
  });

  return res.json() as Promise<UploadCompleteResp>;
}
