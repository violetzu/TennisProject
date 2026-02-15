// hooks/useVideoUpload.ts
"use client";

import { authFetch } from "@/lib/authFetch";
import { getToken } from "@/lib/auth";

export type UploadMeta = {
  width?: number;
  height?: number;
  fps?: number;
  duration?: number;
};

type UploadCompleteResp = {
  ok: boolean;
  session_id: string;
  meta?: UploadMeta;
  filename?: string;
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
      const loaded = evt.loaded || 0;
      onProgress?.(Math.min(loaded, total), total);
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
    crypto && "randomUUID" in crypto && typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `${Date.now()}_${Math.random()}`;

  // 每塊目前已上傳 bytes（用來算總進度）
  const uploadedByChunk = new Array<number>(totalChunks).fill(0);

  function reportProgress() {
    const uploadedTotal = uploadedByChunk.reduce((a, b) => a + b, 0);
    const pct = Math.round((uploadedTotal / totalSize) * 100);
    onProgress?.(pct);
  }

  const tasks = Array.from({ length: totalChunks }, (_, index) => {
    const start = index * chunkSize;
    const end = Math.min(totalSize, start + chunkSize);
    const blob = file.slice(start, end);

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

      // 完成時保底設滿
      uploadedByChunk[index] = blob.size;
      reportProgress();
    };
  });

  // 併發 worker pool
  let cursor = 0;
  async function worker() {
    while (cursor < tasks.length) {
      const i = cursor++;
      await tasks[i]();
    }
  }

  const n = Math.min(concurrency, tasks.length);
  await Promise.all(Array.from({ length: n }, () => worker()));

  // 上傳完成 → 通知後端合併 + 建立 session
  const completeRes = await authFetch("/api/upload_complete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ upload_id: uploadId, filename: file.name }),
  });

  if (!completeRes.ok) {
    const text = await completeRes.text().catch(() => "");
    throw new Error(text || `完成上傳失敗（HTTP ${completeRes.status}）`);
  }

  return (await completeRes.json()) as UploadCompleteResp;
}
