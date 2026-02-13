// static/video.js － Chunk Upload（可併發）+ 平滑上傳進度 + 背景任務 + 輪詢進度條 + 完成後變下載按鈕

export function initVideo({ state, dom, utils }) {
  const { appendMessage, showError, showInfo, setBusy } = utils;

  // ------- 時間換算工具 -------
  function formatDuration(seconds) {
    if (
      seconds == null ||
      typeof seconds !== "number" ||
      !isFinite(seconds) ||
      seconds <= 0
    ) {
      return null;
    }

    const total = Math.floor(seconds);
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;

    if (h > 0) {
      return `${h}時${m}分${s}秒`;
    }
    if (m > 0) {
      return `${m}分${s}秒`;
    }
    return `${s}秒`;
  }

  // ------- 進度條工具 -------

  function setProgress(pct) {
    const clamped = Math.max(0, Math.min(100, pct || 0));
    dom.progressBar.style.width = clamped + "%";

    if (clamped > 0 && clamped <= 100) {
      dom.progressContainer.style.display = "block";
    } else {
      dom.progressContainer.style.display = "none";
    }
  }

  // ------- XHR 上傳單一 chunk（有 upload progress） -------

  function uploadChunkXHR({ uploadId, index, totalChunks, blob, onProgress }) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const url =
        `/upload_chunk?upload_id=${encodeURIComponent(uploadId)}` +
        `&index=${index}&total=${totalChunks}`;

      xhr.open("POST", url, true);

      xhr.upload.onprogress = (evt) => {
        // 某些情況 lengthComputable 會 false；用 blob.size 當 total 也能算得很準
        const total = evt.lengthComputable ? evt.total : blob.size;
        const loaded = evt.loaded || 0;
        onProgress?.(Math.min(loaded, total), total);
      };

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(true);
        } else {
          reject(
            new Error(
              xhr.responseText ||
              `chunk ${index} 上傳失敗（HTTP ${xhr.status}）`
            )
          );
        }
      };

      xhr.onerror = () => reject(new Error(`chunk ${index} 網路錯誤`));

      const fd = new FormData();
      fd.append("chunk", blob);
      xhr.send(fd);
    });
  }

  // ------- Chunk Upload（併發 + 平滑總進度） -------

  async function uploadInChunksSmooth(
    file,
    { concurrency = 3, chunkSize = 10 * 1024 * 1024 } = {},
    onProgress
  ) {
    const totalSize = file.size;
    const totalChunks = Math.ceil(totalSize / chunkSize);

    const uploadId =
      crypto && crypto.randomUUID
        ? crypto.randomUUID()
        : `${Date.now()}_${Math.random()}`;

    // 每塊目前已上傳 bytes（用來算總進度）
    const uploadedByChunk = new Array(totalChunks).fill(0);

    function reportProgress() {
      const uploadedTotal = uploadedByChunk.reduce((a, b) => a + b, 0);
      const pct = Math.round((uploadedTotal / totalSize) * 100);
      onProgress?.(pct);
    }

    // 建立每個 chunk 的上傳任務
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

    const n = Math.max(1, Math.min(concurrency, tasks.length));
    await Promise.all(Array.from({ length: n }, () => worker()));

    // 上傳完成 → 通知後端合併 + 建立 session
    const completeRes = await fetch("/upload_complete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ upload_id: uploadId, filename: file.name }),
    });

    if (!completeRes.ok) {
      const text = await completeRes.text().catch(() => "");
      throw new Error(text || `完成上傳失敗（HTTP ${completeRes.status}）`);
    }

    return await completeRes.json();
  }

  // ------- /status 輪詢 -------
  // ------- 404 過多視為伺服器失效：彈窗問要不要重整 -------

  // 可調參數
  const STATUS_404_THRESHOLD = 6;     // 連續 404 幾次算掛掉
  const STATUS_WINDOW_MS = 15_000;    // 或者在這個時間窗內累積（這裡用連續為主）
  const ALERT_COOLDOWN_MS = 30_000;   // 彈窗後至少等多久才可能再彈

  // 記錄狀態
  state.status404Count = 0;
  state.status404FirstAt = 0;
  state.serverDownAlertAt = 0;
  state.serverDownAlerting = false;

  function maybeAlertServerDown(reasonText = "伺服器似乎失效（狀態查詢多次 404）。") {
    const now = Date.now();

    // 避免重複彈窗
    if (state.serverDownAlerting) return;
    if (state.serverDownAlertAt && now - state.serverDownAlertAt < ALERT_COOLDOWN_MS) return;

    state.serverDownAlerting = true;
    state.serverDownAlertAt = now;

    // 停掉輪詢，避免繼續打爆
    if (state.pollInterval) {
      clearInterval(state.pollInterval);
      state.pollInterval = null;
    }

    setBusy(false);
    setProgress(0);
    dom.statusEl.textContent = "伺服器連線異常（/status 404）";

    const ok = window.confirm(`${reasonText}\n\n要重整頁面再試一次嗎？`);
    state.serverDownAlerting = false;

    if (ok) {
      window.location.reload();
    }
  }


  async function fetchStatusOnce() {
    if (!state.sessionId) return;

    try {
      const res = await fetch(`/status/${state.sessionId}`);

      // --- 針對 404 做保護 ---
      if (res.status === 404) {
        const now = Date.now();

        // 初始化時間窗
        if (!state.status404FirstAt) state.status404FirstAt = now;

        // 超過時間窗就重算（避免很久以前的 404 影響現在）
        if (now - state.status404FirstAt > STATUS_WINDOW_MS) {
          state.status404FirstAt = now;
          state.status404Count = 0;
        }

        state.status404Count += 1;

        // 你也可以顯示一下目前連續 404 次數（選用）
        dom.statusEl.textContent = `伺服器狀態查詢異常（404）... (${state.status404Count}/${STATUS_404_THRESHOLD})`;

        if (state.status404Count >= STATUS_404_THRESHOLD) {
          maybeAlertServerDown();
        }
        return; // 404 就先不往下走
      }

      // 非 404，代表狀態路由有回應 → 清掉 404 計數
      state.status404Count = 0;
      state.status404FirstAt = 0;

      if (!res.ok) return;

      const data = await res.json();

      const p = data.progress ?? 0;
      setProgress(p);

      if (data.status === "processing") {
        dom.statusEl.textContent = `YOLO 分析中... ${p}%`;
      }

      if (data.status === "failed") {
        dom.statusEl.textContent = "分析失敗";
        if (data.error) showError("YOLO 分析失敗：" + data.error);

        if (state.pollInterval) {
          clearInterval(state.pollInterval);
          state.pollInterval = null;
        }

        setBusy(false);
        return;
      }

      if (data.status === "completed") {
        if (state.pollInterval) {
          clearInterval(state.pollInterval);
          state.pollInterval = null;
        }

        setProgress(100);

        if (data.yolo_video_url) {
          if (state.yoloVideoUrl !== data.yolo_video_url) {
            state.yoloVideoUrl = data.yolo_video_url;
            dom.videoEl.src = data.yolo_video_url;
            dom.videoEl.style.display = "block";
            dom.placeholderEl.style.display = "none";
            dom.videoEl.play().catch(() => { });
          }

          dom.statusEl.textContent = "分析完成（後端已畫好標註）";
          showInfo("YOLO 分析完成，現在播放的是標註影片！");

          dom.analyzeBtn.textContent = "下載分析後影片";
          dom.analyzeBtn.disabled = false;

          dom.analyzeBtn.onclick = () => {
            if (!state.yoloVideoUrl) return;
            const a = document.createElement("a");
            a.href = state.yoloVideoUrl;
            const baseName = state.filename || "video.mp4";
            a.download = "analyzed_" + baseName;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
          };
        }

        setBusy(false);

        setTimeout(() => {
          dom.progressContainer.style.display = "none";
        }, 1500);
      }
    } catch (e) {
      console.error("查詢狀態失敗", e);
    }
  }


  function startStatusPolling() {
    if (!state.sessionId) return;
    if (state.pollInterval) clearInterval(state.pollInterval);
    fetchStatusOnce();
    state.pollInterval = setInterval(fetchStatusOnce, 1000);
  }

  // -------- 上傳 (本地預覽) --------

  function triggerFileSelect() {
    if (state.isBusy) return;
    dom.fileEl.click();
  }

  dom.videoUploadBtn.onclick = triggerFileSelect;
  dom.placeholderEl.onclick = triggerFileSelect;

  dom.fileEl.onchange = () => {
    if (dom.fileEl.files?.[0]) handleUpload(dom.fileEl.files[0]);
  };

  async function handleUpload(file) {
    if (!file.type.startsWith("video/")) {
      showError("請上傳影片檔案");
      return;
    }

    setBusy(true);
    dom.statusEl.textContent = "影片上傳中... 0%";
    setProgress(1);

    try {
      // (1) 本地 blob 預覽
      if (state.localVideoUrl) URL.revokeObjectURL(state.localVideoUrl);
      state.localVideoUrl = URL.createObjectURL(file);
      dom.videoEl.src = state.localVideoUrl;
      dom.videoEl.style.display = "block";
      dom.placeholderEl.style.display = "none";

      // (2) Chunk Upload（併發 + 平滑進度）
      // 你可以調整 concurrency / chunkSize
      const data = await uploadInChunksSmooth(
        file,
        { concurrency: 3, chunkSize: 10 * 1024 * 1024 },
        (pct) => {
          setProgress(pct);
          dom.statusEl.textContent = `影片上傳中... ${pct}%`;
        }
      );

      if (!data || !data.ok) throw new Error(data?.error || "上傳失敗");

      state.sessionId = data.session_id;
      state.videoMeta = data.meta;
      state.filename = data.filename;

      const m = data.meta || {};
      const lines = [];
      lines.push(`檔名：${data.filename}`);
      lines.push(`解析度：${m.width ?? "?"} x ${m.height ?? "?"}`);
      lines.push(`FPS：${m.fps ?? "?"}`);

      const durationText = formatDuration(m.duration);
      if (durationText) {
        lines.push(`時長：${durationText}`);
      }

      dom.videoInfoEl.textContent = lines.join("\n");


      setProgress(100);
      dom.statusEl.textContent = "影片上傳完成，可預覽或開始 YOLO 分析";
      dom.analyzeBtn.disabled = false;

      appendMessage("assistant", "影片上傳完成，可以開始分析囉！");

      // 上傳完成後稍微停一下再收起進度條
      setTimeout(() => setProgress(0), 800);
    } catch (err) {
      showError("上傳失敗：" + (err?.message || String(err)));
      setProgress(0);
    } finally {
      setBusy(false);
    }
  }

  // -------- YOLO 分析：啟動背景任務 + 輪詢 /status --------

  dom.analyzeBtn.onclick = async () => {
    if (!state.sessionId) {
      showError("請先上傳影片");
      return;
    }

    setBusy(true);
    dom.statusEl.textContent = "分析任務啟動中...";
    setProgress(0);

    try {
      const payload = { session_id: state.sessionId };
      try {
        const dur = state.videoMeta && state.videoMeta.duration;
        if (dur && typeof dur === "number" && isFinite(dur) && dur > 0) {
          payload.max_seconds = Math.ceil(dur);
        }
      } catch (e) { }

      const res = await fetch("/analyze_yolo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "啟動 YOLO 分析失敗");
      }

      const data = await res.json();
      if (!data.ok) throw new Error(data.error || "啟動 YOLO 分析失敗");

      dom.statusEl.textContent = "YOLO 分析已啟動，伺服器正在處理...";
      startStatusPolling();
      // 不要 setBusy(false)，等 completed/failed 再解鎖
    } catch (err) {
      showError("YOLO 分析啟動失敗：" + (err?.message || String(err)));
      setProgress(0);
      setBusy(false);
    }
  };

  // -------- Reset --------
  dom.resetBtn.onclick = () => window.location.reload();

  // -------- 離開頁面清理 --------
  window.addEventListener("beforeunload", () => {
    if (state.localVideoUrl) URL.revokeObjectURL(state.localVideoUrl);
    if (state.pollInterval) {
      clearInterval(state.pollInterval);
      state.pollInterval = null;
    }
  });
}
