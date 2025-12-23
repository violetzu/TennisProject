// static/video.js － 使用背景任務 + 輪詢進度條 + 完成後變下載按鈕 + 上傳進度

export function initVideo({ state, dom, utils }) {
  const { appendMessage, showError, showInfo, setBusy } = utils;

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

  // ------- 上傳（帶進度）工具 -------

  function uploadFileWithProgress(url, formData, onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", url, true);

      xhr.upload.onprogress = (evt) => {
        if (!evt.lengthComputable) return;
        const pct = Math.round((evt.loaded / evt.total) * 100);
        onProgress?.(pct);
      };

      xhr.onload = () => {
        try {
          const data = JSON.parse(xhr.responseText || "{}");
          if (xhr.status >= 200 && xhr.status < 300) resolve(data);
          else reject(new Error(data.error || xhr.responseText || "上傳失敗"));
        } catch (e) {
          reject(new Error(xhr.responseText || "上傳失敗（非 JSON 回應）"));
        }
      };

      xhr.onerror = () => reject(new Error("網路錯誤，上傳失敗"));
      xhr.send(formData);
    });
  }

  // ------- /status 輪詢 -------

  async function fetchStatusOnce() {
    if (!state.sessionId) return;
    try {
      const res = await fetch(`/status/${state.sessionId}`);
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
            dom.videoEl.play();
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

      // (2) 上傳給後端（帶進度）
      const fd = new FormData();
      fd.append("file", file);

      const data = await uploadFileWithProgress("/upload", fd, (pct) => {
        setProgress(pct);
        dom.statusEl.textContent = `影片上傳中... ${pct}%`;
      });

      if (!data.ok) throw new Error(data.error || "上傳失敗");

      state.sessionId = data.session_id;
      state.videoMeta = data.meta;
      state.filename = data.filename;

      const m = data.meta;
      dom.videoInfoEl.textContent =
        `檔名：${data.filename}\n` +
        `解析度：${m.width} x ${m.height}\n` +
        `FPS：${m.fps}\n` +
        `時長：${m.duration?.toFixed(2)}`;

      setProgress(100);
      dom.statusEl.textContent = "影片上傳完成，可預覽或開始 YOLO 分析";
      dom.analyzeBtn.disabled = false;

      appendMessage("assistant", "影片上傳完成，可以開始分析囉！");

      // 上傳完成後稍微停一下再收起進度條
      setTimeout(() => setProgress(0), 800);
    } catch (err) {
      showError("上傳失敗：" + err.message);
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
      } catch (e) {}

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
      showError("YOLO 分析啟動失敗：" + err.message);
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
