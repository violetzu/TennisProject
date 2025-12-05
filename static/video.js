// static/video.js － 使用背景任務 + 輪詢進度條 + 完成後變下載按鈕

export function initVideo({ state, dom, utils }) {
    const { appendMessage, showError, showInfo, setBusy } = utils;

    // ------- 進度條工具 -------

    function setProgress(pct) {
        const clamped = Math.max(0, Math.min(100, pct || 0));
        dom.progressBar.style.width = clamped + "%";

        if (clamped > 0 && clamped <= 100) {
            // 一旦有進度就顯示進度條（配合你的 CSS 的 display:none）
            dom.progressContainer.style.display = "block";
        } else {
            // 進度歸零時就藏起來
            dom.progressContainer.style.display = "none";
        }
    }

    async function fetchStatusOnce() {
        if (!state.sessionId) return;
        try {
            const res = await fetch(`/status/${state.sessionId}`);
            if (!res.ok) return;
            const data = await res.json();

            const p = data.progress ?? 0;
            setProgress(p);

            // 顯示百分比文字
            if (data.status === "processing") {
                dom.statusEl.textContent = `YOLO 分析中... ${p}%`;
            }

            if (data.status === "failed") {
                dom.statusEl.textContent = "分析失敗";
                if (data.error) {
                    showError("YOLO 分析失敗：" + data.error);
                }
                if (state.pollInterval) {
                    clearInterval(state.pollInterval);
                    state.pollInterval = null;
                }
                // 分析失敗 → 解鎖按鈕
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
                    // 設定影片播放
                    if (state.yoloVideoUrl !== data.yolo_video_url) {
                        state.yoloVideoUrl = data.yolo_video_url;
                        dom.videoEl.src = data.yolo_video_url;
                        dom.videoEl.style.display = "block";
                        dom.placeholderEl.style.display = "none";
                        dom.videoEl.play();
                    }

                    dom.statusEl.textContent = "分析完成（後端已畫好標註）";
                    showInfo("YOLO 分析完成，現在播放的是標註影片！");

                    // 把 YOLO 按鈕改成「下載分析後影片」
                    dom.analyzeBtn.textContent = "下載分析後影片";
                    dom.analyzeBtn.disabled = false;

                    dom.analyzeBtn.onclick = () => {
                        if (!state.yoloVideoUrl) return;
                        const a = document.createElement("a");
                        a.href = state.yoloVideoUrl;
                        // 用原始檔名組一個下載名
                        const baseName = state.filename || "video.mp4";
                        a.download = "analyzed_" + baseName;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                    };
                }

                // 分析完成 → 解鎖按鈕
                setBusy(false);

                // 如果你想讓條停在那邊，不要隱藏可以註解掉下面這段
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
        if (state.pollInterval) {
            clearInterval(state.pollInterval);
        }
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
        if (dom.fileEl.files?.[0]) {
            handleUpload(dom.fileEl.files[0]);
        }
    };

    async function handleUpload(file) {
        if (!file.type.startsWith("video/")) {
            showError("請上傳影片檔案");
            return;
        }

        setBusy(true);
        dom.statusEl.textContent = "影片上傳中...";

        try {
            // (1) 本地 blob 預覽
            if (state.localVideoUrl) URL.revokeObjectURL(state.localVideoUrl);
            state.localVideoUrl = URL.createObjectURL(file);
            dom.videoEl.src = state.localVideoUrl;
            dom.videoEl.style.display = "block";
            dom.placeholderEl.style.display = "none";

            // (2) 上傳給後端
            const fd = new FormData();
            fd.append("file", file);

            const res = await fetch("/upload", {
                method: "POST",
                body: fd,
            });

            const data = await res.json();
            if (!data.ok) throw new Error(data.error || "上傳失敗");

            state.sessionId = data.session_id;
            state.videoMeta = data.meta;
            state.filename = data.filename; // ★ 存起來，下載時用

            const m = data.meta;
            dom.videoInfoEl.textContent =
                `檔名：${data.filename}\n` +
                `解析度：${m.width} x ${m.height}\n` +
                `FPS：${m.fps}\n` +
                `時長：${m.duration?.toFixed(2)}`;

            dom.statusEl.textContent = "影片上傳完成，可預覽或開始 YOLO 分析";
            dom.analyzeBtn.disabled = false;

            setProgress(0);

            appendMessage("assistant", "影片上傳完成，可以開始分析囉！");
        } catch (err) {
            showError("上傳失敗：" + err.message);
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
            const res = await fetch("/analyze_yolo", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    session_id: state.sessionId,
                    max_frames: 1000
                }),
            });

            if (!res.ok) {
                const text = await res.text();
                throw new Error(text || "啟動 YOLO 分析失敗");
            }

            const data = await res.json();
            if (!data.ok) {
                throw new Error(data.error || "啟動 YOLO 分析失敗");
            }

            dom.statusEl.textContent = "YOLO 分析已啟動，伺服器正在處理...";

            // 分析在背景跑，這裡開始輪詢進度
            startStatusPolling();

            // 注意：這裡「不要」馬上 setBusy(false)
            // 要等分析真正完成，在 fetchStatusOnce 裡處理
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
