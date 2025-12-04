// static/video.js

export function initVideo({ state, dom, utils }) {
    const { appendMessage, showError, showInfo, setBusy, updateButtons } = utils;

    const ctx = dom.canvasEl.getContext("2d");

    const SKELETON_LINKS = [
        [5, 6], [5, 11], [6, 12], [11, 12],
        [5, 7], [7, 9], [6, 8], [8, 10],
        [11, 13], [13, 15], [12, 14], [14, 16],
    ];

    // ========= Frame maps =========
    function buildFrameMaps() {
        state.tracksByFrame = {};
        state.ballTracks.forEach((b) => {
            const f = b.frame;
            if (!state.tracksByFrame[f]) state.tracksByFrame[f] = [];
            state.tracksByFrame[f].push(b);
        });

        state.posesByFrame = {};
        state.poses.forEach((p) => {
            const f = p.frame;
            if (!state.posesByFrame[f]) state.posesByFrame[f] = [];
            state.posesByFrame[f].push(p);
        });
    }

    function syncCanvasSize() {
        if (!dom.videoEl.videoWidth || !dom.videoEl.videoHeight) return;
        dom.canvasEl.width = dom.videoEl.videoWidth;
        dom.canvasEl.height = dom.videoEl.videoHeight;
        drawOverlay();
    }

    function drawOverlay() {
        if (!ctx) return;

        const vw = dom.videoEl.videoWidth;
        const vh = dom.videoEl.videoHeight;
        const cw = dom.wrapperEl.clientWidth;
        const ch = dom.wrapperEl.clientHeight;

        ctx.clearRect(0, 0, dom.canvasEl.width, dom.canvasEl.height);
        if (!state.detectionReady || !vw || !vh || !cw || !ch) return;

        const videoRatio = vw / vh;
        const containerRatio = cw / ch;

        let renderW, renderH;
        if (videoRatio > containerRatio) {
            renderW = cw;
            renderH = cw / videoRatio;
        } else {
            renderH = ch;
            renderW = ch * videoRatio;
        }

        const xOffset = (cw - renderW) / 2;
        const yOffset = (ch - renderH) / 2;
        const scale = renderW / vw;

        ctx.save();
        ctx.translate(xOffset, yOffset);
        ctx.scale(scale, scale);

        const fps = (state.videoMeta && state.videoMeta.fps) || 30;
        let frameIdx = Math.round(dom.videoEl.currentTime * fps) + state.frameOffset;
        if (frameIdx < 0) frameIdx = 0;

        const boxes = (state.tracksByFrame && state.tracksByFrame[frameIdx]) || [];
        const framePoses = (state.posesByFrame && state.posesByFrame[frameIdx]) || [];

        ctx.lineWidth = 3 / scale;
        ctx.strokeStyle = "#facc15";

        boxes.forEach((b) => {
            ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
        });

        framePoses.forEach((p) => {
            const kps = p.keypoints || [];
            ctx.fillStyle = "#22c55e";
            ctx.strokeStyle = "#22c55e";
            ctx.lineWidth = 3 / scale;

            kps.forEach(([x, y, conf]) => {
                if (conf < 0.3) return;
                ctx.beginPath();
                ctx.arc(x, y, 6 / scale, 0, Math.PI * 2);
                ctx.fill();
            });

            SKELETON_LINKS.forEach(([i, j]) => {
                const a = kps[i];
                const b = kps[j];
                if (!a || !b) return;
                if (a[2] < 0.3 || b[2] < 0.3) return;
                ctx.beginPath();
                ctx.moveTo(a[0], a[1]);
                ctx.lineTo(b[0], b[1]);
                ctx.stroke();
            });
        });

        ctx.restore();
    }

    function startLoop() {
        if (state.rafId) cancelAnimationFrame(state.rafId);
        function loop() {
            if (!dom.videoEl.paused && !dom.videoEl.ended) {
                drawOverlay();
                state.rafId = requestAnimationFrame(loop);
            }
        }
        loop();
    }

    dom.videoEl.addEventListener("loadedmetadata", syncCanvasSize);
    dom.videoEl.addEventListener("play", startLoop);
    dom.videoEl.addEventListener("pause", () => {
        if (state.rafId) cancelAnimationFrame(state.rafId);
        drawOverlay();
    });

    // ========= ResizeObserver =========
    const ro = new ResizeObserver(() => {
        syncCanvasSize();
    });
    ro.observe(dom.wrapperEl);

    // ========= 上傳 / 拖曳 =========
    function triggerFileSelect() {
        if (state.analysisCompleted) return;
        dom.fileEl.click();
    }

    dom.videoUploadBtn.onclick = triggerFileSelect;
    dom.placeholderEl.onclick = triggerFileSelect;

    dom.fileEl.onchange = () => {
        if (dom.fileEl.files && dom.fileEl.files[0]) {
            handleUpload(dom.fileEl.files[0]);
        }
    };

    ["dragenter", "dragover"].forEach((evtName) => {
        dom.wrapperEl.addEventListener(evtName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (state.analysisCompleted) return;
            dom.wrapperEl.classList.add("drop-active");
        });
    });

    ["dragleave", "drop"].forEach((evtName) => {
        dom.wrapperEl.addEventListener(evtName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dom.wrapperEl.classList.remove("drop-active");
        });
    });

    dom.wrapperEl.addEventListener("drop", (e) => {
        if (state.analysisCompleted) return;
        const dt = e.dataTransfer;
        if (!dt || !dt.files || !dt.files[0]) return;
        handleUpload(dt.files[0]);
    });

    async function handleUpload(file) {
        if (!file) return;

        if (!file.type.startsWith("video/")) {
            showError("請上傳影片檔案（video/*）");
            return;
        }

        const MAX_SIZE = 1024 * 1024 * 500; // 500MB
        if (file.size > MAX_SIZE) {
            showError("檔案太大，請剪短或壓縮影片後再上傳");
            return;
        }

        setBusy(true);
        dom.statusEl.textContent = "影片上傳中...";

        try {
            if (state.localVideoUrl) URL.revokeObjectURL(state.localVideoUrl);
            state.localVideoUrl = URL.createObjectURL(file);
            dom.videoEl.src = state.localVideoUrl;
            dom.placeholderEl.style.display = "none";
            dom.videoEl.style.display = "block";

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
            state.detectionReady = false;
            state.ballTracks = [];
            state.poses = [];
            state.tracksByFrame = {};
            state.posesByFrame = {};
            state.frameOffset = 0;

            dom.analyzeBtn.disabled = false;
            dom.sendBtn.disabled = false;

            const m = data.meta;
            dom.videoInfoEl.textContent =
                `檔名：${data.filename}\n` +
                `解析度：${m.width} x ${m.height}\n` +
                `FPS：${m.fps}\n` +
                `時長：${m.duration.toFixed(2)}s`;

            dom.statusEl.textContent = "影片上傳完成";
            appendMessage("assistant", "影片上傳完成！可以開始進行 YOLO 分析或直接提問。");
            updateButtons();
        } catch (err) {
            showError("上傳失敗：" + err.message);
        } finally {
            setBusy(false);
        }
    }

    // ========= YOLO 分析 =========
    dom.analyzeBtn.onclick = async () => {
        if (!state.sessionId) {
            showError("請先上傳影片再開始 YOLO 分析");
            return;
        }
        if (state.analysisCompleted) return;

        setBusy(true);
        dom.analyzeBtn.disabled = true;
        dom.progressContainer.style.display = "block";
        dom.progressBar.style.width = "0%";
        dom.statusEl.textContent = "準備開始分析...";

        try {
            const res = await fetch("/analyze_yolo", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: state.sessionId, max_frames: 1000 }),
            });

            const data = await res.json();
            if (!data.ok) throw new Error(data.detail || "啟動分析失敗");

            const intervalId = setInterval(async () => {
                if (state.analysisCompleted) return;
                try {
                    const s = await fetch(`/analyze_status/${state.sessionId}`);
                    const st = await s.json();

                    const pct = st.progress || 0;
                    dom.progressBar.style.width = pct + "%";
                    dom.statusEl.textContent = `YOLO 分析中 (${pct}%)`;

                    if (st.status === "completed") {
                        clearInterval(intervalId);
                        state.pollInterval = null;
                        if (state.analysisCompleted) return;
                        state.analysisCompleted = true;

                        state.ballTracks = st.ball_tracks || [];
                        state.poses = st.poses || [];
                        state.detectionReady = true;

                        buildFrameMaps();
                        syncCanvasSize();

                        dom.statusEl.textContent =
                            `分析完成：球軌跡 ${state.ballTracks.length}、姿態 ${state.poses.length}`;
                        dom.progressContainer.style.display = "none";

                        updateButtons();

                        showInfo("YOLO 分析完成，可以針對比賽內容開始提問囉！");
                        setBusy(false);
                    } else if (st.status === "failed") {
                        clearInterval(intervalId);
                        state.pollInterval = null;
                        dom.progressContainer.style.display = "none";
                        showError("YOLO 分析失敗：" + (st.error || "未知錯誤"));
                        setBusy(false);
                    }
                } catch (e) {
                    console.error("輪詢錯誤", e);
                }
            }, 900);
            state.pollInterval = intervalId;
        } catch (err) {
            dom.progressContainer.style.display = "none";
            showError("啟動分析失敗：" + err.message);
            setBusy(false);
        }

    };

    // ========= 頁面關閉時清理資源 =========
    window.addEventListener("beforeunload", () => {
        if (state.pollInterval) clearInterval(state.pollInterval);
        if (state.rafId) cancelAnimationFrame(state.rafId);
        if (state.localVideoUrl) URL.revokeObjectURL(state.localVideoUrl);
    });
}
