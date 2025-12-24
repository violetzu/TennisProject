// static/chat.js

export function initChat({ state, dom, utils }) {
  const { appendMessage, showError } = utils;

  let thinkingTimer = null;

  function startThinking(bubble) {
    let dots = 1;
    bubble.textContent = "思考中.";
    thinkingTimer = setInterval(() => {
      dots = (dots % 3) + 1;
      bubble.textContent = "思考中" + ".".repeat(dots);
    }, 500);
  }

  function stopThinking() {
    if (thinkingTimer) {
      clearInterval(thinkingTimer);
      thinkingTimer = null;
    }
  }

  function setChatBusy(isBusy) {
    dom.sendBtn.disabled = isBusy || !state.sessionId;
    dom.queryEl.disabled = isBusy;
  }

  async function sendQuestion() {
    if (!state.sessionId) {
      showError("請先上傳影片再提問");
      return;
    }

    const text = dom.queryEl.value.trim();
    if (!text) return;

    appendMessage("user", text);
    dom.queryEl.value = "";

    const thinkingBubble = appendMessage("assistant", "");
    startThinking(thinkingBubble);
    setChatBusy(true);

    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.sessionId, question: text }),
      });

      if (!res.ok || !res.body) {
        stopThinking();
        thinkingBubble.textContent =
          "錯誤：" + ((await res.text()) || res.statusText);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let full = "";
      let gotChunk = false;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        if (value) {
          if (!gotChunk) {
            gotChunk = true;
            stopThinking();
            thinkingBubble.textContent = "";
          }
          const chunkText = decoder.decode(value, { stream: true });
          full += chunkText;
          thinkingBubble.append(document.createTextNode(chunkText));

          if (state.autoScroll) {
            dom.chatEl.scrollTop = dom.chatEl.scrollHeight;
          }
        }
      }

      if (!gotChunk) {
        stopThinking();
        thinkingBubble.textContent = "(無回應內容)";
      }
    } catch (err) {
      stopThinking();
      thinkingBubble.textContent = "連線失敗：" + err.message;
    } finally {
      setChatBusy(false);
    }
  }

  dom.sendBtn.onclick = sendQuestion;

  dom.queryEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendQuestion();
    }
  });
  setChatBusy(false);
}
