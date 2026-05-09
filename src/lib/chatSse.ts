export type ChatStatusPhase = "thinking" | "retrieving" | "tool" | "finalizing";

export type ChatStatusPayload = {
  phase: ChatStatusPhase;
  text: string;
  rally_id?: number;
};

export type ChatMessagePayload = {
  delta: string;
};

export type ChatErrorPayload = {
  message: string;
};

export type ChatStreamEvent =
  | { type: "status"; payload: ChatStatusPayload }
  | { type: "message"; payload: ChatMessagePayload }
  | { type: "error"; payload: ChatErrorPayload }
  | { type: "done"; payload: Record<string, never> };

export type ChatStreamUiState = {
  outputStarted: boolean;
  fullText: string;
  assistantText: string;
  streamErrored: boolean;
  doneReceived: boolean;
  isStreamingStatus: boolean;
  statusPhase: ChatStatusPhase | null;
};

export const INITIAL_CHAT_STREAM_UI_STATE: ChatStreamUiState = {
  outputStarted: false,
  fullText: "",
  assistantText: "",
  streamErrored: false,
  doneReceived: false,
  isStreamingStatus: false,
  statusPhase: null,
};

export function parseChatSseFrame(frame: string): ChatStreamEvent | null {
  let eventName = "message";
  const dataLines: string[] = [];

  for (const rawLine of frame.split("\n")) {
    const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
    if (!line) continue;

    if (line.startsWith("event:")) {
      eventName = line.slice("event:".length).trim();
      continue;
    }

    if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trimStart());
    }
  }

  if (!dataLines.length) return null;

  let payload: unknown;
  try {
    payload = JSON.parse(dataLines.join("\n"));
  } catch {
    return null;
  }

  switch (eventName) {
    case "status":
      if (
        payload &&
        typeof payload === "object" &&
        typeof (payload as ChatStatusPayload).phase === "string" &&
        typeof (payload as ChatStatusPayload).text === "string"
      ) {
        return { type: "status", payload: payload as ChatStatusPayload };
      }
      return null;

    case "message":
      if (
        payload &&
        typeof payload === "object" &&
        typeof (payload as ChatMessagePayload).delta === "string"
      ) {
        return { type: "message", payload: payload as ChatMessagePayload };
      }
      return null;

    case "error":
      if (
        payload &&
        typeof payload === "object" &&
        typeof (payload as ChatErrorPayload).message === "string"
      ) {
        return { type: "error", payload: payload as ChatErrorPayload };
      }
      return null;

    case "done":
      return { type: "done", payload: {} };

    default:
      return null;
  }
}

export function createChatSseParser(
  onEvent: (event: ChatStreamEvent) => void
) {
  let buffer = "";

  const flushFrames = () => {
    while (true) {
      const boundary = buffer.indexOf("\n\n");
      if (boundary === -1) break;

      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);

      const event = parseChatSseFrame(frame);
      if (event) onEvent(event);
    }
  };

  return {
    push(chunk: string) {
      if (!chunk) return;
      buffer += chunk;
      flushFrames();
    },

    finish() {
      const tail = buffer.trim();
      buffer = "";
      if (!tail) return;

      const event = parseChatSseFrame(tail);
      if (event) onEvent(event);
    },
  };
}

export function reduceChatStreamUiState(
  state: ChatStreamUiState,
  event: ChatStreamEvent
): ChatStreamUiState {
  if (event.type === "status") {
    if (state.outputStarted) return state;
    return {
      ...state,
      assistantText: event.payload.text,
      isStreamingStatus: true,
      statusPhase: event.payload.phase,
    };
  }

  if (event.type === "message") {
    const fullText = state.fullText + event.payload.delta.replace(/\u200b/g, "");
    return {
      ...state,
      outputStarted: true,
      fullText,
      assistantText: fullText,
      isStreamingStatus: false,
      statusPhase: null,
    };
  }

  if (event.type === "error") {
    return {
      ...state,
      streamErrored: true,
      assistantText: "錯誤：" + event.payload.message,
      isStreamingStatus: false,
      statusPhase: null,
    };
  }

  return {
    ...state,
    doneReceived: true,
    isStreamingStatus: false,
    statusPhase: null,
  };
}
