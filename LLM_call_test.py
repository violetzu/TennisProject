import json
import requests
from dotenv import load_dotenv
import os
load_dotenv()

API_URL = "https://qwen3vl.marimo.idv.tw/chat-vl"
API_KEY = os.getenv("API_KEY", None)

headers = {
    "X-API-Key": API_KEY,
}

messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a video analysis assistant."}
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "file"},
            {"type": "text", "text": "幫我描述這支影片內容。"}
        ]
    }
]

data = {
    "messages": json.dumps(messages, ensure_ascii=False),
    "max_new_tokens": "2048"
}

files = {"file": open("test.mp4", "rb")}

resp = requests.post(API_URL, headers=headers, data=data, files=files)
print(resp.status_code, resp.text)
