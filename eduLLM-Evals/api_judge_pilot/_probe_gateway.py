import time
from pathlib import Path

import httpx

env = {}
for line in Path("eduLLM-Evals/.env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"')

base = env["MODEL_API_BASE"].rstrip("/")
key = env.get("MODEL_API_KEY", "")
headers = {"Content-Type": "application/json"}
if key:
    headers["Authorization"] = f"Bearer {key}"

payload = {
    "model": "claude-group/claude-haiku-4-5",
    "messages": [{"role": "user", "content": "reply with the single word OK"}],
    "max_tokens": 10,
    "temperature": 0,
}

for i in range(4):
    t = time.time()
    try:
        r = httpx.post(f"{base}/chat/completions", headers=headers, timeout=60, json=payload)
        body = r.text[:180].replace("\n", " ")
        print(f"probe{i}: HTTP {r.status_code} in {time.time() - t:.1f}s :: {body}")
    except Exception as e:
        print(f"probe{i}: EXC {type(e).__name__}: {e}")
    time.sleep(1)
