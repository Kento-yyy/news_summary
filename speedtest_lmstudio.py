import time, requests, json

url = "http://127.0.0.1:1234/v1/completions"
data = {
    "model": "openai/gpt-oss-20b",
    "prompt": "Describe the architecture of GPUs in 200 words.",
    "stream": True
}

start = time.time()
response = requests.post(url, json=data, stream=True)

token_count = 0
output = []

for line in response.iter_lines():
    if not line:
        continue
    decoded = line.decode("utf-8").strip()
    if decoded.startswith("data: "):
        payload = decoded[6:]
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
            text = obj["choices"][0].get("text", "")
            if text:
                print(text, end="", flush=True)
                output.append(text)
                token_count += len(text.split())  # 単語数をトークン近似
        except json.JSONDecodeError:
            continue

end = time.time()
elapsed = end - start
print(f"\n\nTokens (approx): {token_count}, Time: {elapsed:.2f}s, Speed: {token_count/elapsed:.2f} tokens/sec")
