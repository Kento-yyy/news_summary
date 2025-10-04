import time, requests, json

url = "http://127.0.0.1:11434/api/generate" #ollama
data = {"model": "gpt-oss:20b", "prompt": "Describe the architecture of GPUs in 500 words."}

start = time.time()
response = requests.post(url, json=data, stream=True)

token_count = 0
for line in response.iter_lines():
    if line:
        token_count += 1
end = time.time()

print(f"Tokens: {token_count}, Time: {end - start:.2f}s, Speed: {token_count/(end - start):.2f} tokens/sec")
