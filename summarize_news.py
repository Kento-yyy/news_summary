#!/usr/bin/env python3
"""
Summarize latest news per company using a local LLM.

By default, it fetches news via Google News RSS (same logic as news_latest.py),
then prompts a local OpenAI-compatible endpoint (e.g., LM Studio / LocalAI / Ollama-compatible) to summarize.

Examples:
    # LM Studio / LocalAI (completions-style API)
    python summarize_news.py --file corporate.list --top 3 \
        --llm-endpoint http://127.0.0.1:1234/v1/completions \
        --llm-model "openai/gpt-oss-20b"

    # Ollama (chat endpoint is often more compatible)
    python summarize_news.py --file corporate.list --top 3 \
        --llm-endpoint http://127.0.0.1:11434/v1/chat/completions \
        --llm-model "llama3.1:8b" --use-chat

    # Use the web app's JSON API as the news source instead of RSS
    python summarize_news.py --source web --web-api http://127.0.0.1:8080/api/news?top=3 \
        --llm-endpoint http://127.0.0.1:1234/v1/completions \
        --llm-model "openai/gpt-oss-20b"

Output:
    - Prints Markdown to stdout
    - Saves to ./markdown/news_summary_YYYYMMDD_HHMMSS.md
"""
import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote, urlparse
import xml.etree.ElementTree as ET

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests", file=sys.stderr)
    sys.exit(1)

JST = timezone(timedelta(hours=9))
OUTPUT_DIR = Path("markdown")

# -----------------------------
# News fetching
# -----------------------------
def build_gnews_rss_url(query: str, lang: str, region: str) -> str:
    return f"https://news.google.com/rss/search?q={quote(query)}&hl={lang}&gl={region}&ceid={region}:{lang}"

def parse_rss_items(xml_text: str):
    items = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return items
    for item in root.findall(".//item"):
        title_el = item.find("title")
        link_el = item.find("link")
        pub_el = item.find("pubDate")
        title = title_el.text.strip() if title_el is not None and title_el.text else ""
        link = link_el.text.strip() if link_el is not None and link_el.text else ""
        pub = pub_el.text.strip() if pub_el is not None and pub_el.text else ""
        items.append({"title": title, "link": link, "pubDate": pub})
    return items

def jst_str_from_pubdate(pubdate: str) -> str:
    try:
        dt = datetime.strptime(pubdate, "%a, %d %b %Y %H:%M:%S %Z")
        dt = dt.replace(tzinfo=timezone.utc).astimezone(JST)
        return dt.strftime("%Y-%m-%d %H:%M JST")
    except Exception:
        return pubdate or ""

def load_companies(path: Path):
    defaults = ["Nintendo", "Kioxia", "Socionext", "Nvidia", "Intel"]
    if not path.exists():
        return defaults
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")] or defaults

def fetch_rss_for_company(company: str, lang: str, region: str, timeout=15):
    url = build_gnews_rss_url(company, lang, region)
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    items = parse_rss_items(r.text)
    for it in items:
        it["pubDateJST"] = jst_str_from_pubdate(it.get("pubDate", ""))
    return items

def fetch_webapi_json(api_url: str, timeout=15):
    r = requests.get(api_url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.json()

# -----------------------------
# LLM prompting
# -----------------------------
def hostname(url: str) -> str:
    try:
        netloc = urlparse(url).netloc
        return netloc or ""
    except Exception:
        return ""

def build_prompt(company: str, items, language: str = "ja") -> str:
    # Keep prompt compact to avoid exceeding context window
    header = (
        "あなたは有能なアナリストです。以下のニュース見出しから、"
        "各社の最新動向を日本語で簡潔に要約してください。"
        "事実のみ、誇張なし、固有名詞は正確に。最後にインパクト評価(↑/→/↓)も付与してください。\n"
    )
    bullet_lines = []
    for it in items:
        title = it.get("title", "").replace("\n", " ").strip()
        link = it.get("link", "").strip()
        when = it.get("pubDateJST") or it.get("pubDate") or ""
        src = hostname(link)
        bullet_lines.append(f"- [{when}] {title} ({src})")
    bullets = "\n".join(bullet_lines)
    instr = (
        f"\n会社: {company}\n"
        "出力フォーマット:\n"
        "1) 3〜5行の要約（箇条書き可）\n"
        "2) インパクト評価: ↑/→/↓（短評1行）\n"
        "3) 注意点: 事実と推測は分ける\n"
        "----- 見出し -----\n"
        f"{bullets}\n"
        "------------------\n"
        "出力は日本語のみ。"
    )
    return header + instr

def call_llm(endpoint: str, model: str, prompt: str, use_chat: bool = False, temperature: float = 0.2, max_tokens: int = 512, stream: bool = False):
    headers = {"Content-Type": "application/json"}
    if use_chat:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "日本語で簡潔・正確に要約してください。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
    else:
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
    r = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=120)
    r.raise_for_status()
    data = r.json()
    if use_chat:
        # OpenAI-style chat
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return json.dumps(data, ensure_ascii=False, indent=2)
    else:
        # OpenAI-style completions
        try:
            return data["choices"][0]["text"]
        except Exception:
            return json.dumps(data, ensure_ascii=False, indent=2)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["rss", "web"], default="rss", help="news source: rss (Google News) or web (use --web-api)")
    ap.add_argument("--web-api", default="http://127.0.0.1:8080/api/news?top=3", help="When --source web, JSON endpoint to read")
    ap.add_argument("--file", default="corporate.list")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--region", default="JP")
    ap.add_argument("--llm-endpoint", default="http://127.0.0.1:1234/v1/completions")
    ap.add_argument("--llm-model", default="openai/gpt-oss-20b")
    ap.add_argument("--use-chat", action="store_true", help="Use /v1/chat/completions format")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--stream", action="store_true")
    args = ap.parse_args()

    companies = load_companies(Path(args.file))

    # Gather news
    news_map = {}
    if args.source == "web":
        raw = fetch_webapi_json(args.web_api)
        # Expect: { company: { "items": [ {title, link, pubDate, pubDateJST?}, ... ] } }
        for c in companies:
            payload = raw.get(c, {}) if isinstance(raw, dict) else {}
            items = payload.get("items", []) if isinstance(payload, dict) else []
            news_map[c] = items[: max(args.top, 0)]
    else:
        # RSS directly
        for c in companies:
            try:
                items = fetch_rss_for_company(c, args.lang, args.region)[: max(args.top, 0)]
            except Exception as e:
                sys.stderr.write(f"[WARN] {c}: RSS fetch failed: {e}\n")
                items = []
            news_map[c] = items

    # Summarize via LLM
    sections = []
    for c in companies:
        items = news_map.get(c, [])
        prompt = build_prompt(c, items, language=args.lang)
        try:
            summary = call_llm(
                endpoint=args.llm_endpoint,
                model=args.llm_model,
                prompt=prompt,
                use_chat=args.use_chat,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                stream=args.stream,
            ).strip()
        except Exception as e:
            summary = f"LLM呼び出しに失敗しました: {e}"

        # Build Markdown section
        section = [f"## {c}", "", summary, "", "**利用した見出し**:" ]
        if not items:
            section.append("- (なし)")
        else:
            for it in items:
                when = it.get("pubDateJST") or it.get("pubDate") or ""
                title = it.get("title","").replace("\n"," ").strip()
                link = it.get("link","").strip()
                src = hostname(link)
                section.append(f"- [{when}] {title} ({src})  \n  {link}")
        sections.append("\n".join(section))

    md = ["# 企業別ニュース要約", f"_生成時刻: {datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S JST')}_", ""]
    md.extend(sections)
    out_text = "\n\n".join(md) + "\n"
    print(out_text)

    # Save to file under the markdown output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"news_summary_{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}.md"
    out_path = OUTPUT_DIR / out_name
    out_path.write_text(out_text, encoding="utf-8")
    print(f"\n[Saved] {out_path}")

if __name__ == "__main__":
    main()
