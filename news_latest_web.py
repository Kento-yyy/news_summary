#!/usr/bin/env python3
"""
Tiny web app to view latest headlines per company.
Great for playing with Chrome DevTools (Network tab, etc.).

Run:
    pip install flask requests
    python news_latest_web.py --port 8080

Then open: http://127.0.0.1:8080
"""
import argparse
from pathlib import Path
from urllib.parse import quote
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta

from flask import Flask, jsonify, render_template_string, request
import requests

JST = timezone(timedelta(hours=9))

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

def fetch_company_news(company: str, lang: str, region: str, timeout=15):
    url = build_gnews_rss_url(company, lang, region)
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    items = parse_rss_items(r.text)
    for it in items:
        it["pubDateJST"] = jst_str_from_pubdate(it.get("pubDate", ""))
    return items

def create_app(companies, lang="ja", region="JP"):
    app = Flask(__name__)

    @app.get("/api/news")
    def api_news():
        top = int(request.args.get("top", "3"))
        out = {}
        for c in companies:
            try:
                items = fetch_company_news(c, lang, region)[:top]
            except Exception as e:
                out[c] = {"error": str(e), "items": []}
            else:
                out[c] = {"items": items}
        return jsonify(out)

    @app.get("/")
    def index():
        return render_template_string("""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Corporate Latest News</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Noto Sans JP", "Hiragino Kaku Gothic ProN", "Yu Gothic", Meiryo, sans-serif; margin: 2rem; }
    h1 { margin-top: 0; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
    .title { font-weight: 700; margin-bottom: .5rem; }
    .item { margin: .5rem 0; }
    .when { color: #555; font-size: .9em; }
    .error { color: #b00020; }
    .row { display: flex; gap: .5rem; align-items: center; margin-bottom: 1rem; }
    input[type=number] { width: 5rem; }
    button { padding: .4rem .8rem; border-radius: 8px; border: 1px solid #ccc; background: #f9f9f9; cursor: pointer; }
    button:hover { background: #f1f1f1; }
  </style>
</head>
<body>
  <h1>Corporate Latest News</h1>
  <div class="row">
    <label>Top: <input id="top" type="number" min="1" max="10" value="3"></label>
    <button onclick="loadNews()">Reload</button>
  </div>
  <div id="grid" class="grid"></div>

  <script>
    async function loadNews() {
      const top = document.getElementById('top').value || 3;
      const res = await fetch(`/api/news?top=${top}`);
      const data = await res.json();
      const grid = document.getElementById('grid');
      grid.innerHTML = '';
      for (const [company, payload] of Object.entries(data)) {
        const card = document.createElement('div');
        card.className = 'card';
        const title = document.createElement('div');
        title.className = 'title';
        title.textContent = company;
        card.appendChild(title);
        if (payload.error) {
          const err = document.createElement('div');
          err.className = 'error';
          err.textContent = payload.error;
          card.appendChild(err);
        } else {
          payload.items.forEach((it, idx) => {
            const wrap = document.createElement('div');
            wrap.className = 'item';
            const a = document.createElement('a');
            a.href = it.link;
            a.target = '_blank';
            a.rel = 'noopener';
            a.textContent = `${idx + 1}. ${it.title}`;
            const when = document.createElement('div');
            when.className = 'when';
            when.textContent = it.pubDateJST || it.pubDate || '';
            wrap.appendChild(a);
            wrap.appendChild(when);
            card.appendChild(wrap);
          });
        }
        grid.appendChild(card);
      }
    }
    loadNews();
  </script>
</body>
</html>
        """)

    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="corporate.list")
    parser.add_argument("--lang", default="ja")
    parser.add_argument("--region", default="JP")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    companies = load_companies(Path(args.file))
    app = create_app(companies, lang=args.lang, region=args.region)
    app.run(host=args.host, port=args.port, debug=True)
