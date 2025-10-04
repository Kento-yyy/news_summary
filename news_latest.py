#!/usr/bin/env python3
"""
Latest news viewer for companies listed in `corporate.list`.

- Default companies (if `corporate.list` is missing):
    Nintendo, Kioxia, Socionext, Nvidia, Intel
- Uses Google News RSS (no API key needed).
- Prints top N (default 3) headlines per company as text and JSON.

Usage:
    python news_latest.py --file corporate.list --top 3 --lang ja --region JP
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote
import xml.etree.ElementTree as ET

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests", file=sys.stderr)
    sys.exit(1)

JST = timezone(timedelta(hours=9))

def build_gnews_rss_url(query: str, lang: str, region: str) -> str:
    # Google News RSS (search)
    # Example: https://news.google.com/rss/search?q=Nintendo&hl=ja&gl=JP&ceid=JP:ja
    return f"https://news.google.com/rss/search?q={quote(query)}&hl={lang}&gl={region}&ceid={region}:{lang}"

def parse_rss_items(xml_text: str):
    # Works for Google News RSS (RSS 2.0); fallback tolerant parsing
    # Some feeds use namespaces; we ignore them for core fields.
    # Returns list of dicts: {title, link, pubDate}
    items = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return items

    # RSS 2.0 path: channel/item
    for item in root.findall(".//item"):
        title_el = item.find("title")
        link_el = item.find("link")
        pub_el = item.find("pubDate")
        title = title_el.text.strip() if title_el is not None and title_el.text else ""
        link = link_el.text.strip() if link_el is not None and link_el.text else ""
        pub = pub_el.text.strip() if pub_el is not None and pub_el.text else ""
        items.append({"title": title, "link": link, "pubDate": pub})
    return items

def load_companies(path: Path):
    # defaults = ["Nintendo", "Kioxia", "Socionext", "Nvidia", "Intel"]
    if not path.exists():
        return defaults
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    # filter blanks and comments
    return [ln for ln in lines if ln and not ln.startswith("#")] or defaults

def jst_str_from_pubdate(pubdate: str) -> str:
    # pubDate example: 'Sat, 04 Oct 2025 10:01:00 GMT'
    try:
        dt = datetime.strptime(pubdate, "%a, %d %b %Y %H:%M:%S %Z")
        # Assume UTC when %Z is present (usually GMT)
        dt = dt.replace(tzinfo=timezone.utc).astimezone(JST)
        return dt.strftime("%Y-%m-%d %H:%M JST")
    except Exception:
        return pubdate or ""

def fetch_company_news(company: str, lang: str, region: str, timeout=15):
    url = build_gnews_rss_url(company, lang, region)
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    items = parse_rss_items(r.text)
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="corporate.list", help="Path to corporate list file")
    ap.add_argument("--top", type=int, default=3, help="Number of headlines per company")
    ap.add_argument("--lang", default="ja", help="Language code for Google News (e.g., ja, en)")
    ap.add_argument("--region", default="JP", help="Region code for Google News (e.g., JP, US)")
    ap.add_argument("--json", action="store_true", help="Also print consolidated JSON to stdout at the end")
    args = ap.parse_args()

    companies = load_companies(Path(args.file))
    all_results = {}

    print(f"[{datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S JST')}] Latest news per company (Google News RSS)")
    print("-" * 80)

    for company in companies:
        try:
            items = fetch_company_news(company, args.lang, args.region)
        except Exception as e:
            print(f"{company}: ERROR fetching feed -> {e}", file=sys.stderr)
            all_results[company] = {"error": str(e), "items": []}
            continue

        topn = items[: max(args.top, 0)]
        all_results[company] = {"items": topn}

        print(f"\n## {company}")
        if not topn:
            print("  (no items)")
            continue
        for i, it in enumerate(topn, 1):
            when = jst_str_from_pubdate(it.get("pubDate", ""))
            title = it.get("title", "").replace("\n", " ").strip()
            link = it.get("link", "").strip()
            print(f"  {i}. {title}")
            if when:
                print(f"     - {when}")
            if link:
                print(f"     - {link}")

    if args.json:
        print("\n--- JSON ---")
        print(json.dumps(all_results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()