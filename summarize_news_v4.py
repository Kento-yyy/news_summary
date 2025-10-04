#!/usr/bin/env python3
# summarize_news_v4.py (fixed echo_raw attribute)
import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from hashlib import sha1
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote, urlparse, urlsplit, urlunsplit, parse_qsl, urlencode
import xml.etree.ElementTree as ET

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests", file=sys.stderr)
    sys.exit(1)

try:
    import trafilatura
except Exception:
    trafilatura = None

JST = timezone(timedelta(hours=9))
CACHE_DIR = Path(".news_cache")
MARKDOWN_OUTPUT_DIR = Path("markdown")
OUTPUT_BEGIN = "<<<BEGIN_OUTPUT>>>"
OUTPUT_END = "<<<END_OUTPUT>>>"
FORMAT_ERROR_TOKEN = "FORMAT_ERROR"
MAX_FORMAT_REGENERATIONS = 3


@dataclass
class FormatCheckResult:
    valid: bool
    output: str
    issues: list
    raw_validator: str


CHANNEL_TAG_RE = re.compile(r"<\|[^>]+\|>")


def strip_code_fence(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n", "", t)
        t = re.sub(r"\n```$", "", t)
    t = CHANNEL_TAG_RE.sub("", t)
    return t.strip()


def extract_payload_block(text: str) -> str:
    if not text:
        return ""
    m = re.search(rf"{re.escape(OUTPUT_BEGIN)}(.*?){re.escape(OUTPUT_END)}", text, flags=re.DOTALL)
    if m:
        block = m.group(1)
        block = re.sub(r"<<<?BEGIN_OUTPUT>{1,3}", "", block)
        block = re.sub(r"<<<?END_OUTPUT>{1,3}", "", block)
        return strip_code_fence(block)
    start = text.find(OUTPUT_BEGIN)
    if start != -1:
        fallback = text[start + len(OUTPUT_BEGIN):]
        fallback = re.sub(r"<<<?END_OUTPUT>{1,3}", "", fallback)
        return strip_code_fence(fallback)
    cleaned = strip_code_fence(text)
    cleaned = re.sub(r"<<<?BEGIN_OUTPUT>{1,3}", "", cleaned)
    cleaned = re.sub(r"<<<?END_OUTPUT>{1,3}", "", cleaned)
    return cleaned


def local_validate_markers(payload: str) -> list:
    issues = []
    if not payload:
        issues.append("empty_payload")
        return issues
    blocks = re.findall(r"\[[A-Z]+\]", payload)
    if "[SUMMARY]" not in blocks:
        issues.append("missing_summary")
    if "[IMPACT]" not in blocks:
        issues.append("missing_impact")
    if "[NOTES]" not in blocks:
        issues.append("missing_notes")
    return issues


def build_validator_prompt(schema: str, candidate: str) -> str:
    if schema == "markers":
        rules = (
            "- `<<<BEGIN_OUTPUT>>>` と `<<<END_OUTPUT>>>` で囲まれていること\n"
            "- `\n[SUMMARY]`, `[IMPACT]`, `[NOTES]` の順でセクションが並ぶこと\n"
            "- SUMMARYは3〜5行、各行は先頭に`・`を含め事実/推測表示を含むこと\n"
            "- IMPACTは2行: `mark: ↑|→|↓` と `reason: ...`\n"
            "- NOTESは1〜2行で簡潔に\n"
            "- 余計な文や囲み、説明、コードブロックを含まないこと"
        )
    else:
        rules = (
            "- `<<<BEGIN_OUTPUT>>>` と `<<<END_OUTPUT>>>` の1対で囲まれていること\n"
            "- 出力全体がJSON1個のみ（sentinelを除く）\n"
            "- キー: summary(配列3〜5件), impact(mark/理由), notes(配列1〜2件)\n"
            "- JSON以外の文字列やコメントを含まないこと"
        )
    return (
        "あなたはフォーマット検証エージェントです。候補出力が仕様に厳密に従っているかを判定し、不備があれば修正案を提示してください。\n"
        "期待するフォーマット規則:\n"
        f"{rules}\n"
        "応答要件:\n"
        "- 必ず次のJSONのみを返すこと。コードブロック禁止。\n"
        '{"valid": true|false, "normalized": "<遵守した出力または空文字>", "issues": ["..."]}\n'
        "- `normalized` は修正後出力。修正できない場合は空文字。\n"
        "候補出力:\n"
        f"<<<CANDIDATE>>>{candidate}\n<<<END>>>"
    )


def run_format_validator(endpoint: str, model: str, candidate: str, schema: str, *, temperature: float = 0.0, max_tokens: int = 400, stop=None, retries: int = 1) -> FormatCheckResult:
    prompt = build_validator_prompt(schema, candidate)
    try:
        raw = call_llm(endpoint, model, prompt, temperature, max_tokens, stop=stop, retries=retries)
    except Exception as e:
        return FormatCheckResult(False, candidate, [f"validator_call_error: {e}"], "")
    cleaned = strip_code_fence(raw)
    if cleaned and cleaned[0] not in "{""[":
        brace_idx = cleaned.find("{")
        bracket_idx = cleaned.find("[")
        idx_candidates = [i for i in (brace_idx, bracket_idx) if i != -1]
        if idx_candidates:
            cleaned = cleaned[min(idx_candidates):]
    try:
        data = json.loads(cleaned)
        valid = bool(data.get("valid"))
        normalized = data.get("normalized")
        if not normalized:
            normalized = candidate
        issues = data.get("issues") or []
        return FormatCheckResult(valid, normalized, issues, raw)
    except Exception as e:
        return FormatCheckResult(False, candidate, [f"validator_parse_error: {e}"], raw)

def hostname(url: str) -> str:
    try:
        return urlparse(url).netloc or ""
    except Exception:
        return ""

def hkey(url: str) -> str:
    return sha1(url.encode("utf-8")).hexdigest()

def cache_get(url: str):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{hkey(url)}.txt"
    return p.read_text(encoding="utf-8") if p.exists() else None

def cache_set(url: str, text: str):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{hkey(url)}.txt"
    p.write_text(text, encoding="utf-8")

def build_gnews_rss_url(query: str, lang: str, region: str) -> str:
    return f"https://news.google.com/rss/search?q={quote(query)}&hl={lang}&gl={region}&ceid={region}:{lang}"

def parse_rss_items(xml_text: str):
    items = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return items
    for item in root.findall('.//item'):
        title_el = item.find('title')
        link_el = item.find('link')
        pub_el = item.find('pubDate')
        title = title_el.text.strip() if title_el is not None and title_el.text else ''
        link = link_el.text.strip() if link_el is not None and link_el.text else ''
        pub = pub_el.text.strip() if pub_el is not None and pub_el.text else ''
        items.append({'title': title, 'link': link, 'pubDate': pub})
    return items

def jst_str_from_pubdate(pubdate: str) -> str:
    from datetime import datetime
    try:
        dt = datetime.strptime(pubdate, "%a, %d %b %Y %H:%M:%S %Z")
        dt = dt.replace(tzinfo=timezone.utc).astimezone(JST)
        return dt.strftime("%Y-%m-%d %H:%M JST")
    except Exception:
        return pubdate or ""


def parse_datetime_guess(value: str):
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        dt = parsedate_to_datetime(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass
    cleaned = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass
    suffix_map = {
        "JST": JST,
        "UTC": timezone.utc,
        "GMT": timezone.utc,
    }
    for suffix, tz in suffix_map.items():
        if text.endswith(suffix):
            core = text[: -len(suffix)].strip()
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                try:
                    dt = datetime.strptime(core, fmt)
                    dt = dt.replace(tzinfo=tz)
                    return dt.astimezone(timezone.utc)
                except Exception:
                    continue
            try:
                dt = datetime.fromisoformat(core + ("+09:00" if tz == JST else "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=tz)
                return dt.astimezone(timezone.utc)
            except Exception:
                continue
    return None


def item_published_datetime(item: dict):
    if not isinstance(item, dict):
        return None
    for key in ("pubDate", "pubDateJST", "publishedAt", "published_at", "date", "datetime"):
        candidate = item.get(key)
        dt = parse_datetime_guess(candidate)
        if dt is not None:
            return dt
    return None


def filter_items_by_period(items, period_days: int):
    iterable = list(items or [])
    if period_days is None or period_days <= 0:
        return iterable
    threshold = datetime.now(timezone.utc) - timedelta(days=period_days)
    filtered = []
    for it in iterable:
        dt = item_published_datetime(it)
        if dt is None or dt >= threshold:
            filtered.append(it)
    return filtered


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


def adjust_web_api_top(api_url: str, fetch_limit: int) -> str:
    if fetch_limit <= 0:
        return api_url
    try:
        parts = list(urlsplit(api_url))
        query_pairs = parse_qsl(parts[3], keep_blank_values=True)
        query = dict(query_pairs)
        query["top"] = str(fetch_limit)
        parts[3] = urlencode(query)
        return urlunsplit(parts)
    except Exception:
        return api_url

def fetch_fulltext(url: str, timeout=15, force=False):
    if trafilatura is None:
        return ""
    if not force:
        cached = cache_get(url)
        if cached is not None:
            return cached
    try:
        downloaded = trafilatura.fetch_url(url, timeout=timeout)
        if not downloaded:
            return ""
        extracted = trafilatura.extract(downloaded) or ""
        text = extracted.strip()
        if text:
            cache_set(url, text)
        return text
    except Exception:
        return ""

def build_bullets(items, fulltext=False, snippet_chars=600):
    lines = []
    for it in items:
        title = it.get("title", "").replace("\n", " ").strip()
        link = it.get("link", "").strip()
        when = it.get("pubDateJST") or it.get("pubDate") or ""
        src = hostname(link)
        line = f"- [{when}] {title} ({src}) :: {link}"
        if fulltext and link:
            body = fetch_fulltext(link)
            if body:
                snippet = body[:snippet_chars].replace("\n", " ").strip()
                if len(body) > snippet_chars:
                    snippet += "…"
                line += f"\n  摘要: {snippet}"
        lines.append(line)
    return "\n".join(lines) if lines else "- (なし)"

FEWSHOT_MARKERS = (
    "以下は厳密な出力例です。これと同じ構造で出力してください。\n"
    f"{OUTPUT_BEGIN}\n"
    "[SUMMARY]\n"
    "・製品出荷開始が公式に発表（事実）。\n"
    "・主要パートナーとの協業範囲を拡大（事実）。\n"
    "・短中期の売上押上げが期待（推測）。\n"
    "[IMPACT]\n"
    "mark: ↑\n"
    "reason: 量産開始と販路拡大は成長ドライバ。\n"
    "[NOTES]\n"
    "・数値は未開示のため、規模感は推測に留まる。\n"
    f"{OUTPUT_END}\n"
)

def prompt_markers(company: str, items, fulltext=False, snippet_chars=600) -> str:
    bullets = build_bullets(items, fulltext=fulltext, snippet_chars=snippet_chars)
    return (
        "あなたは有能なアナリストです。以下の見出し（必要なら摘要）に基づき、簡潔に要約してください。\n"
        "出力プロトコル:\n"
        f"- 最初の行に{OUTPUT_BEGIN}、最後の行に{OUTPUT_END}を置くこと\n"
        "- その間に指定されたセクションのみを含め、追加説明は禁止\n"
        f"- 指示に従えない場合は `{FORMAT_ERROR_TOKEN}` のみを返すこと\n"
        "- SUMMARY内では各行末尾で（事実）や（推測）を明示\n"
        f"{FEWSHOT_MARKERS}"
        "出力要件:\n"
        "- SUMMARY: 3〜5行の箇条書き（事実/推測を明示）\n"
        "- IMPACT: {mark, reason} の2行のみ\n"
        "- NOTES: 1〜2行\n"
        f"会社: {company}\n"
        "----- 見出し ----\n"
        f"{bullets}\n"
        "------------------\n"
        "注意: マーカー以外の文は出力しないこと。"
    )

def prompt_json(company: str, items, fulltext=False, snippet_chars=600) -> str:
    bullets = build_bullets(items, fulltext=fulltext, snippet_chars=snippet_chars)
    return (
        "あなたは有能なアナリストです。以下の見出し（必要なら摘要）に基づき、簡潔に出力してください。\n"
        "出力プロトコル:\n"
        f"- 最初の行に{OUTPUT_BEGIN}、最後の行に{OUTPUT_END}を置き、間は純粋なJSON1個のみ\n"
        f"- 形式を外れる場合は `{FORMAT_ERROR_TOKEN}` のみを返すこと\n"
        '{ "summary": ["...","...","..."], "impact": {"mark":"↑|→|↓","reason":"..."}, "notes":["..."] }\n'
        f"会社: {company}\n"
        "----- 見出し ----\n"
        f"{bullets}\n"
        "------------------"
    )

def parse_markers(text: str):
    section = extract_payload_block(text)
    if section.strip() == FORMAT_ERROR_TOKEN:
        raise ValueError("LLM returned format error")
    def extract(tag):
        rgx = rf"\[{tag}\](.*?)(?=\n\[[A-Z]+\]\n|$)"
        mm = re.search(rgx, section, flags=re.DOTALL)
        return mm.group(1).strip() if mm else ""
    def to_lines(block):
        out = []
        for ln in block.splitlines():
            t = ln.strip()
            if not t:
                continue
            t = re.sub(r"^[・\-•\*]\s*", "", t)
            out.append(t)
        return out
    summary = to_lines(extract("SUMMARY"))
    notes   = to_lines(extract("NOTES"))
    impact_raw = extract("IMPACT")
    mark_m  = re.search(r"mark:\s*([↑→↓])", impact_raw)
    reason_m= re.search(r"reason:\s*(.*)", impact_raw)
    impact = {"mark": mark_m.group(1) if mark_m else "", "reason": (reason_m.group(1).strip() if reason_m else "")}
    return {"summary": summary, "impact": impact, "notes": notes}

def parse_json(text: str):
    t = extract_payload_block(text)
    if t.strip() == FORMAT_ERROR_TOKEN:
        raise ValueError("LLM returned format error")
    obj = json.loads(t)
    summary = [s.strip() for s in obj.get("summary", []) if str(s).strip()]
    notes   = [s.strip() for s in obj.get("notes", []) if str(s).strip()]
    impact  = obj.get("impact", {}) or {}
    return {"summary": summary, "impact": {"mark": impact.get("mark",""), "reason": (impact.get("reason","") or "").strip()}, "notes": notes}

def _choice_text(choice) -> str:
    if not choice:
        return ""
    if "text" in choice and isinstance(choice["text"], str):
        return choice["text"]
    message = choice.get("message") if isinstance(choice, dict) else None
    if isinstance(message, dict):
        content = message.get("content", "")
        if isinstance(content, list):
            return "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)
        if isinstance(content, str):
            return content
    return ""


def _trim_assistant_wrappers(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    # Strip tool-style channel prefixes (e.g., "analysis" / "assistantfinal")
    final_markers = ["assistantfinal", "assistant_final", "final_response"]
    for marker in final_markers:
        idx = t.find(marker)
        if idx != -1:
            t = t[idx + len(marker):]
            break
    confidence_markers = ["assistantconfidence", "assistant_confidence"]
    for marker in confidence_markers:
        idx = t.find(marker)
        if idx != -1:
            t = t[:idx]
            break
    # Remove leading "analysis" artifact if present after trimming
    if t.startswith("analysis"):
        t = t[len("analysis"):]
    return t.strip()


def call_llm(endpoint: str, model: str, prompt: str, temperature: float, max_tokens: int, stop=None, retries=1, backoff=2.0):
    headers = {"Content-Type": "application/json"}
    payload = {"model": model, "prompt": prompt, "temperature": temperature, "max_tokens": max_tokens}
    if stop:
        payload["stop"] = stop
    last = None
    for i in range(retries + 1):
        try:
            r = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=240)
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices") or []
            if not choices:
                raise ValueError("LLM response missing choices")
            raw_text = _choice_text(choices[0])
            cleaned = _trim_assistant_wrappers(CHANNEL_TAG_RE.sub("", raw_text))
            return strip_code_fence(cleaned)
        except Exception as e:
            last = e
            time.sleep(backoff * (i + 1))
    raise last if last else RuntimeError("LLM request failed")

def render_markdown(company: str, parsed, items):
    lines = [f"## {company}", ""]
    summary = parsed.get("summary", [])
    if summary:
        for s in summary[:5]:
            lines.append(f"- {s}")
    else:
        lines.append("- (要約なし)")
    lines.append("")
    impact = parsed.get("impact", {})
    mark = (impact.get("mark") or "→").strip()
    reason = (impact.get("reason") or "").strip()
    if reason:
        lines.append(f"**インパクト評価:** {mark}（{reason}）")
    else:
        lines.append(f"**インパクト評価:** {mark}")
    lines.append("")
    notes = parsed.get("notes", [])
    if notes:
        lines.append("**注意点:**")
        for n in notes[:3]:
            lines.append(f"- {n}")
        lines.append("")
    lines.append("**利用した見出し**:")
    if not items:
        lines.append("- (なし)")
    else:
        for it in items:
            when = it.get("pubDateJST") or it.get("pubDate") or ""
            title = it.get("title","").replace("\n"," ").strip()
            link = it.get("link","").strip()
            src = hostname(link)
            lines.append(f"- [{when}] {title} ({src})  \n  {link}")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["rss","web"], default="rss")
    ap.add_argument("--web-api", default="http://127.0.0.1:8080/api/news?top=100")
    ap.add_argument("--file", default="corporate.list")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--region", default="JP")
    ap.add_argument("--fulltext", action="store_true")
    ap.add_argument("--snippet-chars", type=int, default=600)
    ap.add_argument("--llm-endpoint", default="http://127.0.0.1:1234/v1/completions")
    ap.add_argument("--llm-model", default="openai/gpt-oss-20b")
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--stop", default="<<<END>>>,</s>,<|,assistant:,system:,user:", help="Comma-separated stop sequences")
    ap.add_argument("--validator-model", default="", help="Override model for the format checker")
    ap.add_argument("--validator-max-tokens", type=int, default=400)
    ap.add_argument("--validator-temperature", type=float, default=0.0)
    ap.add_argument(
        "--fetch-limit",
        type=int,
        default=100,
        help="Number of articles to retrieve before applying date filters (default: 100).",
    )
    ap.add_argument(
        "--period-days",
        type=int,
        default=7,
        help="Include only articles published within this many days (default: 7). Use 0 to disable filtering.",
    )
    ap.add_argument("--schema", choices=["markers","json"], default="markers")
    ap.add_argument("--min-lines", type=int, default=2)
    ap.add_argument("--echo-raw", action="store_true")
    ap.add_argument(
        "-o",
        "--output",
        default="",
        help="Output filename (relative paths are saved under markdown/)",
    )

    args = ap.parse_args()

    stop_list = [s for s in (args.stop.split(",") if args.stop else []) if s]
    companies = load_companies(Path(args.file))
    validator_model = args.validator_model or args.llm_model

    # Gather news
    top_limit = max(args.top or 0, 0)
    fetch_limit = args.fetch_limit if args.fetch_limit is not None else 0
    if fetch_limit < 0:
        fetch_limit = 0
    if fetch_limit and top_limit > fetch_limit:
        fetch_limit = top_limit

    if args.source == "web":
        api_url = adjust_web_api_top(args.web_api, fetch_limit) if fetch_limit else args.web_api
        raw = fetch_webapi_json(api_url)
        news_map = {}
        for c in companies:
            payload = (raw.get(c, {}) or {})
            items = payload.get("items") or []
            limited = items[:fetch_limit] if fetch_limit else items
            filtered = filter_items_by_period(limited, args.period_days)
            news_map[c] = filtered[:top_limit]
    else:
        news_map = {}
        for c in companies:
            try:
                fetched = fetch_rss_for_company(c, args.lang, args.region)
                limited = fetched[:fetch_limit] if fetch_limit else fetched
                filtered = filter_items_by_period(limited, args.period_days)
                news_map[c] = filtered[:top_limit]
            except Exception as e:
                sys.stderr.write(f"[WARN] {c}: RSS fetch failed: {e}\n")
                news_map[c] = []

    sections = []
    raw_dir = Path("./llm_raw")
    if args.echo_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    def save_raw(name: str, content: str):
        if args.echo_raw:
            (raw_dir / name).write_text(content, encoding="utf-8")

    def validate_output(schema: str, candidate: str, company: str, attempt_tag: str) -> FormatCheckResult:
        def local_check(text: str):
            issues = []
            if not text or not text.strip():
                issues.append("empty_candidate")
            if OUTPUT_BEGIN not in text or OUTPUT_END not in text:
                issues.append("missing_enclosure")
            payload = extract_payload_block(text)
            if not payload:
                issues.append("empty_payload")
                return payload, issues, False
            if schema == "markers":
                for item in local_validate_markers(payload):
                    issues.append(item)
            elif schema == "json":
                try:
                    json.loads(payload)
                except Exception as e:
                    issues.append(f"json_load_error:{e}")
            return payload, issues, not issues

        if candidate.strip() == FORMAT_ERROR_TOKEN:
            return FormatCheckResult(False, candidate, ["format_error_token"], "")

        if OUTPUT_BEGIN in candidate and OUTPUT_END not in candidate:
            candidate = candidate.rstrip() + f"\n{OUTPUT_END}"

        _, initial_local_issues, initial_local_ok = local_check(candidate)

        res1 = run_format_validator(
            args.llm_endpoint,
            validator_model,
            candidate,
            schema,
            temperature=args.validator_temperature,
            max_tokens=args.validator_max_tokens,
            stop=None,
            retries=1,
        )
        if res1.raw_validator:
            save_raw(f"{company}_{attempt_tag}_validator1.json", res1.raw_validator)

        result = res1
        validator_issues = list(res1.issues or [])

        if (not res1.valid) and res1.output != candidate:
            res2 = run_format_validator(
                args.llm_endpoint,
                validator_model,
                res1.output,
                schema,
                temperature=args.validator_temperature,
                max_tokens=args.validator_max_tokens,
                stop=None,
                retries=1,
            )
            if res2.raw_validator:
                save_raw(f"{company}_{attempt_tag}_validator2.json", res2.raw_validator)
            validator_issues.extend(res2.issues or [])
            result = res2

        final_output = result.output or candidate
        _, final_local_issues, final_local_ok = local_check(final_output)

        issues = []
        if not final_local_ok:
            issues.extend(final_local_issues)
            if validator_issues:
                issues.extend(validator_issues)
        else:
            # Keep validator feedback only as warnings when local check passes.
            issues.extend([f"validator_warning:{msg}" for msg in validator_issues])

        return FormatCheckResult(final_local_ok, final_output, issues, result.raw_validator)

    def summarize_validator_notes(codes):
        markers = set()
        json_msgs = set()
        others = []
        for code in codes:
            if not isinstance(code, str):
                continue
            prefix, _, tail = code.partition(":")
            if prefix == "markers":
                markers.add(tail)
            elif prefix == "json":
                json_msgs.add(tail)
            else:
                others.append(code)
        messages = []
        if markers:
            cleaned = {m for m in markers if m}
            trivial = {
                "empty_candidate",
                "empty_payload",
                "missing_enclosure",
                "候補出力が提供されていないため、フォーマット規則に従っているか判定できません。",
            }
            if cleaned and cleaned.issubset(trivial):
                messages.append("フォーマット検証: マーカー形式の出力が空判定 → JSON形式で再生成。")
            elif cleaned:
                messages.append("フォーマット検証: マーカー形式で不備 (" + ", ".join(sorted(cleaned)) + ")")
        if json_msgs:
            warnings = []
            errors = []
            for msg in json_msgs:
                if not msg:
                    continue
                tag, sep, rest = msg.partition(":")
                if tag == "validator_warning" and sep:
                    warnings.append(rest)
                else:
                    errors.append(msg)
            if errors:
                messages.append("フォーマット検証: JSON形式で不備 (" + ", ".join(sorted(errors)) + ")")
            for w in warnings:
                messages.append("フォーマット検証: JSONバリデータ警告 (" + w + ")")
        for other in others:
            messages.append(f"フォーマット検証: {other}")
        return messages

    for c in companies:
        print(c)
        items = news_map.get(c, [])
        p1 = prompt_markers(c, items, fulltext=args.fulltext, snippet_chars=args.snippet_chars)
        try:
            out1 = call_llm(
                args.llm_endpoint,
                args.llm_model,
                p1,
                args.temperature,
                args.max_tokens,
                stop=stop_list,
                retries=1,
            )
        except Exception as e:
            out1 = f"(error) {e}"

        validator_notes = []
        parsed = None
        marker_source = out1
        validation = None

        for attempt in range(1, MAX_FORMAT_REGENERATIONS + 1):
            attempt_suffix = "" if attempt == 1 else f"_attempt{attempt}"
            if attempt > 1:
                try:
                    out1 = call_llm(
                        args.llm_endpoint,
                        args.llm_model,
                        p1,
                        args.temperature,
                        args.max_tokens,
                        stop=stop_list,
                        retries=1,
                    )
                except Exception as e:
                    out1 = f"(error) {e}"
            save_raw(f"{c}_try1{attempt_suffix}.txt", out1)

            attempt_tag = f"try1{attempt_suffix}"
            validation = validate_output("markers", out1, c, attempt_tag)
            marker_source = validation.output or out1
            try:
                parsed = parse_markers(marker_source)
            except Exception as e:
                validator_notes.append(f"markers:parse_error:{e}")
                parsed = None
            if not validation.valid:
                if validation.issues:
                    validator_notes.extend([f"markers:{msg}" for msg in validation.issues])
                else:
                    validator_notes.append("markers:invalid")
            if validation.valid and parsed and len(parsed.get("summary", [])) >= args.min_lines:
                break
        if not parsed or len(parsed.get("summary", [])) < args.min_lines:
            pj = prompt_json(c, items, fulltext=args.fulltext, snippet_chars=args.snippet_chars)
            out2 = ""
            validation_json = None
            for attempt in range(1, MAX_FORMAT_REGENERATIONS + 1):
                attempt_suffix = "" if attempt == 1 else f"_attempt{attempt}"
                try:
                    out2 = call_llm(
                        args.llm_endpoint,
                        args.llm_model,
                        pj,
                        args.temperature,
                        args.max_tokens,
                        stop=["```"],
                        retries=1,
                    )
                except Exception as e:
                    out2 = f"(error) {e}"
                save_raw(f"{c}_try2_json{attempt_suffix}.txt", out2)
                attempt_tag = f"try2{attempt_suffix}"
                validation_json = validate_output("json", out2, c, attempt_tag)
                try:
                    parsed = parse_json(validation_json.output or out2)
                except Exception as e:
                    parsed = {
                        "summary": [],
                        "impact": {"mark": "→", "reason": ""},
                        "notes": ["LLM出力の解析に失敗。"],
                    }
                    validator_notes.append(f"json:parse_error:{e}")
                if not validation_json.valid:
                    if validation_json.issues:
                        validator_notes.extend([f"json:{msg}" for msg in validation_json.issues])
                    else:
                        validator_notes.append("json:invalid")
                    if attempt < MAX_FORMAT_REGENERATIONS:
                        continue
                break

        if validator_notes:
            parsed.setdefault("notes", [])
            for msg in summarize_validator_notes(sorted(set(validator_notes))):
                parsed["notes"].append(msg)

        sections.append(render_markdown(c, parsed, items))

    md = ["# 企業別ニュース要約", f"_生成時刻: {datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S JST')}_", ""]
    md.extend(sections)
    out_text = "\n\n".join(md) + "\n"
    print(out_text)
    MARKDOWN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_arg = (args.output or "").strip()
    if output_arg:
        candidate = Path(output_arg)
        if not candidate.is_absolute() and candidate.parent == Path('.'):
            out_path = MARKDOWN_OUTPUT_DIR / candidate.name
        else:
            out_path = candidate
    else:
        out_path = MARKDOWN_OUTPUT_DIR / f"news_summary_{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_text, encoding="utf-8")
    print(f"\n[Saved] {out_path}")

if __name__ == "__main__":
    main()
