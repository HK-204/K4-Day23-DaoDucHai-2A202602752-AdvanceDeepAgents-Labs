"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import time
import xml.etree.ElementTree

from dotenv import load_dotenv
import httpx
from langchain_core.tools import tool

load_dotenv()

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

_LAST_ARXIV_CALL = 0.0


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again."""
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as e:
            if attempt == attempts - 1:
                raise
            if e.retry_after is not None:
                delay = float(e.retry_after)
            else:
                delay = min(cap, base * (2 ** attempt)) + random.uniform(0.0, 0.5 * min(cap, base * (2 ** attempt)))
            delay = min(delay, cap)
            time.sleep(delay)


def _check_http(resp: httpx.Response):
    if resp.status_code in (429, 500, 502, 503, 504):
        retry_after = resp.headers.get("Retry-After")
        sec = None
        if retry_after:
            try:
                sec = float(retry_after)
            except ValueError:
                sec = None
        raise RetryableError(f"HTTP {resp.status_code}: {resp.text[:150]}", retry_after=sec)
    resp.raise_for_status()


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    global _LAST_ARXIV_CALL
    try:
        terms = re.findall(r"[\w-]+", query)
        if not terms:
            return "NO RESULTS"

        search_query = " AND ".join(f"all:{t}" for t in terms)
        limit = max(1, min(max_results, 30))

        def _fetch():
            global _LAST_ARXIV_CALL
            now = time.monotonic()
            elapsed = now - _LAST_ARXIV_CALL
            if elapsed < 3.0:
                time.sleep(3.0 - elapsed)
            _LAST_ARXIV_CALL = time.monotonic()

            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.get(
                        ARXIV_URL,
                        params={
                            "search_query": search_query,
                            "sortBy": "submittedDate",
                            "sortOrder": "descending",
                            "max_results": limit,
                        },
                    )
                    _check_http(resp)
                    return resp.text
            except httpx.TransportError as exc:
                raise RetryableError(f"arXiv transport error: {exc}") from exc

        xml_text = with_retry(_fetch, attempts=5, base=3.0, cap=60.0)
        root = xml.etree.ElementTree.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        if not entries:
            return "NO RESULTS"

        records = []
        for entry in entries:
            id_elem = entry.find("atom:id", ns)
            if id_elem is None or not id_elem.text:
                continue
            raw_id = id_elem.text.strip().split("/abs/")[-1]
            clean_id = re.sub(r"v\d+$", "", raw_id)
            url = f"https://arxiv.org/abs/{clean_id}"

            pub_elem = entry.find("atom:published", ns)
            published = pub_elem.text.strip()[:10] if pub_elem is not None and pub_elem.text else ""

            title_elem = entry.find("atom:title", ns)
            title = re.sub(r"\s+", " ", title_elem.text.strip()) if title_elem is not None and title_elem.text else "Untitled"

            sum_elem = entry.find("atom:summary", ns)
            summary = re.sub(r"\s+", " ", sum_elem.text.strip()) if sum_elem is not None and sum_elem.text else ""
            summary = summary[:600]

            records.append({
                "id": clean_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
            })

        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        lim = max(1, min(limit, 100))
        params = {"limit": lim}
        if date:
            params["date"] = date

        def _fetch():
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.get(HF_DAILY_URL, params=params)
                    _check_http(resp)
                    return resp.json()
            except httpx.TransportError as exc:
                raise RetryableError(f"HF Daily transport error: {exc}") from exc

        data = with_retry(_fetch, attempts=5, base=1.0, cap=30.0)
        if not isinstance(data, list) or not data:
            return "NO RESULTS"

        kw = keyword.strip().lower()
        records = []
        for item in data:
            if not isinstance(item, dict):
                continue
            paper = item.get("paper") if isinstance(item.get("paper"), dict) else item
            pid = paper.get("id")
            if not pid:
                continue
            url = f"https://huggingface.co/papers/{pid}"
            published = str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10]
            title = re.sub(r"\s+", " ", str(paper.get("title") or item.get("title") or "").strip())
            summary = re.sub(r"\s+", " ", str(paper.get("summary") or item.get("summary") or "").strip())[:600]
            upvotes = int(paper.get("upvotes") or item.get("upvotes") or 0)
            github = str(paper.get("githubRepo") or item.get("githubRepo") or "")
            stars = int(paper.get("githubStars") or item.get("githubStars") or 0)

            if kw and (kw not in title.lower() and kw not in summary.lower()):
                continue

            records.append({
                "id": str(pid),
                "url": url,
                "published": published,
                "title": title or "Untitled",
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        records.sort(key=lambda r: r.get("upvotes", 0), reverse=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        q = query.strip()
        if not q:
            return "NO RESULTS"
        lim = max(1, min(limit, 50))

        def _fetch():
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.get(HF_SEARCH_URL, params={"q": q, "limit": lim})
                    _check_http(resp)
                    return resp.json()
            except httpx.TransportError as exc:
                raise RetryableError(f"HF Search transport error: {exc}") from exc

        data = with_retry(_fetch, attempts=5, base=1.0, cap=30.0)
        if not isinstance(data, list) or not data:
            return "NO RESULTS"

        records = []
        for item in data:
            if not isinstance(item, dict):
                continue
            paper = item.get("paper") if isinstance(item.get("paper"), dict) else item
            pid = paper.get("id")
            if not pid:
                continue
            url = f"https://huggingface.co/papers/{pid}"
            published = str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10]
            title = re.sub(r"\s+", " ", str(paper.get("title") or item.get("title") or "").strip())
            summary_raw = paper.get("ai_summary") or paper.get("summary") or item.get("summary") or ""
            summary = re.sub(r"\s+", " ", str(summary_raw).strip())[:600]
            upvotes = int(paper.get("upvotes") or item.get("upvotes") or 0)
            github = str(paper.get("githubRepo") or item.get("githubRepo") or "")
            stars = int(paper.get("githubStars") or item.get("githubStars") or 0)

            records.append({
                "id": str(pid),
                "url": url,
                "published": published,
                "title": title or "Untitled",
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _redact_exa(text: str) -> str:
    key = os.getenv("EXA_API_KEY", "").strip()
    if key and key in text:
        return text.replace(key, "[REDACTED]")
    return text


def _call_exa_mcp(tool_name: str, arguments: dict) -> str:
    key = os.getenv("EXA_API_KEY", "").strip()
    target_url = f"{EXA_URL}?exaApiKey={key}" if key else EXA_URL
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }

    def _post():
        try:
            with httpx.Client(timeout=45.0) as client:
                resp = client.post(target_url, headers=headers, json=payload)
                _check_http(resp)
                raw_text = resp.text
        except httpx.TransportError as exc:
            raise RetryableError(f"Exa transport error: {exc}") from exc

        data = None
        for line in raw_text.splitlines():
            line_s = line.strip()
            if line_s.startswith("data:"):
                data = json.loads(line_s[5:].strip())
                break
        if data is None:
            data = json.loads(raw_text)

        if "error" in data:
            err_msg = str(data["error"])
            if "rate limit" in err_msg.lower() or "429" in err_msg:
                raise RetryableError(f"Exa RPC rate limit: {err_msg}", retry_after=10.0)
            raise RuntimeError(f"Exa RPC error: {err_msg}")

        result = data.get("result", {})
        meta = result.get("_meta", {})
        meta_str = json.dumps(meta).lower()
        if "rate_limit" in meta_str or "ratelimit" in meta_str or "rate limit" in meta_str:
            raise RetryableError("Exa rate limit flag in _meta", retry_after=10.0)

        content = result.get("content", [])
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                txt = item.get("text", "")
                if ("rate limit" in txt.lower() or "exceeded" in txt.lower()) and len(txt) < 350:
                    raise RetryableError(f"Exa text rate limit warning: {txt}", retry_after=10.0)
                texts.append(txt)

        return "\n\n".join(texts).strip()

    return with_retry(_post, attempts=5, base=2.0, cap=60.0)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        q = query.strip()
        if not q:
            return "NO RESULTS"
        obj = objective.strip() or f"find academic research papers and surveys about {q}"
        num = max(1, min(num_results, 10))

        result_text = _call_exa_mcp("web_search_exa", {"query": q, "objective": obj, "numResults": num})
        return result_text if result_text else "NO RESULTS"
    except Exception as exc:
        return _redact_exa(f"ERROR: {type(exc).__name__}: {exc}")


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        u = url.strip()
        if not u:
            return "NO RESULTS"

        result_text = _call_exa_mcp("web_fetch_exa", {"urls": [u]})
        if not result_text:
            return "NO RESULTS"
        return result_text[:12000]
    except Exception as exc:
        return _redact_exa(f"ERROR: {type(exc).__name__}: {exc}")


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
