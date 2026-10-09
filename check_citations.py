"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


import re

_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")  # [3]  [1, 2]  [1-3]; not [3](link)
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_URL = re.compile(r"https?://\S+")


def _group_numbers(group):
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        else:
            numbers.append(int(part))
    return numbers


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    problems = []
    if not sources or not isinstance(sources, list):
        return ["no sources in sources.json"]

    seen_urls = {}
    source_by_n = {}
    for idx, s in enumerate(sources):
        if not isinstance(s, dict):
            problems.append(f"source at index {idx} is not an object")
            continue
        n = s.get("n")
        if not isinstance(n, int):
            problems.append(f"source n={n!r} must be an integer")
        elif n in source_by_n:
            problems.append(f"duplicate source number n={n}")
        else:
            source_by_n[n] = s

        url = s.get("url")
        if not isinstance(url, str) or not (url.startswith("http://") or url.startswith("https://")):
            problems.append(f"source [{n}] url must start with http:// or https://: {url!r}")
        else:
            if url in seen_urls:
                problems.append(f"duplicate url in sources.json: {url} (n={seen_urls[url]} and n={n})")
            else:
                seen_urls[url] = n

    matches = list(_REF_HEADING.finditer(report_text))
    if not matches:
        problems.append("missing '## References' section heading")
        return problems

    ref_heading_match = matches[-1]
    body = report_text[:ref_heading_match.start()]
    references_text = report_text[ref_heading_match.end():]

    # Find cited numbers in the body only (ignoring code spans)
    cited = set()
    segments = _CODE.split(body)
    for i, segment in enumerate(segments):
        if i % 2 == 1:
            continue  # code block or inline code
        for match in _GROUP.finditer(segment):
            for num in _group_numbers(match.group(1)):
                cited.add(num)

    for num in sorted(cited):
        if num not in source_by_n:
            problems.append(f"[{num}] cited in body but missing from sources.json")

    for n in sorted(source_by_n.keys()):
        if n not in cited:
            problems.append(f"source [{n}] never cited in report body")

    # References section validation
    ref_lines = {}
    for line in references_text.strip().splitlines():
        line_s = line.strip()
        if not line_s:
            continue
        m = re.match(r"^\[(\d+)\]\s*(.*)$", line_s)
        if not m:
            problems.append(f"reference line does not start with [n]: {line_s[:60]}")
            continue
        num = int(m.group(1))
        if num in ref_lines:
            problems.append(f"duplicate reference line for [{num}]")
        ref_lines[num] = line_s

    for n in sorted(source_by_n.keys()):
        if n not in ref_lines:
            problems.append(f"missing reference line for source [{n}]")

    for num, line_content in sorted(ref_lines.items()):
        if num not in source_by_n:
            problems.append(f"reference [{num}] in References is not in sources.json")
            continue
        raw_urls = _URL.findall(line_content)
        cleaned_urls = [re.sub(r"[),.>]+$", "", u) for u in raw_urls]
        if len(cleaned_urls) != 1:
            problems.append(f"reference line [{num}] must contain exactly one URL (found {len(cleaned_urls)})")
        else:
            expected_url = source_by_n[num].get("url")
            if cleaned_urls[0] != expected_url:
                problems.append(f"reference line [{num}] URL ({cleaned_urls[0]}) does not match sources.json ({expected_url})")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
