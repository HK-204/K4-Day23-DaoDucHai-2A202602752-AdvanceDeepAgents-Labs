"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    s = str(topic or "").strip().lower()
    s = re.sub(r"[^\w]+", "-", s).strip("-")
    s = s[:60].strip("-")
    return s if s else "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f"Please conduct an in-depth scientific literature review on the topic: '{topic}'.\n\n"
        "Follow your required workflow:\n"
        "1. Plan using write_todos: break down the topic into at least 3 distinct sub-questions.\n"
        "2. Delegate each sub-question to a `researcher` subagent via the `task` tool in parallel (at least 3 subagent calls).\n"
        "3. Read notes files from /tmp/work/research/notes/ and merge them into /tmp/work/research/sources.json.\n"
        "   Ensure the sources cover at least 3 distinct source families (arxiv, hf-daily, hf-search, web).\n"
        "4. Write the full report into /tmp/work/report/report.md in English following REPORT_TEMPLATE.md (TL;DR, Background, Themes, Trends and open problems). Do NOT write ## References.\n"
        "5. Execute python3 /tmp/work/research/finalize_citations.py in the sandbox to generate ## References.\n"
        "6. Execute python3 /tmp/work/research/check_citations.py in the sandbox to ensure all citations pass (OK).\n"
        "7. Call citation-checker to spot-check 2-3 key claims."
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}."""
    tool_calls = Counter()
    input_tokens = 0
    output_tokens = 0
    for msg in messages:
        calls = getattr(msg, "tool_calls", None)
        if calls:
            for c in calls:
                name = c.get("name") if isinstance(c, dict) else getattr(c, "name", None)
                if name:
                    tool_calls[name] += 1
        usage = getattr(msg, "usage_metadata", None)
        if isinstance(usage, dict):
            input_tokens += int(usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0) or 0)
            output_tokens += int(usage.get("output_tokens", 0) or usage.get("completion_tokens", 0) or 0)

    subagent_calls = tool_calls.get("task", 0)
    return {
        "model": model_name,
        "elapsed_s": round(elapsed, 1),
        "subagent_calls": subagent_calls,
        "tool_calls": dict(tool_calls),
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
        },
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)

    if not report_bytes or not report_bytes.strip():
        raise RuntimeError(f"Report at {REPORT_PATH} is missing or empty")
    if not sources_bytes or not sources_bytes.strip():
        raise RuntimeError(f"Sources at {SOURCES_PATH} is missing or empty")

    try:
        sources = json.loads(sources_bytes.decode("utf-8"))
        if not isinstance(sources, list):
            raise ValueError("sources.json must be a list")
    except Exception as exc:
        raise RuntimeError(f"Invalid JSON in {SOURCES_PATH}: {exc}") from exc

    slug = slugify(topic)
    source_families = sorted(list({s.get("source") for s in sources if isinstance(s, dict) and s.get("source")}))
    summary = summarize(messages, elapsed, model_name)
    meta = {
        "topic": topic,
        **summary,
        "n_sources": len(sources),
        "source_families": source_families,
    }

    md_path = reports_dir / f"{slug}.md"
    sources_path = reports_dir / f"{slug}.sources.json"
    meta_path = reports_dir / f"{slug}.meta.json"

    md_path.write_bytes(report_bytes)
    sources_path.write_text(json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return md_path


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic_str = str(topic or "").strip()
    if not topic_str:
        print("Usage: python research.py <topic>", file=sys.stderr)
        return 2

    model_name = os.getenv("LAB_MODEL", "default")
    model = make_model()
    start_time = time.monotonic()

    print(f"Starting research on topic: {topic_str}")
    with open_sandbox() as backend:
        print("Sandbox initialized. Preparing directories and scripts...")
        backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
        upload(backend, {
            VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
            FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
        })

        agent = build_lead_agent(backend, model)
        print("Lead agent created. Invoking research workflow...")
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": build_prompt(topic_str)}]},
                config={"recursion_limit": 1000},
            )
            messages = result.get("messages", []) if isinstance(result, dict) else []
        except Exception as exc:
            print(f"Agent execution encountered an error: {exc}", file=sys.stderr)
            messages = []

        elapsed = time.monotonic() - start_time
        try:
            out_path = save_outputs(backend, topic_str, messages, elapsed, model_name)
            print(f"SUCCESS: Report saved to {out_path}")
            return 0
        except RuntimeError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
