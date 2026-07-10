"""YAML frontmatter helpers used by OKF Markdown files."""

from __future__ import annotations

import json
import re

import yaml


def kv_line(key: str, value: object) -> str:
    return f"{key}: {json.dumps(value, ensure_ascii=False)}"


def list_line(key: str, items) -> str:
    return f"{key}: {json.dumps(list(items), ensure_ascii=False)}"


def block(lines: list[str]) -> str:
    return "---\n" + "\n".join(lines) + "\n---\n\n"


def split(text: str) -> tuple[str, str] | None:
    if not text.startswith("---"):
        return None
    match = re.search(r"\n---[ \t]*(?:\r?\n|$)", text[3:])
    if match is None:
        return None
    end = 3 + match.end()
    return text[:end], text[end:]


def parse(text: str) -> dict:
    parts = split(text)
    if parts is None:
        return {}
    raw = parts[0]
    inner = raw[3:]
    close = inner.rfind("\n---")
    if close >= 0:
        inner = inner[:close]
    try:
        data = yaml.safe_load(inner)
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}
