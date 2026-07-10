"""Conservative Markdown sectioning for OKF bundles."""

from __future__ import annotations

import re

from .schema import SectionAnchor, SectioningResult, SectionSpec

_ATX_RE = re.compile(r"^[ \t]{0,3}(#{1,6})(?!#)[ \t]*(.*?)[ \t]*#*[ \t]*$")
_H1_RE = re.compile(r"(?m)^[ \t]{0,3}#(?!#)[ \t]*(.*?)[ \t]*$")
_BOLD_PIPE_RE = re.compile(r"^\*\*\s*\|(.+?)\s*\*\*$")
_BARE_PIPE_RE = re.compile(r"^\|(.+)$")
_BOLD_RE = re.compile(r"^\*\*(.+?)\*\*$")
_ORDINAL_RE = re.compile(r"^(?:第[一二三四五六七八九十百千万]+[：:]|[一二三四五六七八九十]+、|\d+[.、]\s+)(.+)$")
_PLAYER_TEXT = {"follow", "replay", "share", "like", "close", "your browser does not support video tags"}


def extract_title(md: str) -> str | None:
    match = _H1_RE.search(md)
    if match is None:
        return None
    title = match.group(1).strip()
    return title or None


def split_sections(md: str, strategy: str = "auto") -> SectioningResult:
    lines = md.splitlines(keepends=True)
    h_counts, headings = _scan_atx(lines)
    effective_level = _effective_level(h_counts, strategy)
    if effective_level is None:
        sections = [_whole_doc_section(md)]
        return _result(sections, h_counts, None, lines)

    boundaries = [h["line_index"] for h in headings if h["level"] == effective_level]
    sections: list[SectionSpec] = []
    first = boundaries[0]
    if first > 0 and _has_effective_preamble(lines[:first]):
        sections.append(
            SectionSpec(
                index=0,
                title="preamble",
                heading_path="preamble",
                line_start=1,
                line_end=first,
                body="".join(lines[:first]).rstrip("\r\n"),
                boundary_kind="preamble",
            )
        )

    for pos, start in enumerate(boundaries):
        end = boundaries[pos + 1] if pos + 1 < len(boundaries) else len(lines)
        heading = next(h for h in headings if h["line_index"] == start)
        title = heading["title"] or "(untitled)"
        sections.append(
            SectionSpec(
                index=len(sections),
                title=title,
                heading_path=title,
                line_start=start + 1,
                line_end=max(end, start + 1),
                body="".join(lines[start:end]).rstrip("\r\n"),
                markdown_level=effective_level,
                boundary_kind="atx",
            )
        )
    return _result(sections, h_counts, effective_level, lines)


def _scan_atx(lines: list[str]) -> tuple[dict[int, int], list[dict]]:
    counts = {level: 0 for level in range(2, 7)}
    headings: list[dict] = []
    in_fence = False
    fence = ""
    for i, raw in enumerate(lines):
        stripped = raw.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            if not in_fence:
                in_fence, fence = True, marker
            elif marker == fence:
                in_fence, fence = False, ""
            continue
        if in_fence:
            continue
        match = _ATX_RE.match(raw.rstrip("\r\n"))
        if match is None:
            continue
        level = len(match.group(1))
        title = (match.group(2) or "").strip() or "(untitled)"
        headings.append({"line_index": i, "level": level, "title": title})
        if 2 <= level <= 6:
            counts[level] += 1
    return counts, headings


def _effective_level(counts: dict[int, int], strategy: str) -> int | None:
    if strategy not in {"auto", "auto_atx"}:
        raise ValueError(f"unknown sectioning strategy: {strategy!r}")
    for level in range(2, 7):
        if counts[level] >= 2:
            return level
    return None


def _whole_doc_section(md: str) -> SectionSpec:
    return SectionSpec(
        index=0,
        title="document",
        heading_path="document",
        line_start=1,
        line_end=max(len(md.splitlines()), 1),
        body=md.rstrip("\r\n"),
        section_id="s0001",
        boundary_kind="whole_doc",
    )


def _result(
    sections: list[SectionSpec],
    counts: dict[int, int],
    effective_level: int | None,
    lines: list[str],
) -> SectioningResult:
    anchor_seq = 1
    for i, section in enumerate(sections, start=1):
        section.index = i - 1
        section.section_id = f"s{i:04d}"
        anchors: list[SectionAnchor] = []
        for line_no in range(section.line_start, min(section.line_end, len(lines)) + 1):
            if line_no == section.line_start and section.boundary_kind == "atx":
                continue
            anchor = _anchor_from_line(lines[line_no - 1], line_no, f"a{anchor_seq:04d}", effective_level)
            if anchor:
                anchors.append(anchor)
                anchor_seq += 1
        section.anchors = anchors
    return SectioningResult(
        sections=sections,
        strategy="auto_atx",
        effective_level=effective_level,
        h_counts=counts,
        section_count=len(sections),
        anchor_count=sum(len(s.anchors) for s in sections),
    )


def _anchor_from_line(raw: str, line_no: int, anchor_id: str, effective_level: int | None) -> SectionAnchor | None:
    stripped = raw.strip()
    if not stripped or stripped.lower() in _PLAYER_TEXT or stripped.startswith("!["):
        return None
    atx = _ATX_RE.match(raw.rstrip("\r\n"))
    if atx:
        level = len(atx.group(1))
        if level == 1 or level == effective_level:
            return None
        return _make_anchor(anchor_id, atx.group(2), raw, line_no, "atx_subheading", level)
    for pattern, kind, prefix in (
        (_BOLD_PIPE_RE, "bold_pipe", "|"),
        (_BARE_PIPE_RE, "bare_pipe", "|"),
        (_BOLD_RE, "bold", ""),
        (_ORDINAL_RE, "ordinal", ""),
    ):
        match = pattern.match(stripped)
        if match:
            title = stripped if kind == "ordinal" else prefix + match.group(1)
            return _make_anchor(anchor_id, title, raw, line_no, kind)
    return None


def _make_anchor(
    anchor_id: str,
    title: str,
    raw: str,
    line_no: int,
    kind: str,
    markdown_level: int | None = None,
) -> SectionAnchor | None:
    title = re.sub(r"\s+", " ", title).strip().strip("#").strip()
    if not title or len(title) > 80:
        return None
    return SectionAnchor(anchor_id, title, raw.rstrip("\r\n"), line_no, kind, markdown_level)


def _has_effective_preamble(lines: list[str]) -> bool:
    text = "".join(lines)
    body = re.sub(r"(?m)^---\s*$.*?^---\s*$", "", text, count=1, flags=re.DOTALL)
    body = re.sub(r"(?m)^#(?!#).*?$", "", body)
    body = re.sub(r"(?m)^>.*?$", "", body)
    return bool(body.strip("\r\n -"))
