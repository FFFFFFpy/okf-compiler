"""Collect relative Markdown images into a portable OKF bundle."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

from .schema import IMAGES_DIR, ImageRef

_RELATIVE_RE = re.compile(r"!\[([^\]]*)\]\((?!https?://|data:)([^)]+)\)")
_REWRITTEN_PREFIX = f"../{IMAGES_DIR}/"


def collect_images(
    section_bodies: list[str], source_markdown: str, source_dir: Path, images_dir: Path
) -> tuple[list[str], str, list[ImageRef], list[str]]:
    """Copy referenced images and rewrite both section and normalized source links."""
    warnings: list[str] = []
    refs: list[ImageRef] = []
    assigned: dict[str, str] = {}
    taken: set[str] = set()

    def rewrite(text: str, *, collect_refs: bool) -> str:
        parts: list[str] = []
        cursor = 0
        for match in _RELATIVE_RE.finditer(text):
            parts.append(text[cursor : match.start()])
            alt, rel_path = match.group(1), match.group(2)
            norm = _normalize_rel(rel_path)
            src = _resolve_under(rel_path, source_dir)
            replacement = match.group(0)
            if src is None:
                if collect_refs:
                    warnings.append(f"image path escapes source dir: {rel_path}")
            elif not src.is_file():
                if collect_refs:
                    warnings.append(f"relative image not found: {rel_path}")
                    refs.append(ImageRef(rel_path, "", str(src), False, alt))
            else:
                dest_name = assigned.get(norm)
                if dest_name is None:
                    dest_name = _unique_name(src.name, norm, taken)
                    assigned[norm] = dest_name
                    taken.add(dest_name)
                    images_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, images_dir / dest_name)
                if collect_refs:
                    refs.append(ImageRef(rel_path, dest_name, str(src), True, alt))
                replacement = f"![{alt}]({_REWRITTEN_PREFIX}{dest_name})"
            parts.append(replacement)
            cursor = match.end()
        parts.append(text[cursor:])
        return "".join(parts)

    rewritten_sections = [rewrite(body, collect_refs=True) for body in section_bodies]
    normalized_source = rewrite(source_markdown, collect_refs=False)
    return rewritten_sections, normalized_source, refs, warnings


def count_missing(refs: list[ImageRef]) -> int:
    return sum(1 for ref in refs if not ref.found)


def _resolve_under(rel_path: str, source_dir: Path) -> Path | None:
    value = rel_path.strip().strip('"').strip("'")
    if not value or value.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", value):
        return None
    candidate = (source_dir / value).resolve()
    try:
        candidate.relative_to(source_dir.resolve())
    except ValueError:
        return None
    return candidate


def _normalize_rel(rel_path: str) -> str:
    value = rel_path.strip().strip('"').strip("'").replace("\\", "/")
    return "/".join(part for part in value.split("/") if part and part != ".")


def _unique_name(basename: str, norm_rel: str, taken: set[str]) -> str:
    if basename not in taken:
        return basename
    digest = hashlib.sha1(norm_rel.encode()).hexdigest()[:8]
    candidate = f"{digest}_{basename}"
    n = 1
    while candidate in taken:
        candidate = f"{digest}_{n}_{basename}"
        n += 1
    return candidate
