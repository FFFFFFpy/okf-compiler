"""Collect relative Markdown and HTML media into a portable OKF bundle."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

from .schema import IMAGES_DIR, MEDIA_DIR, AssetRef

_MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((?!https?://|data:)([^)]+)\)", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<(?P<tag>video|source|audio|img)\b[^>]*>", re.IGNORECASE)
_HTML_ATTR_RE = re.compile(
    r"(?P<prefix>\b(?P<attr>src|poster)\s*=\s*)"
    r"(?P<quote>[\"'])(?P<path>[^\"']+)(?P=quote)",
    re.IGNORECASE,
)
_IMAGE_SUFFIXES = {".apng", ".avif", ".bmp", ".gif", ".ico", ".jpg", ".jpeg", ".png", ".svg", ".webp"}
_AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}
_VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def collect_assets(
    section_bodies: list[str],
    source_markdown: str,
    source_dir: Path,
    workdir: Path,
) -> tuple[list[str], str, list[AssetRef], list[str]]:
    warnings: list[str] = []
    refs: list[AssetRef] = []
    assigned: dict[tuple[str, str], str] = {}
    taken: dict[str, set[str]] = {IMAGES_DIR: set(), MEDIA_DIR: set()}
    source_dir = Path(source_dir).resolve()
    workdir = Path(workdir)

    def register(rel_path: str, *, alt: str, kind: str, collect_ref: bool) -> str | None:
        if _is_external_or_absolute(rel_path):
            return None
        norm = _normalize_rel(rel_path)
        src = _resolve_under(rel_path, source_dir)
        bundle_dir = IMAGES_DIR if kind == "image" else MEDIA_DIR
        if src is None:
            if collect_ref:
                warnings.append(f"asset path escapes source dir: {rel_path}")
            return None
        if not src.is_file():
            if collect_ref:
                warnings.append(f"relative asset not found: {rel_path}")
                refs.append(AssetRef(rel_path, "", str(src), False, alt, kind, bundle_dir))
            return None

        key = (bundle_dir, norm)
        dest_name = assigned.get(key)
        if dest_name is None:
            dest_name = _unique_name(src.name, norm, taken[bundle_dir])
            assigned[key] = dest_name
            taken[bundle_dir].add(dest_name)
            destination = workdir / bundle_dir / dest_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, destination)
        if collect_ref:
            refs.append(AssetRef(rel_path, dest_name, str(src), True, alt, kind, bundle_dir))
        return f"../{bundle_dir}/{dest_name}"

    def rewrite(text: str, *, collect_refs: bool) -> str:
        def markdown_repl(match: re.Match[str]) -> str:
            alt, rel_path = match.group(1), match.group(2)
            replacement = register(rel_path, alt=alt, kind="image", collect_ref=collect_refs)
            return f"![{alt}]({replacement})" if replacement else match.group(0)

        text = _MARKDOWN_IMAGE_RE.sub(markdown_repl, text)

        def html_tag_repl(tag_match: re.Match[str]) -> str:
            tag = tag_match.group("tag").lower()

            def html_attr_repl(attr_match: re.Match[str]) -> str:
                rel_path = attr_match.group("path")
                attr = attr_match.group("attr").lower()
                kind = _asset_kind(tag, attr, rel_path)
                replacement = register(rel_path, alt=attr, kind=kind, collect_ref=collect_refs)
                if replacement is None:
                    return attr_match.group(0)
                quote = attr_match.group("quote")
                return f"{attr_match.group('prefix')}{quote}{replacement}{quote}"

            return _HTML_ATTR_RE.sub(html_attr_repl, tag_match.group(0))

        return _HTML_TAG_RE.sub(html_tag_repl, text)

    rewritten_sections = [rewrite(body, collect_refs=True) for body in section_bodies]
    normalized_source = rewrite(source_markdown, collect_refs=False)
    return rewritten_sections, normalized_source, refs, warnings


def collect_images(
    section_bodies: list[str], source_markdown: str, source_dir: Path, images_dir: Path
) -> tuple[list[str], str, list[AssetRef], list[str]]:
    images_dir = Path(images_dir)
    try:
        workdir = images_dir.parents[1]
    except IndexError:
        workdir = images_dir.parent
    return collect_assets(section_bodies, source_markdown, source_dir, workdir)


def count_missing(refs: list[AssetRef]) -> int:
    return sum(1 for ref in refs if not ref.found)


def _asset_kind(tag: str, attr: str, rel_path: str) -> str:
    suffix = Path(rel_path.split("?", 1)[0].split("#", 1)[0]).suffix.lower()
    if tag == "img" or attr == "poster" or suffix in _IMAGE_SUFFIXES:
        return "image"
    if tag == "audio" or suffix in _AUDIO_SUFFIXES:
        return "audio"
    if tag == "video" or suffix in _VIDEO_SUFFIXES:
        return "video"
    return "media"


def _is_external_or_absolute(rel_path: str) -> bool:
    value = rel_path.strip().strip('"').strip("'")
    lower = value.lower()
    return (
        not value
        or lower.startswith(("http://", "https://", "data:", "//", "#"))
        or value.startswith(("/", "\\"))
        or bool(re.match(r"^[A-Za-z]:", value))
    )


def _resolve_under(rel_path: str, source_dir: Path) -> Path | None:
    value = rel_path.strip().strip('"').strip("'")
    if _is_external_or_absolute(value):
        return None
    value = value.split("?", 1)[0].split("#", 1)[0]
    candidate = (source_dir / value).resolve()
    try:
        candidate.relative_to(source_dir)
    except ValueError:
        return None
    return candidate


def _normalize_rel(rel_path: str) -> str:
    value = rel_path.strip().strip('"').strip("'").replace("\\", "/")
    value = value.split("?", 1)[0].split("#", 1)[0]
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
