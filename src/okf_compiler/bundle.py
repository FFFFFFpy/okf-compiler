"""Safely assemble a rendered OKF work directory into a ZIP archive."""

from __future__ import annotations

import zipfile
from pathlib import Path

FORBIDDEN_PATHS = {"raw/original.mhtml", "raw/"}


def write_zip(workdir: Path, out_path: Path) -> Path:
    workdir = Path(workdir).resolve()
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    entries = _collect_entries(workdir)
    _assert_clean(entries)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel, path in entries:
            if path.is_dir():
                zf.writestr(rel, "")
            else:
                zf.write(path, rel)
    return out_path


def _collect_entries(workdir: Path) -> list[tuple[str, Path]]:
    entries = []
    for path in sorted(workdir.rglob("*")):
        rel = path.relative_to(workdir).as_posix()
        entries.append((rel + "/" if path.is_dir() else rel, path))
    return sorted(entries, key=lambda item: item[0])


def _assert_clean(entries: list[tuple[str, Path]]) -> None:
    for rel, _ in entries:
        if rel.startswith(("/", "\\")) or (len(rel) >= 2 and rel[1] == ":"):
            raise ValueError(f"absolute path in bundle: {rel!r}")
        if ".." in rel.split("/"):
            raise ValueError(f"path traversal in bundle: {rel!r}")
        for forbidden in FORBIDDEN_PATHS:
            if rel == forbidden or rel.startswith(forbidden):
                raise ValueError(f"forbidden path in Markdown-only bundle: {rel!r}")
