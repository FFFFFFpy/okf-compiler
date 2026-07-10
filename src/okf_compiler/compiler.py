"""Fresh, isolated Markdown-to-OKF compilation."""

from __future__ import annotations

import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .assets import collect_images, count_missing
from .atomic import atomic_write_json
from .bundle import write_zip
from .frontmatter import parse as parse_frontmatter
from .llm import LLMClient, LLMConfig, extract, redact_secrets
from .markdown import extract_title, split_sections
from .render import render_bundle
from .schema import Extracts

OKF_ZIP_SUFFIX = ".okf.zip"
DEFAULT_REPORT_NAME = "batch_report.json"
_MD_SUFFIXES = {".md", ".markdown"}


@dataclass
class CompileOptions:
    workdir: Path | None = None
    keep_workdir: bool = False
    overwrite: bool = False
    no_llm: bool = False
    language: str = "zh"
    max_concepts: int = 12
    max_entities: int = 12
    llm_config: LLMConfig | None = None


@dataclass
class CompileResult:
    input_path: Path
    output_path: Path | None
    ok: bool
    skipped: bool = False
    error: str = ""
    manifest: dict | None = None
    warnings: list[str] = field(default_factory=list)
    workdir_path: Path | None = None

    def to_report(
        self,
        *,
        input_root: Path | None = None,
        output_root: Path | None = None,
    ) -> dict:
        return {
            "input": _relative(self.input_path, input_root),
            "output": _relative(self.output_path, output_root) if self.output_path else None,
            "status": "skipped" if self.skipped else ("ok" if self.ok else "failed"),
            "error": self.error or None,
            "counts": (self.manifest or {}).get("counts"),
            "warnings": self.warnings,
        }


@dataclass
class BatchReport:
    total: int = 0
    ok: int = 0
    skipped: int = 0
    failed: int = 0
    results: list[CompileResult] = field(default_factory=list)
    input_root: Path | None = None
    output_root: Path | None = None

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "ok": self.ok,
            "skipped": self.skipped,
            "failed": self.failed,
            "results": [
                result.to_report(
                    input_root=self.input_root,
                    output_root=self.output_root,
                )
                for result in self.results
            ],
        }


def compile_one(input_md: Path, out: Path, opts: CompileOptions) -> CompileResult:
    input_md = Path(input_md).resolve()
    out = Path(out).resolve()
    if not input_md.is_file():
        return CompileResult(input_md, out, False, error=f"input not found: {input_md}")
    if input_md.suffix.lower() not in _MD_SUFFIXES:
        return CompileResult(input_md, out, False, error="input must be Markdown")
    if out.exists() and not opts.overwrite:
        return CompileResult(input_md, out, False, skipped=True, error="output exists")

    workdir = _create_workdir(opts)
    warnings: list[str] = []
    try:
        markdown = input_md.read_text(encoding="utf-8")
        source_metadata = parse_frontmatter(markdown)
        sectioning = split_sections(markdown)
        sections = sectioning.sections
        title = extract_title(markdown) or str(source_metadata.get("title") or input_md.stem)
        images_dir = workdir / "assets" / "images"
        rewritten, normalized_source, image_refs, asset_warnings = collect_images(
            [section.body for section in sections],
            markdown,
            input_md.parent,
            images_dir,
        )
        warnings.extend(asset_warnings)
        for section, body in zip(sections, rewritten):
            section.body = body

        extracts = Extracts()
        llm_enabled = False
        model = None
        if not opts.no_llm and opts.llm_config and opts.llm_config.is_configured():
            try:
                client = LLMClient(opts.llm_config)
                model = client.model
                extracts = extract(
                    client,
                    markdown,
                    sections,
                    language=opts.language,
                    max_concepts=opts.max_concepts,
                    max_entities=opts.max_entities,
                )
                llm_enabled = True
            except Exception as exc:  # noqa: BLE001
                warnings.append(
                    redact_secrets(f"llm disabled: {type(exc).__name__}: {exc}")
                )
        missing = count_missing(image_refs)
        if missing:
            warnings.append(f"assets: {missing} referenced image(s) not found")

        manifest = render_bundle(
            workdir,
            original_markdown=markdown,
            normalized_markdown=normalized_source,
            sections=sections,
            image_refs=image_refs,
            extracts=extracts,
            original_filename=input_md.name,
            title=title,
            language=opts.language,
            llm_enabled=llm_enabled,
            model=model,
            warnings=warnings,
            sectioning=sectioning,
            source_metadata=_source_metadata(source_metadata),
        )
        write_zip(workdir, out)
        return CompileResult(
            input_md,
            out,
            True,
            manifest=manifest,
            warnings=warnings + extracts.warnings,
            workdir_path=workdir if opts.keep_workdir else None,
        )
    except Exception as exc:  # noqa: BLE001
        return CompileResult(
            input_md,
            out,
            False,
            error=redact_secrets(f"{type(exc).__name__}: {exc}"),
            workdir_path=workdir if opts.keep_workdir else None,
        )
    finally:
        if not opts.keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


def compile_dir(
    input_dir: Path,
    out_dir: Path,
    opts: CompileOptions,
    *,
    mode: str = "auto",
    glob_pattern: str = "*.md",
    recursive: bool = True,
    skip_existing: bool = True,
    max_workers: int = 1,
    fail_fast: bool = False,
    report_path: Path | None = None,
    progress_callback: Callable[[CompileResult, int, int], None] | None = None,
) -> BatchReport:
    input_dir = Path(input_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs, skipped_dirs = enumerate_inputs(
        input_dir,
        mode=mode,
        glob_pattern=glob_pattern,
        recursive=recursive,
    )
    report = BatchReport(
        total=len(inputs) + len(skipped_dirs),
        input_root=input_dir,
        output_root=out_dir,
    )
    completed = 0

    def record(result: CompileResult) -> None:
        nonlocal completed
        report.results.append(result)
        if result.ok:
            report.ok += 1
        elif result.skipped:
            report.skipped += 1
        else:
            report.failed += 1
        completed += 1
        if progress_callback:
            progress_callback(result, completed, report.total)

    for path in skipped_dirs:
        record(
            CompileResult(
                path,
                None,
                False,
                skipped=True,
                error="ambiguous article directory",
            )
        )

    used: set[str] = set()
    pending: list[tuple[Path, Path]] = []
    for md in inputs:
        name = _output_name(md, used, relative_to=input_dir)
        used.add(name)
        target = out_dir / name
        if target.exists() and skip_existing and not opts.overwrite:
            record(
                CompileResult(
                    md,
                    target,
                    False,
                    skipped=True,
                    error="already compiled",
                )
            )
        else:
            pending.append((md, target))

    def run(item: tuple[Path, Path]) -> CompileResult:
        return compile_one(item[0], item[1], opts)

    if max_workers <= 1:
        for item in pending:
            result = run(item)
            record(result)
            if fail_fast and not result.ok and not result.skipped:
                break
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(run, item): item for item in pending}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = CompileResult(
                        futures[future][0],
                        None,
                        False,
                        error=str(exc),
                    )
                record(result)
                if fail_fast and not result.ok and not result.skipped:
                    for other in futures:
                        other.cancel()
                    break

    report.results.sort(key=lambda item: item.input_path.as_posix())
    atomic_write_json(report_path or (out_dir / DEFAULT_REPORT_NAME), report.to_dict())
    return report


def enumerate_inputs(
    input_dir: Path,
    *,
    mode: str,
    glob_pattern: str = "*.md",
    recursive: bool = True,
) -> tuple[list[Path], list[Path]]:
    mode = mode.lower()
    if mode == "flat":
        return _flat_inputs(input_dir, glob_pattern, recursive), []
    if mode == "wechat":
        return _wechat_inputs(input_dir)
    if mode == "auto":
        selected, skipped = _wechat_inputs(input_dir)
        if selected or skipped:
            return selected, skipped
        return _flat_inputs(input_dir, glob_pattern, recursive), []
    raise ValueError("mode must be auto, flat, or wechat")


def _wechat_inputs(input_dir: Path) -> tuple[list[Path], list[Path]]:
    selected: list[Path] = []
    skipped: list[Path] = []
    if not input_dir.is_dir():
        return selected, skipped
    for child in sorted(input_dir.iterdir()):
        if not child.is_dir() or child.name.endswith(".okf"):
            continue
        manifest = child / "manifest.json"
        candidate = _manifest_main(manifest, child) if manifest.is_file() else None
        if candidate:
            selected.append(candidate)
            continue
        mds = sorted(path for path in child.glob("*.md") if path.is_file())
        preferred = next((path for path in mds if path.stem == child.name), None)
        if preferred:
            selected.append(preferred)
        elif (child / "document.md") in mds:
            selected.append(child / "document.md")
        elif len(mds) == 1:
            selected.append(mds[0])
        elif len(mds) > 1:
            skipped.append(child)
    return sorted(selected), sorted(skipped)


def _manifest_main(manifest: Path, root: Path) -> Path | None:
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for key in ("main", "main_source", "document", "markdown"):
        value = data.get(key)
        if not isinstance(value, str):
            continue
        path = (root / value).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            continue
        if path.is_file() and path.suffix.lower() in _MD_SUFFIXES:
            return path
    return None


def _flat_inputs(input_dir: Path, glob_pattern: str, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    out = []
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in _MD_SUFFIXES:
            continue
        rel = path.relative_to(input_dir)
        excluded = {
            ".git",
            ".venv",
            "node_modules",
            "__pycache__",
        }
        if any(part in excluded or part.endswith(".okf") for part in rel.parts[:-1]):
            continue
        if glob_pattern not in {"*.md", "*"} and not path.match(glob_pattern):
            continue
        out.append(path)
    return sorted(out)


def _output_name(md: Path, used: set[str], *, relative_to: Path) -> str:
    base = md.stem + OKF_ZIP_SUFFIX
    if base not in used:
        return base
    parent = md.relative_to(relative_to).parent.name
    candidate = (
        f"{parent}__{base}"
        if parent
        else f"{md.stem}_{len(used)}{OKF_ZIP_SUFFIX}"
    )
    n = 1
    while candidate in used:
        candidate = f"{parent}__{md.stem}_{n}{OKF_ZIP_SUFFIX}"
        n += 1
    return candidate


def _create_workdir(opts: CompileOptions) -> Path:
    if opts.workdir:
        Path(opts.workdir).mkdir(parents=True, exist_ok=True)
        return Path(
            tempfile.mkdtemp(prefix="okf-compile-", dir=opts.workdir)
        ).resolve()
    return Path(tempfile.mkdtemp(prefix="okf-compile-")).resolve()


def _source_metadata(metadata: dict) -> dict:
    mapping = {
        "url": metadata.get("source_url") or metadata.get("url"),
        "author": metadata.get("author"),
        "published_at": metadata.get("published_at") or metadata.get("publish_time"),
        "converter": metadata.get("converted_by"),
    }
    return {
        key: value
        for key, value in mapping.items()
        if value not in (None, "")
    }


def _relative(path: Path | None, root: Path | None) -> str | None:
    if path is None:
        return None
    try:
        if root:
            return path.resolve().relative_to(root.resolve()).as_posix()
        return path.as_posix()
    except ValueError:
        return path.as_posix()
