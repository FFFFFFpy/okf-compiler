"""Render the in-memory OKF model into a portable directory tree."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import frontmatter
from .atomic import atomic_write_json, atomic_write_text
from .llm import redact_secrets
from .schema import (
    ARTICLE_MD,
    ASSETS_DIR,
    CLAIMS_DIR,
    CONCEPTS_DIR,
    ENTITIES_DIR,
    EXTRACTS_DIR,
    IMAGES_DIR,
    INDEX_MD,
    LOG_MD,
    MANIFEST_JSON,
    MEDIA_DIR,
    OKF_BUNDLE_KIND,
    OKF_BUNDLE_TYPE,
    OKF_FORMAT,
    OKF_VERSION,
    OKF_YAML,
    ORIGINAL_MD,
    PROPOSED_EDGES_JSONL,
    RELATIONS_DIR,
    SECTION_TYPE,
    SECTIONS_DIR,
    SOURCE_MAP_JSON,
    SOURCES_DIR,
    SUMMARY_MD,
    AssetRef,
    Extracts,
    SectioningResult,
    SectionSpec,
    evidence_to_dict,
    manifest_compiler_block,
    okf_yaml_text,
    slugify,
)


def render_bundle(
    workdir: Path,
    *,
    original_markdown: str,
    normalized_markdown: str,
    sections: list[SectionSpec],
    asset_refs: list[AssetRef],
    extracts: Extracts,
    original_filename: str,
    title: str,
    language: str,
    llm_enabled: bool,
    model: str | None,
    warnings: list[str],
    sectioning: SectioningResult,
    source_metadata: dict | None = None,
) -> dict:
    workdir = Path(workdir)
    _ensure_dirs(workdir)
    atomic_write_text(workdir / OKF_YAML, okf_yaml_text(title))
    atomic_write_text(workdir / ORIGINAL_MD, original_markdown)
    atomic_write_text(workdir / ARTICLE_MD, normalized_markdown)
    for section in sections:
        _write_section(workdir, section)
    _write_summary(workdir, extracts, title)
    _write_concepts(workdir, extracts)
    _write_entities(workdir, extracts)
    _write_relations(workdir, extracts)
    atomic_write_json(workdir / SOURCE_MAP_JSON, _source_map(sections, asset_refs))
    atomic_write_text(workdir / INDEX_MD, _index_md(title, sections, extracts))
    atomic_write_text(
        workdir / LOG_MD,
        _log_md(title, llm_enabled, warnings + extracts.warnings, extracts.stage_stats),
    )

    counts = {
        "sections": len(sections),
        "concepts": len(extracts.concepts),
        "entities": len(extracts.entities),
        "relations": len(extracts.relations),
        "images": sum(ref.found and ref.kind == "image" for ref in asset_refs),
        "media": sum(ref.found and ref.kind != "image" for ref in asset_refs),
        "missing_assets": sum(not ref.found for ref in asset_refs),
    }
    manifest = {
        "format": OKF_FORMAT,
        "version": OKF_VERSION,
        "bundle_type": OKF_BUNDLE_TYPE,
        "kind": OKF_BUNDLE_KIND,
        "title": title,
        "entry": INDEX_MD,
        "main_source": ARTICLE_MD,
        "original_source": ORIGINAL_MD,
        "language": language or "en",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
        "sectioning": sectioning.to_manifest(),
        "source": source_metadata or {},
        "inputs": {
            "source_md": ARTICLE_MD,
            "original_filename": original_filename,
            "markdown_bytes": len(original_markdown.encode("utf-8")),
        },
        "compiler": manifest_compiler_block(llm_enabled=llm_enabled, model=model),
        "extraction": {
            "status": "degraded" if extracts.degraded else "ok",
            "stages": extracts.stage_stats,
        },
        "warnings": [redact_secrets(w) for w in warnings + extracts.warnings],
    }
    atomic_write_json(workdir / MANIFEST_JSON, manifest)
    return manifest


def _ensure_dirs(workdir: Path) -> None:
    for path in (
        SOURCES_DIR,
        SECTIONS_DIR,
        EXTRACTS_DIR,
        CONCEPTS_DIR,
        ENTITIES_DIR,
        CLAIMS_DIR,
        RELATIONS_DIR,
        ASSETS_DIR,
        IMAGES_DIR,
        MEDIA_DIR,
    ):
        (workdir / path).mkdir(parents=True, exist_ok=True)


def _write_section(workdir: Path, section: SectionSpec) -> None:
    lines = [
        frontmatter.kv_line("type", SECTION_TYPE),
        frontmatter.kv_line("title", section.title),
        frontmatter.kv_line("source", ARTICLE_MD),
        frontmatter.kv_line("section_id", section.section_id),
        frontmatter.kv_line("heading_path", section.heading_path),
        f"line_start: {section.line_start}",
        f"line_end: {section.line_end}",
    ]
    atomic_write_text(workdir / section.filename, frontmatter.block(lines) + section.body + "\n")


def _evidence_lines(evidence) -> list[str]:
    return [
        frontmatter.kv_line("section_id", evidence.section_id if evidence else ""),
        frontmatter.kv_line("heading_path", evidence.heading_path if evidence else ""),
        f"line_start: {evidence.line_start if evidence else 0}",
        f"line_end: {evidence.line_end if evidence else 0}",
        frontmatter.kv_line("evidence_quote", evidence.quote if evidence else ""),
    ]


def _write_summary(workdir: Path, extracts: Extracts, title: str) -> None:
    body = extracts.summary.strip() or "_(no summary available)_"
    fm = frontmatter.block(
        [
            frontmatter.kv_line("type", "Document Summary"),
            frontmatter.kv_line("title", title),
            frontmatter.kv_line("source", ARTICLE_MD),
        ]
    )
    atomic_write_text(workdir / SUMMARY_MD, fm + body + "\n")


def _write_concepts(workdir: Path, extracts: Extracts) -> None:
    for concept in extracts.concepts:
        lines = [
            frontmatter.kv_line("type", "Local Concept"),
            frontmatter.kv_line("title", concept.name),
            frontmatter.kv_line("source", ARTICLE_MD),
            frontmatter.kv_line("scope", "local"),
            *_evidence_lines(concept.evidence),
        ]
        if concept.confidence is not None:
            lines.append(f"confidence: {concept.confidence}")
        path = workdir / CONCEPTS_DIR / f"{slugify(concept.name) or 'concept'}.md"
        atomic_write_text(path, frontmatter.block(lines) + concept.description.strip() + "\n")


def _write_entities(workdir: Path, extracts: Extracts) -> None:
    for entity in extracts.entities:
        lines = [
            frontmatter.kv_line("type", f"Local {entity.entity_type.title()}"),
            frontmatter.kv_line("title", entity.name),
            frontmatter.kv_line("source", ARTICLE_MD),
            frontmatter.kv_line("scope", "local"),
            frontmatter.list_line("aliases", entity.aliases),
            *_evidence_lines(entity.evidence),
        ]
        if entity.confidence is not None:
            lines.append(f"confidence: {entity.confidence}")
        path = workdir / ENTITIES_DIR / f"{slugify(entity.name) or 'entity'}.md"
        atomic_write_text(path, frontmatter.block(lines) + entity.description.strip() + "\n")


def _write_relations(workdir: Path, extracts: Extracts) -> None:
    lines = []
    for relation in extracts.relations:
        lines.append(
            json.dumps(
                {
                    "subject": relation.subject,
                    "relation": relation.relation,
                    "object": relation.object,
                    "note": relation.note,
                    "evidence": evidence_to_dict(relation.evidence),
                },
                ensure_ascii=False,
            )
        )
    atomic_write_text(workdir / PROPOSED_EDGES_JSONL, "\n".join(lines) + ("\n" if lines else ""))


def _index_md(title: str, sections: list[SectionSpec], extracts: Extracts) -> str:
    lines = [
        frontmatter.block(
            [
                frontmatter.kv_line("type", "OKF Bundle Index"),
                frontmatter.kv_line("title", title),
                frontmatter.kv_line("source", ARTICLE_MD),
            ]
        ).rstrip(),
        "",
        f"# {title}",
        "",
        "## Contents",
        "",
    ]
    lines.extend(f"- [{section.title}]({section.filename})" for section in sections)
    lines += ["", "## Extracts", "", f"- [Summary]({SUMMARY_MD})"]
    if extracts.concepts:
        lines.append(f"- Concepts: {len(extracts.concepts)}")
    if extracts.entities:
        lines.append(f"- Entities: {len(extracts.entities)}")
    return "\n".join(lines).rstrip() + "\n"


def _log_md(title: str, llm_enabled: bool, warnings: list[str], stage_stats: dict[str, dict]) -> str:
    lines = [f"# Compilation Log: {title}", "", f"- LLM enabled: {str(llm_enabled).lower()}"]
    if stage_stats:
        lines += ["", "## Extraction", ""]
        for stage, stats in stage_stats.items():
            counts = ""
            if "returned" in stats:
                counts = (
                    f", returned={stats.get('returned', 0)}, accepted={stats.get('accepted', 0)}, "
                    f"rejected={stats.get('rejected', 0)}"
                )
            lines.append(f"- {stage}: {stats.get('status', 'unknown')}{counts}")
    if warnings:
        lines += ["", "## Warnings", ""] + [f"- {redact_secrets(w)}" for w in warnings]
    return "\n".join(lines).rstrip() + "\n"


def _source_map(sections: list[SectionSpec], asset_refs: list[AssetRef]) -> dict:
    assets = [
        {
            "kind": ref.kind,
            "original_ref": ref.original_ref,
            "bundle_ref": ref.bundle_ref,
            "found": ref.found,
            "alt": ref.alt,
        }
        for ref in asset_refs
    ]
    return {
        "sections": [
            {
                "section_id": s.section_id,
                "file": s.filename,
                "heading_path": s.heading_path,
                "line_start": s.line_start,
                "line_end": s.line_end,
                "anchors": [a.__dict__ for a in s.anchors],
            }
            for s in sections
        ],
        "assets": assets,
        "images": [item for item in assets if item["kind"] == "image"],
    }
