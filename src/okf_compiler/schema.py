"""OKF Bundle paths, identities, data structures, and evidence resolution."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field, replace
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

OKF_FORMAT = "okf-bundle"
OKF_VERSION = 1
OKF_BUNDLE_KIND = "markdown_article"
OKF_BUNDLE_TYPE = "markdown_article"
COMPILER_NAME = "okf-compiler"
COMPILER_MODE = "fresh-single-md"

OKF_YAML = "okf.yaml"
MANIFEST_JSON = "manifest.json"
INDEX_MD = "index.md"
LOG_MD = "log.md"
SOURCES_DIR = "sources"
ORIGINAL_MD = "sources/original.md"
ARTICLE_MD = "sources/article.md"
SECTIONS_DIR = "sections"
EXTRACTS_DIR = "extracts"
SUMMARY_MD = "extracts/summary.md"
CONCEPTS_DIR = "extracts/concepts"
ENTITIES_DIR = "extracts/entities"
CLAIMS_DIR = "extracts/claims"
RELATIONS_DIR = "relations"
PROPOSED_EDGES_JSONL = "relations/proposed_edges.jsonl"
ASSETS_DIR = "assets"
IMAGES_DIR = "assets/images"
MEDIA_DIR = "assets/media"
SOURCE_MAP_JSON = "source_map.json"
SECTION_INDEX_WIDTH = 2
SECTION_TYPE = "Section"

_DOUBLE_QUOTES = frozenset('“”„‟＂〝〞〟「」『』')
_SINGLE_QUOTES = frozenset("‘’‚‛＇")


@dataclass
class SectionAnchor:
    anchor_id: str
    title: str
    raw: str
    line_no: int
    kind: str
    markdown_level: int | None = None


@dataclass
class SectionSpec:
    index: int
    title: str
    heading_path: str
    line_start: int
    line_end: int
    body: str
    section_id: str = ""
    markdown_level: int | None = None
    boundary_kind: str = "atx"
    anchors: list[SectionAnchor] = field(default_factory=list)

    @property
    def filename(self) -> str:
        slug = slugify(self.title) or "section"
        return f"{SECTIONS_DIR}/{self.index:0{SECTION_INDEX_WIDTH}d}_{slug}.md"


@dataclass
class SectioningResult:
    sections: list[SectionSpec]
    strategy: str
    effective_level: int | None
    h_counts: dict[int, int]
    section_count: int
    anchor_count: int

    def __iter__(self):
        return iter(self.sections)

    def __len__(self) -> int:
        return len(self.sections)

    def __getitem__(self, index):
        return self.sections[index]

    def __bool__(self) -> bool:
        return bool(self.sections)

    def to_manifest(self) -> dict:
        out = {
            "strategy": self.strategy,
            "effective_level": self.effective_level,
            "section_count": self.section_count,
            "anchor_count": self.anchor_count,
        }
        for level in range(2, 7):
            out[f"h{level}_count"] = self.h_counts.get(level, 0)
        return out


@dataclass
class AssetRef:
    original_ref: str
    dest_name: str
    source_path: str
    found: bool
    alt: str = ""
    kind: str = "image"
    bundle_dir: str = IMAGES_DIR

    @property
    def bundle_ref(self) -> str | None:
        return f"{self.bundle_dir}/{self.dest_name}" if self.found else None


ImageRef = AssetRef


@dataclass
class Evidence:
    heading_path: str = ""
    line_start: int = 0
    line_end: int = 0
    section_id: str = ""
    quote: str = ""

    def is_valid(self) -> bool:
        return bool(self.heading_path) and self.line_start >= 1 and self.line_end >= self.line_start


@dataclass
class EvidenceValidation:
    valid: bool
    reason: str
    evidence: Evidence | None = None
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "evidence": evidence_to_dict(self.evidence),
            "details": self.details,
        }


@dataclass
class ConceptExtract:
    name: str
    description: str
    evidence: Evidence | None = None
    confidence: float | None = None


@dataclass
class EntityExtract:
    name: str
    entity_type: str
    description: str
    aliases: list[str] = field(default_factory=list)
    evidence: Evidence | None = None
    confidence: float | None = None


@dataclass
class ProposedEdge:
    subject: str
    relation: str
    object: str
    evidence: Evidence | None = None
    note: str = ""


@dataclass
class Extracts:
    summary: str = ""
    concepts: list[ConceptExtract] = field(default_factory=list)
    entities: list[EntityExtract] = field(default_factory=list)
    relations: list[ProposedEdge] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stage_stats: dict[str, dict] = field(default_factory=dict)
    validation: list[dict] = field(default_factory=list)

    @property
    def degraded(self) -> bool:
        return any(
            str(stats.get("status", "")).lower() in {"degraded", "failed"}
            for stats in self.stage_stats.values()
        )


def evidence_to_dict(evidence: Evidence | None) -> dict | None:
    if evidence is None:
        return None
    return {
        "section_id": evidence.section_id,
        "heading_path": evidence.heading_path,
        "line_start": evidence.line_start,
        "line_end": evidence.line_end,
        "quote": evidence.quote,
    }


def resolve_evidence(
    evidence: Evidence | None,
    sections: list[SectionSpec],
    markdown: str,
) -> EvidenceValidation:
    """Resolve quote-based evidence to deterministic absolute line coordinates."""
    if evidence is None:
        return EvidenceValidation(False, "missing_evidence")

    section_result = _select_section(evidence, sections)
    if isinstance(section_result, EvidenceValidation):
        return section_result
    section = section_result

    quote = evidence.quote.strip()
    if quote:
        return _locate_quote(section, markdown, quote)

    return validate_evidence_detailed(evidence, sections, max(len(markdown.splitlines()), 1))


def validate_evidence_detailed(
    evidence: Evidence | None,
    sections: list[SectionSpec],
    total_lines: int,
) -> EvidenceValidation:
    if evidence is None:
        return EvidenceValidation(False, "missing_evidence")
    if not evidence.heading_path:
        return EvidenceValidation(False, "missing_heading_path", evidence)
    if evidence.line_start < 1:
        return EvidenceValidation(False, "invalid_line_start", evidence)
    if evidence.line_end < evidence.line_start:
        return EvidenceValidation(False, "invalid_line_range", evidence)
    if evidence.line_end > total_lines:
        return EvidenceValidation(
            False,
            "line_end_out_of_document",
            evidence,
            {"total_lines": total_lines},
        )

    section_result = _select_section(evidence, sections)
    if isinstance(section_result, EvidenceValidation):
        return section_result
    section = section_result
    if not (section.line_start <= evidence.line_start <= evidence.line_end <= section.line_end):
        return EvidenceValidation(
            False,
            "line_range_outside_section",
            evidence,
            {
                "section_id": section.section_id,
                "section_line_start": section.line_start,
                "section_line_end": section.line_end,
            },
        )
    return EvidenceValidation(True, "ok_legacy_line_evidence", evidence)


def validate_evidence(evidence: Evidence | None, sections: list[SectionSpec], total_lines: int) -> bool:
    return validate_evidence_detailed(evidence, sections, total_lines).valid


def _select_section(
    evidence: Evidence,
    sections: list[SectionSpec],
) -> SectionSpec | EvidenceValidation:
    if not sections:
        return EvidenceValidation(False, "no_sections", evidence)
    if evidence.section_id:
        matches = [sec for sec in sections if sec.section_id == evidence.section_id]
        if not matches:
            return EvidenceValidation(False, "unknown_section_id", evidence)
        return matches[0]
    if evidence.heading_path:
        matches = [sec for sec in sections if sec.heading_path == evidence.heading_path]
        if not matches:
            return EvidenceValidation(False, "unknown_heading_path", evidence)
        if len(matches) > 1:
            return EvidenceValidation(False, "ambiguous_heading_path", evidence)
        return matches[0]
    return EvidenceValidation(False, "missing_section_locator", evidence)


def _locate_quote(section: SectionSpec, markdown: str, quote: str) -> EvidenceValidation:
    lines = markdown.splitlines(keepends=True)
    section_text = "".join(lines[max(section.line_start - 1, 0) : min(section.line_end, len(lines))])
    frontmatter_end = _frontmatter_end(section, section_text)

    exact_positions = _all_positions(section_text, quote)
    exact_choice = _choose_position(exact_positions, lambda position: position, frontmatter_end)
    if exact_choice is not None:
        start, body_preferred = exact_choice
        reason = "ok_exact_quote_body_preferred" if body_preferred else "ok_exact_quote"
        return _resolved_quote(section, section_text, start, start + len(quote), reason)
    if exact_positions:
        return _ambiguous_quote(section, quote, exact_positions, "exact")

    normalized_text, mapping = _normalize_with_map(section_text)
    normalized_quote, _ = _normalize_with_map(quote)
    normalized_positions = _all_positions(normalized_text, normalized_quote)
    normalized_choice = _choose_position(
        normalized_positions,
        lambda position: mapping[position],
        frontmatter_end,
    )
    if normalized_choice is not None:
        norm_start, body_preferred = normalized_choice
        norm_end = norm_start + len(normalized_quote) - 1
        reason = (
            "ok_normalized_quote_body_preferred"
            if body_preferred
            else "ok_normalized_quote"
        )
        return _resolved_quote(
            section,
            section_text,
            mapping[norm_start],
            mapping[norm_end] + 1,
            reason,
        )
    if normalized_positions:
        return _ambiguous_quote(
            section,
            quote,
            normalized_positions,
            "normalized_typography",
        )
    return EvidenceValidation(
        False,
        "quote_not_found",
        replace(section_evidence(section), quote=quote),
        {"section_id": section.section_id},
    )


def _ambiguous_quote(
    section: SectionSpec,
    quote: str,
    positions: list[int],
    match_mode: str,
) -> EvidenceValidation:
    return EvidenceValidation(
        False,
        "ambiguous_quote",
        replace(section_evidence(section), quote=quote),
        {"matches": len(positions), "match_mode": match_mode},
    )


def _choose_position(
    positions: list[int],
    original_position,
    frontmatter_end: int,
) -> tuple[int, bool] | None:
    if len(positions) == 1:
        return positions[0], False
    if frontmatter_end and positions:
        body_positions = [p for p in positions if original_position(p) >= frontmatter_end]
        if len(body_positions) == 1:
            return body_positions[0], True
    return None


def _frontmatter_end(section: SectionSpec, section_text: str) -> int:
    if section.line_start != 1 or not section_text.startswith("---"):
        return 0
    match = re.search(r"\n---[ \t]*(?:\r?\n|$)", section_text[3:])
    return 3 + match.end() if match else 0


def _resolved_quote(
    section: SectionSpec,
    section_text: str,
    start: int,
    end: int,
    reason: str,
) -> EvidenceValidation:
    while start < end and section_text[start].isspace():
        start += 1
    while end > start and section_text[end - 1].isspace():
        end -= 1
    last_char = max(start, end - 1)
    line_start = section.line_start + section_text[:start].count("\n")
    line_end = section.line_start + section_text[:last_char].count("\n")
    resolved = Evidence(
        section_id=section.section_id,
        heading_path=section.heading_path,
        line_start=line_start,
        line_end=max(line_start, line_end),
        quote=section_text[start:end],
    )
    return EvidenceValidation(True, reason, resolved)


def section_evidence(section: SectionSpec) -> Evidence:
    return Evidence(
        section_id=section.section_id,
        heading_path=section.heading_path,
        line_start=section.line_start,
        line_end=section.line_end,
    )


def _all_positions(text: str, needle: str) -> list[int]:
    if not needle:
        return []
    positions: list[int] = []
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            return positions
        positions.append(index)
        start = index + 1


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    mapping: list[int] = []
    in_space = False
    for index, char in enumerate(text):
        expanded = _normalize_char(char)
        for normalized in expanded:
            if normalized.isspace():
                if chars and not in_space:
                    chars.append(" ")
                    mapping.append(index)
                in_space = True
                continue
            chars.append(normalized)
            mapping.append(index)
            in_space = False
    while chars and chars[-1] == " ":
        chars.pop()
        mapping.pop()
    return "".join(chars), mapping


def _normalize_char(char: str) -> str:
    if char in _DOUBLE_QUOTES:
        return '"'
    if char in _SINGLE_QUOTES:
        return "'"
    return unicodedata.normalize("NFKC", char)


def okf_yaml_text(title: str) -> str:
    return (
        f"format: {OKF_FORMAT}\n"
        f"version: {OKF_VERSION}\n"
        f"kind: {OKF_BUNDLE_KIND}\n"
        f"title: {yaml_scalar(title)}\n"
        f"entry: {INDEX_MD}\n"
        f"main_source: {ARTICLE_MD}\n"
        f"compiler.name: {COMPILER_NAME}\n"
        f"compiler.mode: {COMPILER_MODE}\n"
        "compiler.global_read: false\n"
        "compiler.global_write: false\n"
    )


def manifest_compiler_block(*, llm_enabled: bool, model: str | None) -> dict:
    try:
        version = package_version("okf-compiler")
    except PackageNotFoundError:
        version = "0+unknown"
    out = {
        "name": COMPILER_NAME,
        "version": version,
        "mode": COMPILER_MODE,
        "global_read": False,
        "global_write": False,
        "llm_enabled": bool(llm_enabled),
    }
    git_commit = os.environ.get("OKF_COMPILER_GIT_COMMIT", "").strip()
    if git_commit:
        out["git_commit"] = git_commit
    if model:
        out["model"] = model
    return out


def slugify(text: str) -> str:
    out: list[str] = []
    for ch in text.strip().lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    return "".join(out).strip("-")[:64]


def yaml_scalar(value: str) -> str:
    value = value.strip()
    if value and not any(c in value for c in ":#{}\n'\"") and not value.startswith(("-", " ", "?")):
        return value
    return json.dumps(value, ensure_ascii=False)
