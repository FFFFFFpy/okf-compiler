"""OKF Bundle paths, identities, and data structures."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

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
SOURCE_MAP_JSON = "source_map.json"
SECTION_INDEX_WIDTH = 2
SECTION_TYPE = "Section"


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
class ImageRef:
    original_ref: str
    dest_name: str
    source_path: str
    found: bool
    alt: str = ""


@dataclass
class Evidence:
    heading_path: str
    line_start: int
    line_end: int
    section_id: str = ""

    def is_valid(self) -> bool:
        return bool(self.heading_path) and self.line_start >= 1 and self.line_end >= self.line_start


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


def validate_evidence(evidence: Evidence | None, sections: list[SectionSpec], total_lines: int) -> bool:
    if evidence is None or not evidence.is_valid():
        return False
    if evidence.line_end > total_lines:
        return False
    for sec in sections:
        if evidence.section_id and sec.section_id != evidence.section_id:
            continue
        if not evidence.section_id and sec.heading_path != evidence.heading_path:
            continue
        return sec.line_start <= evidence.line_start <= evidence.line_end <= sec.line_end
    return not sections and evidence.line_end <= total_lines


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
    out = {
        "name": COMPILER_NAME,
        "mode": COMPILER_MODE,
        "global_read": False,
        "global_write": False,
        "llm_enabled": bool(llm_enabled),
    }
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
