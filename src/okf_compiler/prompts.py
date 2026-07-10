"""Prompt builders for local, per-document OKF extraction."""

from __future__ import annotations

import json

from .schema import ConceptExtract, EntityExtract, SectionSpec


def _section_map(sections: list[SectionSpec]) -> str:
    rows = [
        {
            "section_id": s.section_id,
            "heading_path": s.heading_path,
            "line_start": s.line_start,
            "line_end": s.line_end,
        }
        for s in sections
    ]
    return json.dumps(rows, ensure_ascii=False, indent=2)


def summary_messages(markdown: str, sections: list[SectionSpec], language: str) -> tuple[str, str]:
    system = (
        "You summarize one Markdown document in isolation. Do not use external knowledge. "
        "Return JSON only: {\"summary\": \"...\"}."
    )
    user = f"Language: {language}\nSections:\n{_section_map(sections)}\n\nDocument:\n{markdown}"
    return system, user


def concepts_messages(
    markdown: str, sections: list[SectionSpec], max_concepts: int, *, summary: str
) -> tuple[str, str]:
    system = (
        "Extract document-local concepts. Return JSON only with a concepts array. Each item needs "
        "name, description, confidence, and evidence containing section_id, heading_path, "
        "line_start, line_end. Evidence must match the supplied section map."
    )
    user = (
        f"Maximum concepts: {max_concepts}\nSummary: {summary}\n"
        f"Section map:\n{_section_map(sections)}\n\nDocument:\n{markdown}"
    )
    return system, user


def entities_messages(
    markdown: str,
    sections: list[SectionSpec],
    max_entities: int,
    *,
    summary: str,
    concepts: list[ConceptExtract],
) -> tuple[str, str]:
    system = (
        "Extract document-local named entities. Return JSON only with an entities array. Each item "
        "needs name, type, description, aliases, confidence, and valid section evidence."
    )
    concept_names = [c.name for c in concepts]
    user = (
        f"Maximum entities: {max_entities}\nSummary: {summary}\nConcepts: {concept_names}\n"
        f"Section map:\n{_section_map(sections)}\n\nDocument:\n{markdown}"
    )
    return system, user


def relations_messages(
    markdown: str,
    sections: list[SectionSpec],
    *,
    summary: str,
    concepts: list[ConceptExtract],
    entities: list[EntityExtract],
) -> tuple[str, str]:
    system = (
        "Propose relations between the supplied local concepts/entities only. Return JSON only with "
        "a relations array. Each relation needs subject, relation, object, note, and valid evidence."
    )
    nodes = [c.name for c in concepts] + [e.name for e in entities]
    user = (
        f"Summary: {summary}\nAllowed nodes: {nodes}\n"
        f"Section map:\n{_section_map(sections)}\n\nDocument:\n{markdown}"
    )
    return system, user
