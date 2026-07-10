"""Prompt builders for local, per-document OKF extraction."""

from __future__ import annotations

import json
import re
import unicodedata

from .schema import ConceptExtract, EntityExtract, SectionSpec


def _section_map(sections: list[SectionSpec]) -> str:
    rows = [
        {
            "section_id": section.section_id,
            "heading_path": section.heading_path,
            "line_start": section.line_start,
            "line_end": section.line_end,
        }
        for section in sections
    ]
    return json.dumps(rows, ensure_ascii=False, indent=2)


def _evidence_contract() -> str:
    return (
        'Evidence MUST be exactly shaped as '
        '{"section_id":"s0001","quote":"verbatim contiguous text copied from Document"}. '
        "Copy the quote exactly. Do not paraphrase, add ellipses, change punctuation, or invent lines."
    )


def summary_messages(markdown: str, sections: list[SectionSpec], language: str) -> tuple[str, str]:
    system = (
        "You summarize one Markdown document in isolation. Do not use external knowledge. "
        'Return JSON only: {"summary": "..."}.'
    )
    user = f"Language: {language}\nSections:\n{_section_map(sections)}\n\nDocument:\n{markdown}"
    return system, user


def concepts_messages(
    markdown: str,
    sections: list[SectionSpec],
    max_concepts: int,
    *,
    summary: str,
) -> tuple[str, str]:
    system = (
        "Extract reusable document-local concepts, mechanisms, strategies, design patterns, and "
        "business ideas. Do NOT return named people, organizations, products, games, platforms, "
        "brands, works, places, or events; those belong in entities. Return JSON only with a "
        "concepts array. Each item needs name, description, confidence, and evidence. "
        + _evidence_contract()
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
        "Extract only concrete named entities. Allowed type values are person, organization, "
        "product, platform, brand, work, location, event, and other_named_entity. Do NOT return "
        "abstract concepts, mechanisms, strategies, gameplay patterns, business models, hooks, or "
        "design labels. Return JSON only with an entities array. Each item needs name, type, "
        "description, aliases, confidence, and evidence. "
        + _evidence_contract()
    )
    concept_names = [concept.name for concept in concepts]
    user = (
        f"Maximum entities: {max_entities}\nSummary: {summary}\n"
        f"Known concepts, normally do not repeat as entities: {concept_names}\n"
        f"Section map:\n{_section_map(sections)}\n\nDocument:\n{markdown}"
    )
    return system, user


def relation_nodes(
    concepts: list[ConceptExtract],
    entities: list[EntityExtract],
) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    concept_index = 1
    entity_index = 1
    for concept in concepts:
        key = _name_key(concept.name)
        if not key or key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "node_id": f"concept:c{concept_index:04d}",
                "name": concept.name,
                "kind": "concept",
                "type": "concept",
            }
        )
        concept_index += 1
    for entity in entities:
        key = _name_key(entity.name)
        if not key or key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "node_id": f"entity:e{entity_index:04d}",
                "name": entity.name,
                "kind": "entity",
                "type": entity.entity_type,
            }
        )
        entity_index += 1
    return rows


def relations_messages(
    markdown: str,
    sections: list[SectionSpec],
    *,
    summary: str,
    concepts: list[ConceptExtract],
    entities: list[EntityExtract],
) -> tuple[str, str]:
    system = (
        "Propose evidence-backed relations between the supplied typed nodes only. Return JSON only "
        "with a relations array. Each relation needs subject_id, relation, object_id, note, and "
        "evidence. subject_id and object_id MUST be copied from Allowed nodes. "
        + _evidence_contract()
    )
    nodes = relation_nodes(concepts, entities)
    user = (
        f"Summary: {summary}\nAllowed nodes:\n"
        f"{json.dumps(nodes, ensure_ascii=False, indent=2)}\n"
        f"Section map:\n{_section_map(sections)}\n\nDocument:\n{markdown}"
    )
    return system, user


def retry_system_message(stage: str, system: str, error: str) -> str:
    schemas = {
        "summary": '{"summary":"string"}',
        "concepts": '{"concepts":[{"name":"...","description":"...","confidence":0.9,'
        '"evidence":{"section_id":"s0001","quote":"..."}}]}',
        "entities": '{"entities":[{"name":"...","type":"product","description":"...",'
        '"aliases":[],"confidence":0.9,"evidence":{"section_id":"s0001","quote":"..."}}]}',
        "relations": '{"relations":[{"subject_id":"concept:c0001","relation":"uses",'
        '"object_id":"entity:e0001","note":"...",'
        '"evidence":{"section_id":"s0001","quote":"..."}}]}',
    }
    return (
        f"{system}\n\nYour previous response was invalid for stage {stage}: {error}. "
        f"Return one JSON object matching this top-level schema exactly: {schemas[stage]}"
    )


def _name_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", "", normalized)
