import json
from pathlib import Path

from okf_compiler.diagnostics import DebugRecorder
from okf_compiler.llm import extract
from okf_compiler.markdown import split_sections
from okf_compiler.prompts import relation_nodes
from okf_compiler.schema import (
    ConceptExtract,
    EntityExtract,
    Evidence,
    SectionSpec,
    resolve_evidence,
)


class FakeClient:
    api_key = None

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def json_completion(self, system, user):
        self.calls.append((system, user))
        value = next(self.responses)
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _article_sections():
    markdown = (
        "# T\n\n## Body\n"
        "《赵云与阿斗》的做法是把广告放在“最合理的救局时刻”。\n"
        "短局设计增加广告展示。\n\n## End\n结束。\n"
    )
    return markdown, split_sections(markdown).sections


def _concept_item(name="短局设计"):
    return {
        "name": name,
        "description": "概念",
        "confidence": 0.9,
        "evidence": {"section_id": "s0001", "quote": "短局设计增加广告展示。"},
    }


def _entity_item(name="赵云与阿斗", entity_type="游戏产品"):
    return {
        "name": name,
        "type": entity_type,
        "description": "游戏",
        "aliases": [],
        "confidence": 0.9,
        "evidence": {
            "section_id": "s0001",
            "quote": "《赵云与阿斗》的做法是把广告放在“最合理的救局时刻”。",
        },
    }


def test_typography_normalization_preserves_original_quote():
    markdown, sections = _article_sections()
    result = resolve_evidence(
        Evidence(
            section_id="s0001",
            quote='《赵云与阿斗》的做法是把广告放在"最合理的救局时刻"。',
        ),
        sections,
        markdown,
    )
    assert result.valid
    assert result.reason == "ok_normalized_quote"
    assert "“最合理的救局时刻”" in result.evidence.quote
    assert '"最合理的救局时刻"' not in result.evidence.quote


def test_frontmatter_duplicate_prefers_visible_body_match():
    title = "重复标题"
    markdown = f'---\ntitle: "{title}"\n---\n\n# {title}\n正文\n'
    section = SectionSpec(
        index=0,
        title="preamble",
        heading_path="preamble",
        line_start=1,
        line_end=6,
        body=markdown,
        section_id="s0001",
        boundary_kind="preamble",
    )
    result = resolve_evidence(Evidence(section_id="s0001", quote=title), [section], markdown)
    assert result.valid
    assert result.reason == "ok_exact_quote_body_preferred"
    assert result.evidence.line_start == 5


def test_malformed_relation_response_retries_once_and_recovers(tmp_path: Path):
    markdown, sections = _article_sections()
    client = FakeClient(
        [
            {"summary": "摘要"},
            {"concepts": [_concept_item()]},
            {"entities": [_entity_item()]},
            {},
            {
                "relations": [
                    {
                        "subject_id": "entity:e0001",
                        "relation": "uses",
                        "object_id": "concept:c0001",
                        "note": "",
                        "evidence": {
                            "section_id": "s0001",
                            "quote": "短局设计增加广告展示。",
                        },
                    }
                ]
            },
        ]
    )
    debug = DebugRecorder(tmp_path, include_llm_payloads=True)
    result = extract(
        client,
        markdown,
        sections,
        language="zh",
        max_concepts=12,
        max_entities=12,
        debug=debug,
    )
    assert len(result.relations) == 1
    assert result.stage_stats["relations"]["attempts"] == 2
    assert result.stage_stats["relations"]["retries"] == 1
    assert (tmp_path / "stages/relations/attempt-2/request.json").is_file()
    assert "previous response was invalid" in client.calls[-1][0]


def test_two_malformed_relation_responses_fail_stage():
    markdown, sections = _article_sections()
    client = FakeClient(
        [
            {"summary": "摘要"},
            {"concepts": [_concept_item()]},
            {"entities": [_entity_item()]},
            {},
            {"relation": []},
        ]
    )
    result = extract(
        client,
        markdown,
        sections,
        language="zh",
        max_concepts=12,
        max_entities=12,
    )
    assert result.stage_stats["relations"]["status"] == "failed"
    assert result.stage_stats["relations"]["attempts"] == 2
    assert result.degraded


def test_entity_classification_reconciles_concept_overlap():
    markdown, sections = _article_sections()
    concept_named_entity = {
        "name": "赵云与阿斗",
        "description": "误分概念",
        "confidence": 0.8,
        "evidence": {
            "section_id": "s0001",
            "quote": "《赵云与阿斗》的做法是把广告放在“最合理的救局时刻”。",
        },
    }
    conceptual_entity = {
        "name": "短局设计",
        "type": "玩法机制",
        "description": "不是实体",
        "aliases": [],
        "confidence": 0.9,
        "evidence": {"section_id": "s0001", "quote": "短局设计增加广告展示。"},
    }
    client = FakeClient(
        [
            {"summary": "摘要"},
            {"concepts": [concept_named_entity, _concept_item()]},
            {"entities": [_entity_item(), conceptual_entity]},
            {"relations": []},
        ]
    )
    result = extract(
        client,
        markdown,
        sections,
        language="zh",
        max_concepts=12,
        max_entities=12,
    )
    assert [concept.name for concept in result.concepts] == ["短局设计"]
    assert [entity.name for entity in result.entities] == ["赵云与阿斗"]
    assert result.stage_stats["concepts"]["reclassified"] == 1
    assert result.stage_stats["entities"]["rejected"] == 1
    assert any(item["reason"] == "invalid_or_conceptual_entity_type" for item in result.validation)


def test_relation_nodes_are_typed_unique_and_deterministic():
    nodes = relation_nodes(
        [
            ConceptExtract("短局设计", ""),
            ConceptExtract("短局设计", "重复"),
        ],
        [
            EntityExtract("赵云与阿斗", "product", ""),
            EntityExtract("短局设计", "product", "冲突"),
        ],
    )
    assert nodes == [
        {
            "node_id": "concept:c0001",
            "name": "短局设计",
            "kind": "concept",
            "type": "concept",
        },
        {
            "node_id": "entity:e0001",
            "name": "赵云与阿斗",
            "kind": "entity",
            "type": "product",
        },
    ]
