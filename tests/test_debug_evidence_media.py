import json
import zipfile
from pathlib import Path

from okf_compiler import CompileOptions, compile_one
from okf_compiler.assets import collect_assets
from okf_compiler.llm import extract
from okf_compiler.markdown import split_sections
from okf_compiler.schema import Evidence, resolve_evidence


class FakeClient:
    api_key = None

    def __init__(self, responses):
        self.responses = iter(responses)

    def json_completion(self, system, user):
        return json.dumps(next(self.responses), ensure_ascii=False)


def test_quote_evidence_resolves_absolute_lines():
    markdown = "# Title\n\n## First\nalpha line\nsecond line\n\n## Second\nbeta line\n"
    sections = split_sections(markdown).sections
    result = resolve_evidence(
        Evidence(section_id="s0001", quote="alpha line second line"),
        sections,
        markdown,
    )
    assert result.valid
    assert result.reason == "ok_normalized_quote"
    assert (result.evidence.line_start, result.evidence.line_end) == (4, 5)


def test_collects_markdown_images_and_html_video(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    assets = source / "assets"
    assets.mkdir()
    (assets / "photo.webp").write_bytes(b"image")
    (assets / "clip.mp4").write_bytes(b"video")
    markdown = (
        "![photo](assets/photo.webp)\n"
        '<video controls poster="assets/photo.webp" src="assets/clip.mp4"></video>\n'
    )
    workdir = tmp_path / "work"
    sections, normalized, refs, warnings = collect_assets([markdown], markdown, source, workdir)
    assert not warnings
    assert "../assets/images/photo.webp" in sections[0]
    assert "../assets/media/clip.mp4" in sections[0]
    assert "../assets/media/clip.mp4" in normalized
    assert (workdir / "assets/media/clip.mp4").read_bytes() == b"video"
    assert {ref.kind for ref in refs} == {"image", "video"}


def test_quote_evidence_is_resolved_for_all_extraction_stages():
    markdown = "# T\n\n## Body\n赵云与阿斗是一款小游戏。\n短局设计增加广告展示。\n\n## End\n结束。\n"
    sections = split_sections(markdown).sections
    client = FakeClient(
        [
            {"summary": "摘要"},
            {
                "concepts": [
                    {
                        "name": "短局设计",
                        "description": "短局",
                        "confidence": 0.9,
                        "evidence": {"section_id": "s0001", "quote": "短局设计增加广告展示。"},
                    }
                ]
            },
            {
                "entities": [
                    {
                        "name": "赵云与阿斗",
                        "type": "game",
                        "description": "游戏",
                        "aliases": [],
                        "confidence": 0.9,
                        "evidence": {"section_id": "s0001", "quote": "赵云与阿斗是一款小游戏。"},
                    }
                ]
            },
            {
                "relations": [
                    {
                        "subject": "赵云与阿斗",
                        "relation": "uses",
                        "object": "短局设计",
                        "note": "",
                        "evidence": {"section_id": "s0001", "quote": "短局设计增加广告展示。"},
                    }
                ]
            },
        ]
    )
    result = extract(client, markdown, sections, language="zh", max_concepts=12, max_entities=12)
    assert len(result.concepts) == 1
    assert len(result.entities) == 1
    assert len(result.relations) == 1
    assert result.entities[0].evidence.line_start == 4
    assert not result.degraded


def test_invalid_quote_is_reported_as_degraded():
    markdown = "# T\n\n## Body\n真实内容。\n\n## End\n结束。\n"
    sections = split_sections(markdown).sections
    client = FakeClient(
        [
            {"summary": "摘要"},
            {"concepts": []},
            {
                "entities": [
                    {
                        "name": "不存在",
                        "type": "thing",
                        "description": "",
                        "aliases": [],
                        "evidence": {"section_id": "s0001", "quote": "模型编造的证据"},
                    }
                ]
            },
            {"relations": []},
        ]
    )
    result = extract(client, markdown, sections, language="zh", max_concepts=12, max_entities=12)
    assert result.degraded
    assert result.stage_stats["entities"]["rejected"] == 1
    assert result.validation[0]["reason"] == "quote_not_found"


def test_compile_bundles_video_and_writes_debug_sidecar(tmp_path: Path):
    source = tmp_path / "article"
    source.mkdir()
    assets = source / "assets"
    assets.mkdir()
    (assets / "clip.mp4").write_bytes(b"video")
    markdown = source / "article.md"
    markdown.write_text(
        '# Demo\n\n## One\n<video controls src="assets/clip.mp4"></video>\n\n## Two\nDone.\n',
        encoding="utf-8",
    )
    output = source / "article.okf.zip"
    result = compile_one(
        markdown,
        output,
        CompileOptions(no_llm=True, debug_dir=tmp_path / "debug"),
    )
    assert result.successful
    assert (result.debug_dir / "run.json").is_file()
    with zipfile.ZipFile(output) as bundle:
        assert "assets/media/clip.mp4" in bundle.namelist()
        assert '../assets/media/clip.mp4' in bundle.read("sources/article.md").decode("utf-8")
        manifest = json.loads(bundle.read("manifest.json"))
        assert manifest["counts"]["media"] == 1
        assert manifest["compiler"]["version"]
