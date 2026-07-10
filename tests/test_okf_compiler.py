import json
import zipfile
from pathlib import Path

from click.testing import CliRunner

from okf_compiler.assets import collect_images
from okf_compiler.cli import main
from okf_compiler.compiler import CompileOptions, compile_dir, compile_one, enumerate_inputs
from okf_compiler.llm import load_dotenv_values, resolve_config
from okf_compiler.markdown import extract_title, split_sections


def make_article(root: Path, name: str = "article") -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "assets").mkdir()
    (folder / "assets" / "pic.png").write_bytes(b"img")
    md = folder / f"{name}.md"
    md.write_text(
        "---\n"
        'title: "Test"\n'
        'source_url: "https://example.com/a"\n'
        'author: "Author"\n'
        "---\n\n"
        "# Test\n\n## One\nText ![p](assets/pic.png)\n\n## Two\nMore\n",
        encoding="utf-8",
    )
    return md


def test_extract_title_and_split_h2():
    md = "# Title\n\n## One\nA\n### Child\nB\n## Two\nC\n"
    result = split_sections(md)
    assert extract_title(md) == "Title"
    assert result.effective_level == 2
    assert [section.title for section in result.sections] == ["One", "Two"]
    assert result.sections[0].line_start == 3
    assert result.sections[0].anchors[0].title == "Child"


def test_single_heading_falls_back_to_whole_document():
    result = split_sections("# T\n\n## Only\nBody\n")
    assert result.effective_level is None
    assert len(result.sections) == 1
    assert result.sections[0].boundary_kind == "whole_doc"


def test_code_fence_headings_are_ignored():
    md = "# T\n\n```md\n## Fake\n```\n\n## Real A\nA\n## Real B\nB\n"
    result = split_sections(md)
    titles = [section.title for section in result.sections]
    assert "Fake" not in titles
    assert titles[-2:] == ["Real A", "Real B"]


def test_collect_images_rewrites_source_and_sections(tmp_path: Path):
    source = tmp_path / "input"
    source.mkdir()
    (source / "assets").mkdir()
    (source / "assets" / "a.png").write_bytes(b"png")
    md = "## S\n![A](assets/a.png)\n"
    images = tmp_path / "work" / "assets" / "images"
    sections, normalized, refs, warnings = collect_images([md], md, source, images)
    assert not warnings
    assert refs[0].found
    assert "../assets/images/a.png" in sections[0]
    assert "../assets/images/a.png" in normalized
    assert (images / "a.png").read_bytes() == b"png"


def test_collect_images_rejects_escape(tmp_path: Path):
    sections, normalized, refs, warnings = collect_images(
        ["![x](../secret.png)"],
        "![x](../secret.png)",
        tmp_path,
        tmp_path / "images",
    )
    assert warnings
    assert not refs
    assert "../secret.png" in sections[0]
    assert "../secret.png" in normalized


def test_compile_one_builds_portable_bundle(tmp_path: Path):
    md = make_article(tmp_path)
    out = tmp_path / "article.okf.zip"
    result = compile_one(md, out, CompileOptions(no_llm=True))
    assert result.ok, result.error
    assert out.is_file()
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        assert "sources/original.md" in names
        assert "sources/article.md" in names
        assert "assets/images/pic.png" in names
        normalized = zf.read("sources/article.md").decode()
        assert "../assets/images/pic.png" in normalized
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["source"]["url"] == "https://example.com/a"
        assert manifest["counts"]["sections"] == 2
        assert manifest["compiler"]["global_read"] is False


def test_wechat_enumeration_prefers_manifest(tmp_path: Path):
    folder = tmp_path / "abc"
    folder.mkdir()
    (folder / "document.md").write_text("# T", encoding="utf-8")
    (folder / "debug.md").write_text("# D", encoding="utf-8")
    (folder / "manifest.json").write_text('{"main":"document.md"}', encoding="utf-8")
    selected, skipped = enumerate_inputs(tmp_path, mode="wechat")
    assert selected == [folder / "document.md"]
    assert not skipped


def test_compile_dir_isolates_articles(tmp_path: Path):
    input_dir = tmp_path / "input"
    make_article(input_dir, "a")
    make_article(input_dir, "b")
    out = tmp_path / "bundles"
    report = compile_dir(input_dir, out, CompileOptions(no_llm=True), mode="wechat")
    assert report.ok == 2
    assert report.failed == 0
    assert len(list(out.glob("*.okf.zip"))) == 2
    assert (out / "batch_report.json").is_file()


def test_cli_compile_no_llm(tmp_path: Path):
    md = tmp_path / "a.md"
    md.write_text("# A\n\n## One\nX\n## Two\nY\n", encoding="utf-8")
    out = tmp_path / "a.okf.zip"
    result = CliRunner().invoke(
        main,
        ["compile", str(md), "--out", str(out), "--no-llm"],
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()


def test_dotenv_discovery_and_config_precedence(tmp_path: Path, monkeypatch):
    (tmp_path / ".env").write_text(
        "OKF_LLM_BASE_URL=https://dotenv.example/v1\n"
        "OKF_LLM_MODEL=dotenv-model\n"
        "OKF_LLM_API_KEY=dotenv-key\n"
        "OKF_LLM_TIMEOUT=45\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    dotenv = load_dotenv_values(env={})
    config = resolve_config(
        base_url=None,
        model=None,
        api_key="cli-key",
        timeout=None,
        env={"OKF_LLM_MODEL": "env-model"},
        dotenv=dotenv,
    )
    assert config.base_url == "https://dotenv.example/v1"
    assert config.model == "env-model"
    assert config.api_key == "cli-key"
    assert config.timeout == 45


def test_dotenv_discovery_falls_back_to_input_dir(tmp_path: Path, monkeypatch):
    cwd = tmp_path / "cwd"
    input_dir = tmp_path / "articles"
    cwd.mkdir()
    input_dir.mkdir()
    (input_dir / ".env").write_text("OKF_LLM_MODEL=input-model\n", encoding="utf-8")
    monkeypatch.chdir(cwd)
    values = load_dotenv_values(search_dirs=[input_dir], env={})
    assert values["OKF_LLM_MODEL"] == "input-model"


def test_cli_test_llm_accepts_env_file(tmp_path: Path, monkeypatch):
    env_file = tmp_path / "custom.env"
    env_file.write_text(
        "OKF_LLM_BASE_URL=https://custom.example/v1\n"
        "OKF_LLM_MODEL=custom-model\n"
        "OKF_LLM_API_KEY=custom-key\n",
        encoding="utf-8",
    )
    captured = {}

    class FakeClient:
        def __init__(self, config):
            captured["config"] = config

        def test(self):
            return '{"ok": true}'

    monkeypatch.setattr("okf_compiler.cli.LLMClient", FakeClient)
    result = CliRunner().invoke(main, ["test-llm", "--env-file", str(env_file)])
    assert result.exit_code == 0, result.output
    assert captured["config"].base_url == "https://custom.example/v1"
    assert captured["config"].model == "custom-model"
    assert captured["config"].api_key == "custom-key"
