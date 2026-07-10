"""Command-line interface for the standalone OKF compiler."""

from __future__ import annotations

from pathlib import Path

import click

from .compiler import CompileOptions, compile_dir, compile_one
from .llm import LLMClient, load_dotenv_values, resolve_config


def _llm_options(fn):
    for decorator in (
        click.option("--timeout", type=float, default=None, help="LLM timeout in seconds."),
        click.option("--api-key", default=None, help="LLM API key. Env: OKF_LLM_API_KEY."),
        click.option("--model", default=None, help="LLM model. Env: OKF_LLM_MODEL."),
        click.option("--base-url", default=None, help="OpenAI-compatible base URL."),
    ):
        fn = decorator(fn)
    return fn


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="okf-compiler")
def main() -> None:
    """Compile Markdown into isolated, self-contained OKF Bundles."""


@main.command("compile")
@click.argument("input_md", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--workdir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--keep-workdir", is_flag=True)
@click.option("--overwrite", is_flag=True)
@click.option("--no-llm", is_flag=True)
@click.option("--language", default="zh", show_default=True)
@click.option("--max-concepts", type=int, default=12, show_default=True)
@click.option("--max-entities", type=int, default=12, show_default=True)
@_llm_options
def compile_cmd(
    input_md: Path,
    out: Path | None,
    workdir: Path | None,
    keep_workdir: bool,
    overwrite: bool,
    no_llm: bool,
    language: str,
    max_concepts: int,
    max_entities: int,
    base_url: str | None,
    model: str | None,
    api_key: str | None,
    timeout: float | None,
) -> None:
    """Compile one Markdown file into one .okf.zip."""
    target = out or input_md.with_name(input_md.stem + ".okf.zip")
    config = None
    if not no_llm:
        config = resolve_config(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            dotenv=load_dotenv_values(),
        )
    result = compile_one(
        input_md,
        target,
        CompileOptions(
            workdir=workdir,
            keep_workdir=keep_workdir,
            overwrite=overwrite,
            no_llm=no_llm,
            language=language,
            max_concepts=max_concepts,
            max_entities=max_entities,
            llm_config=config,
        ),
    )
    _print_result(result)
    if not result.ok:
        raise click.exceptions.Exit(1)


@main.command("compile-dir")
@click.argument(
    "input_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--out",
    "out_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option("--mode", type=click.Choice(["auto", "flat", "wechat"]), default="auto")
@click.option("--glob", "glob_pattern", default="*.md", show_default=True)
@click.option("--recursive/--no-recursive", default=True)
@click.option("--overwrite", is_flag=True)
@click.option("--skip-existing/--no-skip-existing", default=True)
@click.option("--no-llm", is_flag=True)
@click.option("--max-workers", type=int, default=1, show_default=True)
@click.option("--fail-fast", is_flag=True)
@click.option(
    "--report",
    "report_path",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option("--language", default="zh", show_default=True)
@click.option("--max-concepts", type=int, default=12, show_default=True)
@click.option("--max-entities", type=int, default=12, show_default=True)
@_llm_options
def compile_dir_cmd(
    input_dir: Path,
    out_dir: Path,
    mode: str,
    glob_pattern: str,
    recursive: bool,
    overwrite: bool,
    skip_existing: bool,
    no_llm: bool,
    max_workers: int,
    fail_fast: bool,
    report_path: Path | None,
    language: str,
    max_concepts: int,
    max_entities: int,
    base_url: str | None,
    model: str | None,
    api_key: str | None,
    timeout: float | None,
) -> None:
    """Compile a directory, producing one OKF Bundle per Markdown article."""
    config = None
    if not no_llm:
        config = resolve_config(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            dotenv=load_dotenv_values(),
        )

    def progress(result, completed: int, total: int) -> None:
        status = "OK" if result.ok else ("SKIP" if result.skipped else "FAIL")
        click.echo(f"[{completed}/{total}] {status} {result.input_path.name}")

    report = compile_dir(
        input_dir,
        out_dir,
        CompileOptions(
            overwrite=overwrite,
            no_llm=no_llm,
            language=language,
            max_concepts=max_concepts,
            max_entities=max_entities,
            llm_config=config,
        ),
        mode=mode,
        glob_pattern=glob_pattern,
        recursive=recursive,
        skip_existing=skip_existing,
        max_workers=max_workers,
        fail_fast=fail_fast,
        report_path=report_path,
        progress_callback=progress,
    )
    click.echo(
        f"Completed: {report.ok} ok, {report.skipped} skipped, "
        f"{report.failed} failed, {report.total} total"
    )
    if report.failed:
        raise click.exceptions.Exit(1)


@main.command("test-llm")
@_llm_options
def test_llm_cmd(
    base_url: str | None,
    model: str | None,
    api_key: str | None,
    timeout: float | None,
) -> None:
    """Test the configured OpenAI-compatible endpoint."""
    config = resolve_config(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=timeout,
        dotenv=load_dotenv_values(),
    )
    try:
        click.echo(LLMClient(config).test())
    except Exception as exc:  # noqa: BLE001
        click.echo(f"LLM test failed: {exc}", err=True)
        raise click.exceptions.Exit(1) from exc


def _print_result(result) -> None:
    if result.ok:
        click.echo(f"OKF bundle written: {result.output_path}")
        counts = (result.manifest or {}).get("counts", {})
        click.echo(
            "Counts: " + ", ".join(f"{key}={value}" for key, value in counts.items())
        )
        for warning in result.warnings:
            click.echo(f"Warning: {warning}", err=True)
    else:
        state = "skipped" if result.skipped else "failed"
        click.echo(f"Compile {state}: {result.error}", err=True)


if __name__ == "__main__":
    main()
