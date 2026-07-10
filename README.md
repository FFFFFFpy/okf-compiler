# OKF Compiler

Compile one Markdown article into one isolated, self-contained **OKF Bundle** (`.okf.zip`).

OKF Compiler follows a subtraction-first rule: every document starts fresh. A compile reads only the current Markdown file and its relative assets. It never reads or mutates a long-lived knowledge base, performs no global concept merge, and materializes no global backlinks.

## What it produces

```text
article.okf.zip
├── okf.yaml
├── manifest.json
├── index.md
├── log.md
├── source_map.json
├── sources/
│   ├── original.md
│   └── article.md
├── sections/
├── assets/
│   ├── images/
│   └── media/
├── extracts/
│   ├── summary.md
│   ├── concepts/
│   ├── entities/
│   └── claims/
└── relations/
    └── proposed_edges.jsonl
```

- `sources/original.md` preserves the input verbatim.
- `sources/article.md` is the portable source with local asset links rewritten into the bundle.
- Referenced Markdown images and local HTML `img`, `video`, `audio`, and `source` assets are copied.
- `sections/*.md` are split conservatively from real ATX headings.
- LLM output is document-local and evidence-anchored. It is proposed knowledge, not a global wiki mutation.

## Install

Basic compiler, without LLM extraction:

```powershell
uv tool install git+https://github.com/FFFFFFpy/okf-compiler.git
```

With optional LiteLLM extraction:

```powershell
uv tool install "okf-compiler[llm] @ git+https://github.com/FFFFFFpy/okf-compiler.git"
```

For development:

```powershell
git clone https://github.com/FFFFFFpy/okf-compiler.git
cd okf-compiler
uv sync --extra dev --extra llm
```

## Configure the LLM with `.env`

```powershell
Copy-Item .env.example .env
```

```env
OKF_LLM_BASE_URL=https://api.example.com/v1
OKF_LLM_MODEL=your-model
OKF_LLM_API_KEY=your-key
OKF_LLM_TIMEOUT=120
```

Configuration priority is:

```text
CLI flags > process environment > .env > defaults
```

Dotenv discovery order is `--env-file`, `OKF_ENV_FILE`, `.env` in the current working directory, then `.env` beside the input.

## Compile one Markdown file

```powershell
uv run okf compile article.md
```

Without LLM:

```powershell
uv run okf compile article.md --no-llm
```

By default the output is `article.okf.zip` beside the source file.

## Debug diagnostics

Normal bundle logs stay compact. For structured sidecar diagnostics:

```powershell
uv run okf compile article.md `
  --debug-dir ".\okf-debug"
```

The run directory contains:

```text
okf-debug/<run>/
├── run.json
├── compiler.jsonl
├── traceback.log              # only when an exception occurs
└── validation/
    ├── concepts.jsonl
    ├── entities.jsonl
    └── relations.jsonl
```

To additionally save full LLM prompts, raw responses, and parsed JSON:

```powershell
uv run okf compile article.md `
  --debug-dir ".\okf-debug" `
  --debug-llm-payloads
```

Full payload logs may contain the complete source article. API keys are not intentionally written, but debug directories should still be treated as private data.

Use strict mode when degraded extraction must fail CI or a batch job:

```powershell
uv run okf compile article.md --strict
```

The bundle is still written for inspection, but the command returns a non-zero exit code when an extraction stage rejected output.

## Evidence model

New LLM extraction uses quote-based evidence:

```json
{
  "section_id": "s0002",
  "quote": "verbatim contiguous text copied from the document"
}
```

The compiler locates the quote inside the selected section and calculates absolute `line_start` and `line_end`. This avoids asking an LLM to count Markdown lines. Legacy line-based evidence remains accepted for compatibility.

## Compile a wxarticle2md output directory

```powershell
uv run okf compile-dir `
  "D:\MyProjects\wxarticle2md\output" `
  --out "D:\MyProjects\okf-bundles" `
  --mode wechat `
  --max-workers 3 `
  --debug-dir "D:\MyProjects\okf-debug"
```

Input selection priority in `wechat` mode:

1. Main Markdown declared by `manifest.json`.
2. Markdown whose stem matches the article directory.
3. `document.md`.
4. The only Markdown file in the directory.
5. Otherwise the directory is reported as ambiguous and skipped.

## Sectioning rules

- H1 is treated as the document title.
- The first ATX level from H2 to H6 with at least two headings becomes the section boundary.
- Lower headings and conservative pseudo-heading matches become section anchors, not extra top-level sections.
- If no suitable heading level exists, the document becomes one whole-document section.

## Python API

```python
from pathlib import Path
from okf_compiler import CompileOptions, compile_one

result = compile_one(
    Path("article.md"),
    Path("article.okf.zip"),
    CompileOptions(no_llm=True),
)
if not result.successful:
    raise RuntimeError(result.error)
```

## Design boundaries

OKF Compiler is a producer, not a knowledge-base runtime. Consumers such as OpenViking, WeKnora, or OpenKB may index sections and extracts, resolve local concepts against global candidates, and accept or reject proposed edges. The compiler itself performs no global mutation.

## License and provenance

Apache License 2.0. See `NOTICE` for extraction provenance and changes.
