# OKF Compiler

Compile one Markdown article into one isolated, self-contained **OKF Bundle** (`.okf.zip`).

OKF Compiler follows a subtraction-first rule: every document starts fresh. A compile reads only the current Markdown file and its relative assets. It never reads or mutates a long-lived knowledge base, performs no global concept merge, and materializes no global backlinks.

The project was extracted from the experimental `openkb/okf` subsystem in `FFFFFFpy/OpenKB` and made independent of the OpenKB runtime.

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
│   └── images/
├── extracts/
│   ├── summary.md
│   ├── concepts/
│   ├── entities/
│   └── claims/
└── relations/
    └── proposed_edges.jsonl
```

- `sources/original.md` preserves the input verbatim.
- `sources/article.md` is the portable source with image links rewritten to bundled assets.
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

Copy the included template and fill in your provider values:

```powershell
Copy-Item .env.example .env
```

```env
OKF_LLM_BASE_URL=https://api.example.com/v1
OKF_LLM_MODEL=your-model
OKF_LLM_API_KEY=your-key
OKF_LLM_TIMEOUT=120
```

Then test the connection:

```powershell
uv run okf test-llm
```

Configuration priority is:

```text
CLI flags > process environment > .env > defaults
```

Dotenv discovery order is:

1. `--env-file PATH`.
2. The path in `OKF_ENV_FILE`.
3. `.env` in the current working directory.
4. `.env` beside the input Markdown or input directory.

Only one dotenv file is loaded. An explicitly selected file must exist; a typo does not silently fall back to some unrelated `.env` lurking elsewhere like a credential poltergeist.

Use a custom file when needed:

```powershell
uv run okf test-llm --env-file "D:\secrets\okf.env"
```

`.env` is ignored by Git. `.env.example` is safe to commit and contains no real credentials.

For migration, the old `OPENKB_LLM_*` names remain accepted as fallback values.

## Compile one Markdown file

Without LLM:

```powershell
uv run okf compile article.md --no-llm
```

With `.env` configured, no repeated LLM flags are needed:

```powershell
uv run okf compile article.md
```

CLI flags remain available for one-off overrides:

```powershell
uv run okf compile article.md `
  --base-url "https://api.example.com/v1" `
  --model "your-model" `
  --api-key "your-key"
```

By default the output is `article.okf.zip` beside the source file.

## Compile a wxarticle2md output directory

`wxarticle2md --md-mode advanced-lite` produces one article directory containing one Markdown file and `assets/`. OKF Compiler can enumerate that layout directly:

```powershell
uv run okf compile-dir `
  "D:\MyProjects\wxarticle2md\output" `
  --out "D:\MyProjects\okf-bundles" `
  --mode wechat `
  --max-workers 3
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

This deliberately favors stable, coarse sections over turning every bold sentence into a tiny knowledge fragment. Software has enough confetti already.

## Python API

```python
from pathlib import Path
from okf_compiler import CompileOptions, compile_one

result = compile_one(
    Path("article.md"),
    Path("article.okf.zip"),
    CompileOptions(no_llm=True),
)

if not result.ok:
    raise RuntimeError(result.error)
```

## Design boundaries

OKF Compiler is a producer, not a knowledge-base runtime. Consumers such as OpenViking, WeKnora, or OpenKB may:

- index sections, summaries, concepts, entities, and images separately;
- resolve local concepts against global candidates;
- accept or reject proposed edges;
- materialize wiki pages or backlinks later.

The compiler itself does none of those global operations.

## License and provenance

Apache License 2.0. See `NOTICE` for extraction provenance and changes.
