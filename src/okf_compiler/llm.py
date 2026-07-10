"""Optional OpenAI-compatible LLM extraction for OKF bundles."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

from .prompts import concepts_messages, entities_messages, relations_messages, summary_messages
from .schema import (
    ConceptExtract,
    EntityExtract,
    Evidence,
    Extracts,
    ProposedEdge,
    SectionSpec,
    validate_evidence,
)

logger = logging.getLogger(__name__)

ENV_BASE_URL = "OKF_LLM_BASE_URL"
ENV_MODEL = "OKF_LLM_MODEL"
ENV_API_KEY = "OKF_LLM_API_KEY"
ENV_TIMEOUT = "OKF_LLM_TIMEOUT"
ENV_FILE = "OKF_ENV_FILE"
_LEGACY = {
    ENV_BASE_URL: "OPENKB_LLM_BASE_URL",
    ENV_MODEL: "OPENKB_LLM_MODEL",
    ENV_API_KEY: "OPENKB_LLM_API_KEY",
    ENV_TIMEOUT: "OPENKB_LLM_TIMEOUT",
}
_DEFAULT_TIMEOUT = 120.0
_REDACTED = "[REDACTED]"
_SECRET_RE = re.compile(r"(?:sk-|Bearer\s+)[A-Za-z0-9_.-]{8,}", re.IGNORECASE)


@dataclass
class LLMConfig:
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None
    timeout: float = _DEFAULT_TIMEOUT

    def is_configured(self) -> bool:
        return bool(self.model and self.model.strip())


def find_dotenv_path(
    path: Path | None = None,
    *,
    search_dirs: list[Path] | tuple[Path, ...] | None = None,
    env: dict[str, str] | None = None,
) -> Path | None:
    """Resolve the one dotenv file used for a command.

    Resolution order:
      1. explicit ``path`` / ``--env-file``;
      2. ``OKF_ENV_FILE`` from the process environment;
      3. ``.env`` in the current working directory;
      4. ``.env`` in each caller-provided search directory.

    Only one file is loaded. Explicit/configured files must exist so a typo
    cannot silently fall back to unrelated credentials.
    """
    current_env = dict(os.environ) if env is None else env
    explicit = path
    if explicit is None:
        configured = current_env.get(ENV_FILE)
        explicit = Path(configured).expanduser() if configured else None

    if explicit is not None:
        candidate = Path(explicit).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"dotenv file not found: {candidate}")
        return candidate

    candidates = [Path.cwd()]
    candidates.extend(Path(item) for item in (search_dirs or []))
    seen: set[Path] = set()
    for directory in candidates:
        candidate = (directory.expanduser() / ".env").resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def load_dotenv_values(
    path: Path | None = None,
    *,
    search_dirs: list[Path] | tuple[Path, ...] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    resolved = find_dotenv_path(path, search_dirs=search_dirs, env=env)
    if resolved is None:
        return {}
    values = dotenv_values(resolved)
    return {str(k): str(v) for k, v in values.items() if k and v is not None}


def resolve_config(
    *,
    base_url: str | None,
    model: str | None,
    api_key: str | None,
    timeout: float | None,
    env: dict[str, str] | None = None,
    dotenv: dict[str, str] | None = None,
) -> LLMConfig:
    env = dict(os.environ) if env is None else env
    dotenv = load_dotenv_values() if dotenv is None else dotenv

    def pick(value: str | None, name: str) -> str | None:
        if value and value.strip():
            return value.strip()
        for key in (name, _LEGACY[name]):
            candidate = env.get(key) or dotenv.get(key)
            if candidate and candidate.strip():
                return candidate.strip()
        return None

    timeout_value = timeout
    if timeout_value is None:
        raw = pick(None, ENV_TIMEOUT)
        try:
            timeout_value = float(raw) if raw else _DEFAULT_TIMEOUT
        except ValueError:
            timeout_value = _DEFAULT_TIMEOUT
    return LLMConfig(
        base_url=pick(base_url, ENV_BASE_URL),
        model=pick(model, ENV_MODEL),
        api_key=pick(api_key, ENV_API_KEY),
        timeout=float(timeout_value),
    )


def normalize_model(model: str | None, base_url: str | None) -> str | None:
    if not model:
        return model
    model = model.strip()
    return f"openai/{model}" if base_url and "/" not in model else model


def redact_secrets(text: str, api_key: str | None = None) -> str:
    out = str(text)
    if api_key:
        out = out.replace(api_key, _REDACTED)
    return _SECRET_RE.sub(_REDACTED, out)


class LLMClient:
    def __init__(self, config: LLMConfig):
        if not config.is_configured():
            raise ValueError("LLM model is required")
        self.config = config
        self.model = normalize_model(config.model, config.base_url)

    @property
    def api_key(self) -> str | None:
        return self.config.api_key

    def json_completion(self, system: str, user: str) -> str:
        try:
            import litellm
        except ImportError as exc:
            raise RuntimeError("install okf-compiler[llm] to enable LLM extraction") from exc
        kwargs = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "timeout": self.config.timeout,
        }
        if self.config.base_url:
            kwargs["api_base"] = self.config.base_url
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        response = litellm.completion(**kwargs)
        return (response.choices[0].message.content or "").strip()

    def test(self) -> str:
        return self.json_completion("Return valid JSON only.", 'Return {"ok": true}.')


def _parse_json(text: str) -> dict | list:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json
        except ImportError as exc:
            raise ValueError("invalid JSON and json-repair is not installed") from exc
        value = json.loads(repair_json(cleaned))
    if not isinstance(value, (dict, list)):
        raise ValueError("expected JSON object or array")
    return value


def _evidence(obj: dict) -> Evidence | None:
    raw = obj.get("evidence")
    if not isinstance(raw, dict):
        return None
    try:
        return Evidence(
            heading_path=str(raw.get("heading_path") or ""),
            line_start=int(raw.get("line_start")),
            line_end=int(raw.get("line_end")),
            section_id=str(raw.get("section_id") or ""),
        )
    except (TypeError, ValueError):
        return None


def _items(value: dict | list, key: str) -> list[dict]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    raw = value.get(key, [])
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def extract(
    client: LLMClient,
    markdown: str,
    sections: list[SectionSpec],
    *,
    language: str,
    max_concepts: int,
    max_entities: int,
) -> Extracts:
    out = Extracts()
    total_lines = max(len(markdown.splitlines()), 1)

    def warn(label: str, exc: Exception | str) -> None:
        out.warnings.append(redact_secrets(f"{label}: {exc}", client.api_key))

    try:
        system, user = summary_messages(markdown, sections, language)
        value = _parse_json(client.json_completion(system, user))
        out.summary = str(value.get("summary", "")) if isinstance(value, dict) else ""
    except Exception as exc:  # noqa: BLE001
        warn("summary extraction failed", exc)

    try:
        system, user = concepts_messages(markdown, sections, max_concepts, summary=out.summary)
        for item in _items(_parse_json(client.json_completion(system, user)), "concepts")[:max_concepts]:
            evidence = _evidence(item)
            if not validate_evidence(evidence, sections, total_lines):
                warn("concept dropped", f"invalid evidence for {item.get('name')!r}")
                continue
            out.concepts.append(
                ConceptExtract(
                    name=str(item.get("name") or "").strip(),
                    description=str(item.get("description") or "").strip(),
                    confidence=_float_or_none(item.get("confidence")),
                    evidence=evidence,
                )
            )
    except Exception as exc:  # noqa: BLE001
        warn("concept extraction failed", exc)

    try:
        system, user = entities_messages(
            markdown, sections, max_entities, summary=out.summary, concepts=out.concepts
        )
        for item in _items(_parse_json(client.json_completion(system, user)), "entities")[:max_entities]:
            evidence = _evidence(item)
            if not validate_evidence(evidence, sections, total_lines):
                warn("entity dropped", f"invalid evidence for {item.get('name')!r}")
                continue
            aliases = item.get("aliases") if isinstance(item.get("aliases"), list) else []
            out.entities.append(
                EntityExtract(
                    name=str(item.get("name") or "").strip(),
                    entity_type=str(item.get("type") or item.get("entity_type") or "entity").strip(),
                    description=str(item.get("description") or "").strip(),
                    aliases=[str(x) for x in aliases],
                    confidence=_float_or_none(item.get("confidence")),
                    evidence=evidence,
                )
            )
    except Exception as exc:  # noqa: BLE001
        warn("entity extraction failed", exc)

    allowed = {c.name for c in out.concepts} | {e.name for e in out.entities}
    try:
        system, user = relations_messages(
            markdown,
            sections,
            summary=out.summary,
            concepts=out.concepts,
            entities=out.entities,
        )
        for item in _items(_parse_json(client.json_completion(system, user)), "relations"):
            evidence = _evidence(item)
            subject, obj = str(item.get("subject") or ""), str(item.get("object") or "")
            if subject not in allowed or obj not in allowed:
                warn("relation dropped", f"unknown node: {subject!r} -> {obj!r}")
                continue
            if not validate_evidence(evidence, sections, total_lines):
                warn("relation dropped", f"invalid evidence: {subject!r} -> {obj!r}")
                continue
            out.relations.append(
                ProposedEdge(
                    subject=subject,
                    relation=str(item.get("relation") or "related_to"),
                    object=obj,
                    evidence=evidence,
                    note=str(item.get("note") or ""),
                )
            )
    except Exception as exc:  # noqa: BLE001
        warn("relation extraction failed", exc)
    return out


def _float_or_none(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
