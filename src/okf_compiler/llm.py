"""Optional OpenAI-compatible LLM extraction for OKF bundles."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

from .diagnostics import DebugRecorder
from .prompts import concepts_messages, entities_messages, relations_messages, summary_messages
from .schema import (
    ConceptExtract,
    EntityExtract,
    Evidence,
    Extracts,
    ProposedEdge,
    SectionSpec,
    evidence_to_dict,
    resolve_evidence,
)

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

    def integer(name: str) -> int:
        value = raw.get(name)
        try:
            return int(value) if value not in (None, "") else 0
        except (TypeError, ValueError):
            return 0

    return Evidence(
        heading_path=str(raw.get("heading_path") or ""),
        line_start=integer("line_start"),
        line_end=integer("line_end"),
        section_id=str(raw.get("section_id") or ""),
        quote=str(raw.get("quote") or ""),
    )


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
    debug: DebugRecorder | None = None,
) -> Extracts:
    out = Extracts()

    def warn(label: str, exc: Exception | str) -> None:
        out.warnings.append(redact_secrets(f"{label}: {exc}", client.api_key))

    def complete(stage: str, system: str, user: str) -> tuple[str, dict | list]:
        if debug:
            debug.stage_request(stage, system, user)
        started = time.monotonic()
        raw = client.json_completion(system, user)
        value = _parse_json(raw)
        if debug:
            debug.stage_response(stage, raw, value)
            debug.event(
                "llm_stage_completed",
                stage=stage,
                duration_ms=round((time.monotonic() - started) * 1000, 2),
            )
        return raw, value

    try:
        system, user = summary_messages(markdown, sections, language)
        _, value = complete("summary", system, user)
        out.summary = str(value.get("summary", "")) if isinstance(value, dict) else ""
        out.stage_stats["summary"] = {"status": "ok" if out.summary else "degraded"}
    except Exception as exc:  # noqa: BLE001
        warn("summary extraction failed", exc)
        out.stage_stats["summary"] = {
            "status": "failed",
            "error": redact_secrets(str(exc), client.api_key),
        }
        if debug:
            debug.event(
                "llm_stage_failed",
                stage="summary",
                error=redact_secrets(str(exc), client.api_key),
            )

    try:
        system, user = concepts_messages(markdown, sections, max_concepts, summary=out.summary)
        _, value = complete("concepts", system, user)
        items = _items(value, "concepts")[:max_concepts]
        rejected = 0
        for item in items:
            name = str(item.get("name") or "").strip()
            result = resolve_evidence(_evidence(item), sections, markdown)
            if debug:
                debug.validation("concepts", name, item, result.to_dict())
            if not result.valid:
                rejected += 1
                warn("concept dropped", f"{result.reason} for {name!r}")
                out.validation.append({"stage": "concepts", "label": name, **result.to_dict()})
                continue
            out.concepts.append(
                ConceptExtract(
                    name=name,
                    description=str(item.get("description") or "").strip(),
                    confidence=_float_or_none(item.get("confidence")),
                    evidence=result.evidence,
                )
            )
        out.stage_stats["concepts"] = _stage_stats(len(items), len(out.concepts), rejected)
    except Exception as exc:  # noqa: BLE001
        warn("concept extraction failed", exc)
        out.stage_stats["concepts"] = _failed_stage(exc, client.api_key)
        if debug:
            debug.event(
                "llm_stage_failed",
                stage="concepts",
                error=redact_secrets(str(exc), client.api_key),
            )

    try:
        system, user = entities_messages(
            markdown, sections, max_entities, summary=out.summary, concepts=out.concepts
        )
        _, value = complete("entities", system, user)
        items = _items(value, "entities")[:max_entities]
        rejected = 0
        for item in items:
            name = str(item.get("name") or "").strip()
            result = resolve_evidence(_evidence(item), sections, markdown)
            if debug:
                debug.validation("entities", name, item, result.to_dict())
            if not result.valid:
                rejected += 1
                warn("entity dropped", f"{result.reason} for {name!r}")
                out.validation.append({"stage": "entities", "label": name, **result.to_dict()})
                continue
            aliases = item.get("aliases") if isinstance(item.get("aliases"), list) else []
            out.entities.append(
                EntityExtract(
                    name=name,
                    entity_type=str(item.get("type") or item.get("entity_type") or "entity").strip(),
                    description=str(item.get("description") or "").strip(),
                    aliases=[str(x) for x in aliases],
                    confidence=_float_or_none(item.get("confidence")),
                    evidence=result.evidence,
                )
            )
        out.stage_stats["entities"] = _stage_stats(len(items), len(out.entities), rejected)
    except Exception as exc:  # noqa: BLE001
        warn("entity extraction failed", exc)
        out.stage_stats["entities"] = _failed_stage(exc, client.api_key)
        if debug:
            debug.event(
                "llm_stage_failed",
                stage="entities",
                error=redact_secrets(str(exc), client.api_key),
            )

    allowed = {c.name for c in out.concepts} | {e.name for e in out.entities}
    try:
        system, user = relations_messages(
            markdown,
            sections,
            summary=out.summary,
            concepts=out.concepts,
            entities=out.entities,
        )
        _, value = complete("relations", system, user)
        items = _items(value, "relations")
        rejected = 0
        for item in items:
            subject = str(item.get("subject") or "")
            obj = str(item.get("object") or "")
            label = f"{subject!r} -> {obj!r}"
            if subject not in allowed or obj not in allowed:
                rejected += 1
                reason = "unknown_node"
                warn("relation dropped", f"{reason}: {label}")
                validation = {
                    "valid": False,
                    "reason": reason,
                    "evidence": evidence_to_dict(_evidence(item)),
                    "details": {},
                }
                out.validation.append({"stage": "relations", "label": label, **validation})
                if debug:
                    debug.validation("relations", label, item, validation)
                continue
            result = resolve_evidence(_evidence(item), sections, markdown)
            if debug:
                debug.validation("relations", label, item, result.to_dict())
            if not result.valid:
                rejected += 1
                warn("relation dropped", f"{result.reason}: {label}")
                out.validation.append({"stage": "relations", "label": label, **result.to_dict()})
                continue
            out.relations.append(
                ProposedEdge(
                    subject=subject,
                    relation=str(item.get("relation") or "related_to"),
                    object=obj,
                    evidence=result.evidence,
                    note=str(item.get("note") or ""),
                )
            )
        out.stage_stats["relations"] = _stage_stats(len(items), len(out.relations), rejected)
    except Exception as exc:  # noqa: BLE001
        warn("relation extraction failed", exc)
        out.stage_stats["relations"] = _failed_stage(exc, client.api_key)
        if debug:
            debug.event(
                "llm_stage_failed",
                stage="relations",
                error=redact_secrets(str(exc), client.api_key),
            )
    return out


def _stage_stats(returned: int, accepted: int, rejected: int) -> dict:
    status = "degraded" if rejected else "ok"
    return {"status": status, "returned": returned, "accepted": accepted, "rejected": rejected}


def _failed_stage(exc: Exception, api_key: str | None) -> dict:
    return {
        "status": "failed",
        "returned": 0,
        "accepted": 0,
        "rejected": 0,
        "error": redact_secrets(str(exc), api_key),
    }


def _float_or_none(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
