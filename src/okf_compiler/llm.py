"""Optional OpenAI-compatible LLM extraction for OKF bundles."""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

from .diagnostics import DebugRecorder
from .prompts import (
    concepts_messages,
    entities_messages,
    relation_nodes,
    relations_messages,
    retry_system_message,
    summary_messages,
)
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
_CANONICAL_ENTITY_TYPES = {
    "person",
    "organization",
    "product",
    "platform",
    "brand",
    "work",
    "location",
    "event",
    "other_named_entity",
}
_ENTITY_TYPE_ALIASES = {
    "人物": "person",
    "角色": "person",
    "作者": "person",
    "person": "person",
    "human": "person",
    "团队": "organization",
    "开发团队": "organization",
    "游戏开发团队": "organization",
    "公司": "organization",
    "工作室": "organization",
    "组织": "organization",
    "开发商": "organization",
    "organization": "organization",
    "company": "organization",
    "studio": "organization",
    "team": "organization",
    "游戏产品": "product",
    "游戏": "product",
    "产品": "product",
    "软件": "product",
    "应用": "product",
    "app": "product",
    "game": "product",
    "product": "product",
    "平台": "platform",
    "平台/渠道": "platform",
    "渠道平台": "platform",
    "platform": "platform",
    "品牌": "brand",
    "brand": "brand",
    "作品": "work",
    "文章": "work",
    "书籍": "work",
    "work": "work",
    "地点": "location",
    "地区": "location",
    "location": "location",
    "事件": "event",
    "活动": "event",
    "event": "event",
    "其他命名实体": "other_named_entity",
    "other_named_entity": "other_named_entity",
}
_CONCEPT_TYPE_MARKERS = (
    "概念",
    "机制",
    "模式",
    "策略",
    "玩法",
    "设计",
    "红利",
    "钩子",
    "商业",
    "时刻",
    "系统",
    "method",
    "mechanism",
    "concept",
    "strategy",
    "pattern",
    "model",
)


@dataclass
class LLMConfig:
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None
    timeout: float = _DEFAULT_TIMEOUT

    def is_configured(self) -> bool:
        return bool(self.model and self.model.strip())


class StageResponseError(ValueError):
    def __init__(self, stage: str, reason: str, attempts: int):
        super().__init__(f"malformed {stage} response after {attempts} attempt(s): {reason}")
        self.stage = stage
        self.reason = reason
        self.attempts = attempts


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
    return {str(key): str(value) for key, value in values.items() if key and value is not None}


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
        return self.json_completion("Return valid JSON only.", '{"ok": true}')


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


def _validate_stage_payload(stage: str, value: dict | list) -> dict:
    if not isinstance(value, dict):
        raise ValueError("top-level value must be an object")
    if stage == "summary":
        if "summary" not in value:
            raise ValueError("missing required top-level field 'summary'")
        if not isinstance(value["summary"], str):
            raise ValueError("top-level field 'summary' must be a string")
        return value
    if stage not in {"concepts", "entities", "relations"}:
        raise ValueError(f"unknown extraction stage: {stage}")
    if stage not in value:
        raise ValueError(f"missing required top-level array '{stage}'")
    if not isinstance(value[stage], list):
        raise ValueError(f"top-level field '{stage}' must be an array")
    if any(not isinstance(item, dict) for item in value[stage]):
        raise ValueError(f"every item in '{stage}' must be an object")
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

    def complete(stage: str, system: str, user: str) -> tuple[dict, int]:
        current_system = system
        for attempt in (1, 2):
            if debug:
                debug.stage_request(stage, current_system, user, attempt=attempt)
            started = time.monotonic()
            raw = client.json_completion(current_system, user)
            parsed: dict | list | None = None
            try:
                parsed = _parse_json(raw)
                value = _validate_stage_payload(stage, parsed)
            except Exception as exc:  # noqa: BLE001
                if debug:
                    debug.stage_response(stage, raw, parsed, attempt=attempt)
                    debug.event(
                        "llm_stage_invalid",
                        stage=stage,
                        attempt=attempt,
                        duration_ms=round((time.monotonic() - started) * 1000, 2),
                        error=redact_secrets(str(exc), client.api_key),
                    )
                if attempt == 1:
                    if debug:
                        debug.event(
                            "llm_stage_retry",
                            stage=stage,
                            attempt=attempt + 1,
                            reason=redact_secrets(str(exc), client.api_key),
                        )
                    current_system = retry_system_message(stage, system, str(exc))
                    continue
                raise StageResponseError(stage, str(exc), attempt) from exc
            if debug:
                debug.stage_response(stage, raw, value, attempt=attempt)
                debug.event(
                    "llm_stage_completed",
                    stage=stage,
                    attempt=attempt,
                    duration_ms=round((time.monotonic() - started) * 1000, 2),
                )
            return value, attempt
        raise AssertionError("unreachable")

    try:
        system, user = summary_messages(markdown, sections, language)
        value, attempts = complete("summary", system, user)
        out.summary = value["summary"].strip()
        out.stage_stats["summary"] = {
            "status": "ok" if out.summary else "degraded",
            "attempts": attempts,
            "retries": attempts - 1,
        }
    except Exception as exc:  # noqa: BLE001
        warn("summary extraction failed", exc)
        out.stage_stats["summary"] = _failed_stage(exc, client.api_key)
        _record_stage_failure(debug, "summary", exc, client.api_key)

    try:
        system, user = concepts_messages(markdown, sections, max_concepts, summary=out.summary)
        value, attempts = complete("concepts", system, user)
        items = value["concepts"][:max_concepts]
        rejected = 0
        seen_concepts: set[str] = set()
        for item in items:
            name = str(item.get("name") or "").strip()
            result = resolve_evidence(_evidence(item), sections, markdown)
            if not result.valid:
                rejected += 1
                _reject_item(out, debug, "concepts", name, item, result.to_dict())
                warn("concept dropped", f"{result.reason} for {name!r}")
                continue
            key = _name_key(name)
            if not key or key in seen_concepts:
                rejected += 1
                validation = _invalid_item("duplicate_or_empty_concept", result.evidence)
                _reject_item(out, debug, "concepts", name, item, validation)
                warn("concept dropped", f"duplicate_or_empty_concept for {name!r}")
                continue
            seen_concepts.add(key)
            out.concepts.append(
                ConceptExtract(
                    name=name,
                    description=str(item.get("description") or "").strip(),
                    confidence=_float_or_none(item.get("confidence")),
                    evidence=result.evidence,
                )
            )
        out.stage_stats["concepts"] = _stage_stats(
            len(items),
            len(out.concepts),
            rejected,
            attempts=attempts,
        )
    except Exception as exc:  # noqa: BLE001
        warn("concept extraction failed", exc)
        out.stage_stats["concepts"] = _failed_stage(exc, client.api_key)
        _record_stage_failure(debug, "concepts", exc, client.api_key)

    try:
        system, user = entities_messages(
            markdown,
            sections,
            max_entities,
            summary=out.summary,
            concepts=out.concepts,
        )
        value, attempts = complete("entities", system, user)
        items = value["entities"][:max_entities]
        rejected = 0
        seen_entities: set[str] = set()
        for item in items:
            name = str(item.get("name") or "").strip()
            result = resolve_evidence(_evidence(item), sections, markdown)
            if not result.valid:
                rejected += 1
                _reject_item(out, debug, "entities", name, item, result.to_dict())
                warn("entity dropped", f"{result.reason} for {name!r}")
                continue
            entity_type = _canonical_entity_type(item.get("type") or item.get("entity_type"))
            if entity_type is None:
                rejected += 1
                validation = _invalid_item(
                    "invalid_or_conceptual_entity_type",
                    result.evidence,
                    {"received_type": str(item.get("type") or item.get("entity_type") or "")},
                )
                _reject_item(out, debug, "entities", name, item, validation)
                warn("entity dropped", f"invalid_or_conceptual_entity_type for {name!r}")
                continue
            key = _name_key(name)
            if not key or key in seen_entities:
                rejected += 1
                validation = _invalid_item("duplicate_or_empty_entity", result.evidence)
                _reject_item(out, debug, "entities", name, item, validation)
                warn("entity dropped", f"duplicate_or_empty_entity for {name!r}")
                continue
            seen_entities.add(key)
            aliases = item.get("aliases") if isinstance(item.get("aliases"), list) else []
            out.entities.append(
                EntityExtract(
                    name=name,
                    entity_type=entity_type,
                    description=str(item.get("description") or "").strip(),
                    aliases=[str(alias) for alias in aliases],
                    confidence=_float_or_none(item.get("confidence")),
                    evidence=result.evidence,
                )
            )
        out.stage_stats["entities"] = _stage_stats(
            len(items),
            len(out.entities),
            rejected,
            attempts=attempts,
        )
        reclassified = _remove_entity_names_from_concepts(out)
        if reclassified:
            stats = out.stage_stats.get("concepts", {})
            stats["accepted"] = len(out.concepts)
            stats["reclassified"] = reclassified
    except Exception as exc:  # noqa: BLE001
        warn("entity extraction failed", exc)
        out.stage_stats["entities"] = _failed_stage(exc, client.api_key)
        _record_stage_failure(debug, "entities", exc, client.api_key)

    try:
        nodes = relation_nodes(out.concepts, out.entities)
        node_by_id = {node["node_id"]: node for node in nodes}
        node_by_name = {_name_key(node["name"]): node for node in nodes}
        system, user = relations_messages(
            markdown,
            sections,
            summary=out.summary,
            concepts=out.concepts,
            entities=out.entities,
        )
        value, attempts = complete("relations", system, user)
        items = value["relations"]
        rejected = 0
        for item in items:
            subject_node, object_node = _relation_nodes(item, node_by_id, node_by_name)
            subject_label = subject_node["name"] if subject_node else str(item.get("subject") or "")
            object_label = object_node["name"] if object_node else str(item.get("object") or "")
            label = f"{subject_label!r} -> {object_label!r}"
            if subject_node is None or object_node is None:
                rejected += 1
                validation = _invalid_item("unknown_node", _evidence(item))
                _reject_item(out, debug, "relations", label, item, validation)
                warn("relation dropped", f"unknown_node: {label}")
                continue
            result = resolve_evidence(_evidence(item), sections, markdown)
            if not result.valid:
                rejected += 1
                _reject_item(out, debug, "relations", label, item, result.to_dict())
                warn("relation dropped", f"{result.reason}: {label}")
                continue
            out.relations.append(
                ProposedEdge(
                    subject=subject_node["name"],
                    relation=str(item.get("relation") or "related_to"),
                    object=object_node["name"],
                    evidence=result.evidence,
                    note=str(item.get("note") or ""),
                )
            )
        out.stage_stats["relations"] = _stage_stats(
            len(items),
            len(out.relations),
            rejected,
            attempts=attempts,
        )
    except Exception as exc:  # noqa: BLE001
        warn("relation extraction failed", exc)
        out.stage_stats["relations"] = _failed_stage(exc, client.api_key)
        _record_stage_failure(debug, "relations", exc, client.api_key)
    return out


def _canonical_entity_type(value) -> str | None:
    raw = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    if not raw:
        return None
    if raw in _CANONICAL_ENTITY_TYPES:
        return raw
    if raw in _ENTITY_TYPE_ALIASES:
        return _ENTITY_TYPE_ALIASES[raw]
    if any(marker in raw for marker in _CONCEPT_TYPE_MARKERS):
        return None
    for alias, canonical in _ENTITY_TYPE_ALIASES.items():
        if alias and alias in raw:
            return canonical
    return None


def _remove_entity_names_from_concepts(out: Extracts) -> int:
    entity_names = {_name_key(entity.name) for entity in out.entities}
    before = len(out.concepts)
    out.concepts = [concept for concept in out.concepts if _name_key(concept.name) not in entity_names]
    return before - len(out.concepts)


def _relation_nodes(item: dict, node_by_id: dict, node_by_name: dict) -> tuple[dict | None, dict | None]:
    subject_id = str(item.get("subject_id") or "").strip()
    object_id = str(item.get("object_id") or "").strip()
    if subject_id or object_id:
        return node_by_id.get(subject_id), node_by_id.get(object_id)
    subject = node_by_name.get(_name_key(str(item.get("subject") or "")))
    obj = node_by_name.get(_name_key(str(item.get("object") or "")))
    return subject, obj


def _reject_item(
    out: Extracts,
    debug: DebugRecorder | None,
    stage: str,
    label: str,
    item: dict,
    validation: dict,
) -> None:
    out.validation.append({"stage": stage, "label": label, **validation})
    if debug:
        debug.validation(stage, label, item, validation)


def _invalid_item(reason: str, evidence: Evidence | None, details: dict | None = None) -> dict:
    return {
        "valid": False,
        "reason": reason,
        "evidence": evidence_to_dict(evidence),
        "details": details or {},
    }


def _record_stage_failure(
    debug: DebugRecorder | None,
    stage: str,
    exc: Exception,
    api_key: str | None,
) -> None:
    if debug:
        debug.event(
            "llm_stage_failed",
            stage=stage,
            attempts=getattr(exc, "attempts", 1),
            error=redact_secrets(str(exc), api_key),
        )


def _stage_stats(
    returned: int,
    accepted: int,
    rejected: int,
    *,
    attempts: int,
    reclassified: int = 0,
) -> dict:
    status = "degraded" if rejected else "ok"
    return {
        "status": status,
        "returned": returned,
        "accepted": accepted,
        "rejected": rejected,
        "reclassified": reclassified,
        "attempts": attempts,
        "retries": attempts - 1,
    }


def _failed_stage(exc: Exception, api_key: str | None) -> dict:
    attempts = int(getattr(exc, "attempts", 1))
    return {
        "status": "failed",
        "returned": 0,
        "accepted": 0,
        "rejected": 0,
        "attempts": attempts,
        "retries": max(attempts - 1, 0),
        "error": redact_secrets(str(exc), api_key),
    }


def _name_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", "", normalized)


def _float_or_none(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
