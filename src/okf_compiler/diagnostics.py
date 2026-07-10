"""Structured, opt-in diagnostics written outside the OKF bundle."""

from __future__ import annotations

import json
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .atomic import atomic_write_json, atomic_write_text


class DebugRecorder:
    def __init__(self, root: Path, *, include_llm_payloads: bool = False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.include_llm_payloads = include_llm_payloads
        self._lock = threading.Lock()
        self._events: list[dict] = []

    def event(self, event: str, **data) -> None:
        row = {"timestamp": _now(), "event": event, **data}
        with self._lock:
            self._events.append(row)
            path = self.root / "compiler.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def stage_request(self, stage: str, system: str, user: str) -> None:
        self.event(
            "llm_request",
            stage=stage,
            system_chars=len(system),
            user_chars=len(user),
            payload_saved=self.include_llm_payloads,
        )
        if self.include_llm_payloads:
            base = self.root / "stages" / stage
            atomic_write_json(base / "request.json", {"system": system, "user": user})

    def stage_response(self, stage: str, raw: str, parsed: object | None = None) -> None:
        self.event(
            "llm_response",
            stage=stage,
            response_chars=len(raw),
            payload_saved=self.include_llm_payloads,
        )
        if self.include_llm_payloads:
            base = self.root / "stages" / stage
            atomic_write_text(base / "response.raw.txt", raw)
            if parsed is not None:
                atomic_write_json(base / "response.parsed.json", parsed)

    def validation(self, stage: str, label: str, item: object, result: object) -> None:
        base = self.root / "validation"
        base.mkdir(parents=True, exist_ok=True)
        row = {
            "timestamp": _now(),
            "stage": stage,
            "label": label,
            "item": item,
            "result": result,
        }
        with self._lock:
            with (base / f"{stage}.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def traceback(
        self,
        exc: BaseException,
        *,
        sanitizer: Callable[[str], str] | None = None,
    ) -> None:
        rendered = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        message = str(exc)
        if sanitizer:
            rendered = sanitizer(rendered)
            message = sanitizer(message)
        atomic_write_text(self.root / "traceback.log", rendered)
        self.event("exception", exception_type=type(exc).__name__, message=message)

    def finish(self, data: dict) -> None:
        payload = {"finished_at": _now(), **data, "events": self._events}
        atomic_write_json(self.root / "run.json", payload)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
