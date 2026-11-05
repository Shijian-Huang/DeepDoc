"""Per-run accounting of LLM calls: tokens and time, for cost and capacity estimates.

Providers report the token counts of their latest response with ``set_last_usage``;
``generate_json`` turns each call into an entry with ``record_llm_call``. Entries are
collected only inside ``track_llm_usage()``, so concurrent requests stay separate.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

TOKEN_FIELDS = ("prompt_tokens", "output_tokens", "thinking_tokens")

_active_calls: ContextVar[list[dict[str, Any]] | None] = ContextVar("llm_usage_calls", default=None)
_last_usage: ContextVar[dict[str, Any] | None] = ContextVar("llm_last_usage", default=None)


def set_last_usage(usage: dict[str, Any] | None) -> None:
    _last_usage.set(usage)


def take_last_usage() -> dict[str, Any]:
    usage = _last_usage.get() or {}
    _last_usage.set(None)
    return dict(usage)


@contextmanager
def track_llm_usage() -> Iterator[list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    token = _active_calls.set(calls)
    try:
        yield calls
    finally:
        _active_calls.reset(token)


def record_llm_call(entry: dict[str, Any]) -> None:
    calls = _active_calls.get()
    if calls is not None:
        calls.append(entry)


def summarize_llm_usage(calls: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Any] = {"call_count": len(calls)}
    for field in TOKEN_FIELDS:
        reported = [int(call[field]) for call in calls if call.get(field) is not None]
        # None, not 0, when no call reported the field. Ollama never reports thinking
        # tokens separately: they are counted in output_tokens.
        totals[field] = sum(reported) if reported or not calls else None
    totals["seconds"] = round(sum(float(call.get("seconds") or 0) for call in calls), 4)
    # False when a provider did not report token counts for some call, so totals undercount.
    totals["tokens_reported"] = all(
        call.get("prompt_tokens") is not None and call.get("output_tokens") is not None for call in calls
    )
    return {"calls": calls, "totals": totals}
