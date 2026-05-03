from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from google.genai import errors as genai_errors
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from services.cache_runtime import LLM_CACHE

log_batch = logging.getLogger("legal_analyzer.batch")

DEFAULT_CHAT_MODEL_CHAIN = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]
RETRY_MAX_ATTEMPTS = 4
RETRY_WAIT = wait_exponential(multiplier=1, min=1, max=16)

def is_transient_error(exc: BaseException) -> bool:
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        return getattr(exc, "code", None) == 429
    return False


def is_model_unavailable_error(exc: BaseException) -> bool:
    if isinstance(exc, genai_errors.ClientError):
        return getattr(exc, "code", None) == 404
    msg = str(exc).lower()
    return "not_found" in msg or "404" in msg or "no longer available" in msg


def invoke_with_retry(
    llm: Any,
    messages: Any,
    *,
    context: str = "",
    model_name: str = "",
    stats: Optional[Dict[str, int]] = None,
) -> Any:
    def on_retry(retry_state: Any) -> None:
        if stats is not None:
            stats["llm_retries"] = stats.get("llm_retries", 0) + 1
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        sleep_s = retry_state.next_action.sleep if retry_state.next_action else 0.0
        log_batch.warning(
            "%s: %s failed (%s), retry %d/%d in %.1fs",
            context or "llm",
            model_name or "?",
            type(exc).__name__ if exc else "UnknownError",
            retry_state.attempt_number,
            RETRY_MAX_ATTEMPTS,
            sleep_s,
        )

    retrying = Retrying(
        reraise=True,
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        wait=RETRY_WAIT,
        retry=retry_if_exception(is_transient_error),
        before_sleep=on_retry,
    )
    for attempt in retrying:
        with attempt:
            return llm.invoke(messages)
    raise RuntimeError("Retrying exited without a result")
