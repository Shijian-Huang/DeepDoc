import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator


class AnalysisCancelled(RuntimeError):
    pass


class AnalysisCancellationToken:
    def __init__(self):
        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        self._response = None

    def cancel(self) -> None:
        self._cancelled.set()
        with self._lock:
            response = self._response
        if response is not None:
            try:
                response.close()
            except Exception:
                pass

    def check(self) -> None:
        if self._cancelled.is_set():
            raise AnalysisCancelled("Analysis cancelled by user.")

    def attach_response(self, response: Any) -> None:
        with self._lock:
            self._response = response
        self.check()

    def detach_response(self, response: Any) -> None:
        with self._lock:
            if self._response is response:
                self._response = None


request_cancellation_token: ContextVar[AnalysisCancellationToken | None] = ContextVar(
    "request_cancellation_token",
    default=None,
)


@contextmanager
def use_analysis_cancellation(token: AnalysisCancellationToken) -> Iterator[None]:
    token_handle = request_cancellation_token.set(token)
    try:
        token.check()
        yield
    finally:
        request_cancellation_token.reset(token_handle)


def check_analysis_cancelled() -> None:
    token = request_cancellation_token.get()
    if token is not None:
        token.check()
