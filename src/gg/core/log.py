import logging
import re
import sys
from collections.abc import MutableMapping
from typing import Any, Literal

import structlog
from structlog.typing import EventDict, Processor

from gg.core.jsonutil import dumps_str

_DROP_KEYS = frozenset(
    {"authorization", "api_key", "headers", "messages", "prompt", "content", "text", "body", "vault"}
)
_SECRET_PATTERNS = (
    (re.compile(r"gg-(live|test)-[A-Za-z0-9]{6,}"), r"gg-\1-****"),
    (re.compile(r"sk-[A-Za-z0-9_-]{10,}"), "sk-****"),
    (re.compile(r"AIza[0-9A-Za-z_-]{30,}"), "AIza****"),
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), "hf_****"),
    (re.compile(r"Bearer\s+\S+", re.IGNORECASE), "Bearer ****"),
)


def mask_secrets(value: str) -> str:
    for pattern, replacement in _SECRET_PATTERNS:
        value = pattern.sub(replacement, value)
    return value


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return mask_secrets(value)
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items() if str(k).lower() not in _DROP_KEYS}
    if isinstance(value, (list, tuple)):
        return [_redact(v) for v in value]
    return value


def redact_processor(_: Any, __: str, event_dict: EventDict) -> EventDict:
    out: MutableMapping[str, Any] = {}
    for k, v in event_dict.items():
        if k.lower() in _DROP_KEYS:
            continue
        out[k] = _redact(v)
    return dict(out)


def configure_logging(level: str = "info", fmt: Literal["json", "console"] = "json") -> None:
    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        redact_processor,
    ]
    renderer: Processor = (
        structlog.processors.JSONRenderer(serializer=lambda obj, **_: dumps_str(obj))
        if fmt == "json"
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelNamesMapping()[level.upper()]),
        cache_logger_on_first_use=False,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    logging.getLogger("uvicorn.access").disabled = True
    for noisy in ("httpx", "httpx2", "httpcore", "httpcore2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
