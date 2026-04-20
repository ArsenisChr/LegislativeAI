"""One-stop logging configuration for the legal analyzer app.

Call `setup_logging()` exactly once at process startup (from `main.py`).
Every module then obtains its logger via
`logging.getLogger("legal_analyzer.<area>")` and writes structured
messages with consistent formatting.

Areas currently used:
  - legal_analyzer.parser : comments_parser.py (phases & summary)
  - legal_analyzer.ident  : TargetIdentifier init / corpus embedding
  - legal_analyzer.batch  : narrow_batch / narrow_within_range / free
  - legal_analyzer.cache  : disk cache reads/writes (debug-level)

Format: `[HH:MM:SS LEVEL area    ] message`
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


_CONFIGURED = False

_FORMAT = "[%(asctime)s %(levelname)-7s %(name_short)-6s] %(message)s"
_DATEFMT = "%H:%M:%S"


class _ShortNameFilter(logging.Filter):
    """Inject `name_short` into the log record.

    Converts `legal_analyzer.parser` into `parser` so the log prefix stays
    compact regardless of how deep the logger hierarchy grows.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.name_short = record.name.rsplit(".", 1)[-1]
        return True


def setup_logging(level: int = logging.INFO, stream: Optional[object] = None) -> None:
    """Configure the `legal_analyzer` logger hierarchy.

    Idempotent: safe to call multiple times (only configures once). In
    Streamlit the whole script re-runs on every interaction, so guarding
    against double-configuration matters to avoid duplicate handlers.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    logger = logging.getLogger("legal_analyzer")
    logger.setLevel(level)
    # Don't propagate up to the root logger, because Streamlit attaches its
    # own handlers there and we'd end up with doubled lines.
    logger.propagate = False

    handler = logging.StreamHandler(stream=stream or sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    handler.addFilter(_ShortNameFilter())
    logger.addHandler(handler)

    _CONFIGURED = True
