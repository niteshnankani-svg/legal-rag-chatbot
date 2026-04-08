"""
app/core/logger.py
──────────────────────────────────────────────
Structured logging for the entire project.
Every file uses this logger to print messages.

Usage in any file:
    from app.core.logger import get_logger
    log = get_logger(__name__)
    log.info("something happened", key="value")
"""
import logging
import structlog


def setup_logging(level: str = "info") -> None:
    """
    Call this once when the app starts.
    Sets up structured JSON logging.
    """
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def get_logger(name: str):
    """
    Returns a logger for any file.
    Pass __name__ so we know which file the log came from.

    Example:
        log = get_logger(__name__)
        log.info("query_received", question="What is BNS?")
        log.error("something_failed", error=str(e))
    """
    return structlog.get_logger(name)