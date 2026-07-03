"""Structured logging utility for Context Engine.

Provides JSON-formatted logging with context and proper exception handling.
"""
import logging
import json
import os
import sys
import traceback
from functools import lru_cache
from typing import Any, Dict, Optional, Union
from datetime import datetime, timezone

# Configure root logger with LOG_LEVEL from environment
_log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
_log_level = getattr(logging, _log_level_str, logging.INFO)
logging.basicConfig(
    level=_log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Cache loggers to avoid repeated lookups
_logger_cache: Dict[str, logging.Logger] = {}


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""
    
    __slots__ = ()
    
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }
        
        # Add extra fields
        extra = getattr(record, 'extra_fields', None)
        if extra:
            log_data.update(extra)
        
        return json.dumps(log_data, default=str)


def get_logger(name: str, json_format: bool = False) -> logging.Logger:
    """Get a logger instance with optional JSON formatting.
    
    Args:
        name: Logger name (typically __name__)
        json_format: If True, use JSON formatter; otherwise use default
    
    Returns:
        Configured logger instance
    """
    cache_key = f"{name}:{json_format}"
    if cache_key in _logger_cache:
        return _logger_cache[cache_key]
    
    logger = logging.getLogger(name)
    logger.setLevel(_log_level)
    
    if json_format and not any(isinstance(h.formatter, JSONFormatter) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    
    _logger_cache[cache_key] = logger
    return logger


class ContextLogger:
    """Logger wrapper that adds context fields to all log messages."""
    
    __slots__ = ('logger', 'context')
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
    
    def _log(self, level: int, msg: str, exc_info: Any = None, **extra):
        """Internal log method that merges context."""
        if not self.logger.isEnabledFor(level):
            return
        merged = {**self.context, **extra} if extra else self.context
        record = self.logger.makeRecord(
            self.logger.name,
            level,
            "(unknown file)",
            0,
            msg,
            (),
            exc_info,
        )
        record.extra_fields = merged
        self.logger.handle(record)
    
    def debug(self, msg: str, **extra):
        self._log(logging.DEBUG, msg, **extra)
    
    def info(self, msg: str, **extra):
        self._log(logging.INFO, msg, **extra)
    
    def warning(self, msg: str, **extra):
        self._log(logging.WARNING, msg, **extra)
    
    def error(self, msg: str, exc_info: Any = None, **extra):
        self._log(logging.ERROR, msg, exc_info=exc_info, **extra)
    
    def exception(self, msg: str, **extra):
        """Log an exception with traceback."""
        self._log(logging.ERROR, msg, exc_info=sys.exc_info(), **extra)
    
    def critical(self, msg: str, exc_info: Any = None, **extra):
        self._log(logging.CRITICAL, msg, exc_info=exc_info, **extra)
    
    def isEnabledFor(self, level: int) -> bool:
        """Check if this logger is enabled for the given level."""
        return self.logger.isEnabledFor(level)


# Custom exceptions for Context Engine
class ContextEngineError(Exception):
    """Base exception for all Context Engine errors."""
    pass


class RetrievalError(ContextEngineError):
    """Error during retrieval/search operations."""
    pass


class IndexingError(ContextEngineError):
    """Error during indexing operations."""
    pass


class DecoderError(ContextEngineError):
    """Error during LLM decoder operations."""
    pass


class ValidationError(ContextEngineError):
    """Error during input validation."""
    pass


class ConfigurationError(ContextEngineError):
    """Error in configuration or environment setup."""
    pass


# Convenience function for exception handling with logging
def log_and_reraise(logger: logging.Logger, msg: str, exc: Exception, **context):
    """Log an exception with context and re-raise it.
    
    Args:
        logger: Logger instance
        msg: Error message
        exc: Exception to log and re-raise
        **context: Additional context fields
    """
    if isinstance(logger, ContextLogger):
        logger.exception(msg, **context)
    else:
        logger.exception(msg, extra=context)
    raise exc


def safe_int(value: Any, default: int, logger: Optional[logging.Logger] = None, context: str = "") -> int:
    """Safely convert value to int with logging on failure.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        logger: Optional logger for warnings
        context: Context string for log message
    
    Returns:
        Converted int or default
    """
    try:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return default
        return int(value)
    except (ValueError, TypeError) as e:
        if logger:
            logger.warning(f"Failed to convert {context} to int: {value}", exc_info=e)
        return default


def safe_float(value: Any, default: float, logger: Optional[logging.Logger] = None, context: str = "") -> float:
    """Safely convert value to float with logging on failure.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        logger: Optional logger for warnings
        context: Context string for log message
    
    Returns:
        Converted float or default
    """
    try:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return default
        return float(value)
    except (ValueError, TypeError) as e:
        if logger:
            logger.warning(f"Failed to convert {context} to float: {value}", exc_info=e)
        return default


def safe_bool(value: Any, default: bool, logger: Optional[logging.Logger] = None, context: str = "") -> bool:
    """Safely convert value to bool with logging on failure.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        logger: Optional logger for warnings
        context: Context string for log message
    
    Returns:
        Converted bool or default
    """
    try:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return default
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in {"1", "true", "yes", "on"}:
            return True
        if s in {"0", "false", "no", "off"}:
            return False
        return default
    except (ValueError, TypeError) as e:
        if logger:
            logger.warning(f"Failed to convert {context} to bool: {value}", exc_info=e)
        return default

