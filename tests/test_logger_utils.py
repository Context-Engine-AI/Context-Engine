"""Unit tests for scripts.logger helper functions."""

import logging

import scripts.logger as logger_mod


def test_safe_converters_default_on_invalid_input():
    log = logging.getLogger("test.logger")

    assert logger_mod.safe_int("", 42, logger=log, context="int_field") == 42
    assert logger_mod.safe_int("17", 0) == 17

    assert logger_mod.safe_float("abc", 3.14, logger=log, context="float_field") == 3.14
    assert logger_mod.safe_float("2.5", 0.0) == 2.5

    assert logger_mod.safe_bool("yes", False) is True
    assert logger_mod.safe_bool("NO", True) is False
    assert logger_mod.safe_bool("maybe", True, logger=log, context="bool_field") is True


def test_context_logger_injects_extra_fields(caplog):
    base_logger = logger_mod.get_logger("context-engine-test", json_format=False)
    contextual = logger_mod.ContextLogger(base_logger, stage="test", request_id="abc123")

    with caplog.at_level(logging.INFO, logger="context-engine-test"):
        contextual.info("hello", answer="ok")

    assert any("hello" in message for message in caplog.messages)
    # Ensure contextual fields are preserved
    record = caplog.records[-1]
    assert getattr(record, "extra_fields", {}).get("stage") == "test"
    assert getattr(record, "extra_fields", {}).get("request_id") == "abc123"
    assert getattr(record, "extra_fields", {}).get("answer") == "ok"
