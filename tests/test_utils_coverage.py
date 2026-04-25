"""Tests for exoclaw/utils/__init__.py coverage."""

from __future__ import annotations


class TestUtilsModule:
    def test_import_utils(self) -> None:
        import exoclaw.utils as utils

        assert hasattr(utils, "__all__")

    def test_all_exposes_create_isolated_task(self) -> None:
        import exoclaw.utils as utils

        assert "create_isolated_task" in utils.__all__
