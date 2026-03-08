"""Tests for exoclaw/utils/__init__.py coverage."""

from __future__ import annotations


class TestUtilsModule:
    def test_import_utils(self):
        import exoclaw.utils as utils
        assert hasattr(utils, "__all__")

    def test_all_is_empty_list(self):
        import exoclaw.utils as utils
        assert utils.__all__ == []
