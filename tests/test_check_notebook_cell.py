"""The notebook checker: magics are stripped only when a cell is not plain Python."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _check_notebook_cell import check_source  # noqa: E402


def test_percent_format_continuation_line_is_not_a_magic():
    src = 'x = 1\nprint("%s"\n      % (x,))\n'
    assert check_source(src) == []


def test_real_magics_are_still_stripped():
    src = '!pip install foo\n%cd /tmp\ny = 2\nprint(y)\n'
    assert check_source(src) == []


def test_use_before_definition_is_still_caught():
    assert check_source("print(z)\nz = 1\n") != []
