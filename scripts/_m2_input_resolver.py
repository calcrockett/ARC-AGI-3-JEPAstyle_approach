"""Make the milestone-2 derived notebooks independent of Kaggle's /kaggle/input mount layout.

The upstream notebook hardcodes /kaggle/input/datasets/<owner>/<slug>, /kaggle/input/models/<owner>/... and
/kaggle/input/competitions/<comp>; some GPU sessions mount the older /kaggle/input/<slug> instead
(JustAdev742, 2026-10-07: 5 of 9 sessions). apply_input_resolver() edits a notebook in place:

  * inlines kaggle_submission_milestone2_fork/input_resolver/input_resolver.py at the top of the cell that
    defines the path constants (anchor "# model and dataset paths");
  * rewrites each `NAME_DIR = '/kaggle/input/...'` there to `NAME_DIR = resolve_input('name', '...')`;
  * defines COMPETITION_WHEELS_DIR the same way and uses it for the two competition literals (the offline
    wheel install and the offline environment files).

/kaggle/input is read-only, so nothing is symlinked; every downstream cell keeps reading its constant.
The incumbent v1 notebook (kaggle_submission_m2_level_memory/) is NOT built through this.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HELPER = ROOT / "kaggle_submission_milestone2_fork" / "input_resolver" / "input_resolver.py"
ANCHOR = "# model and dataset paths\n"
COMPETITION_WHEELS = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"
COMPETITION_NAME = "COMPETITION_WHEELS_DIR"
MARKER = "INPUT_RESOLVED"
ASSIGN = re.compile(r"^(?P<n>[A-Z][A-Z_]*_DIR)(?P<sp>[ \t]*)=[ \t]*'(?P<p>/kaggle/input/[^'\n]+)'(?P<tail>[^\n]*)$", re.M)
LITERAL = re.compile(r"""(['"])(/kaggle/input[^'"\n]*)\1""")


def helper_source() -> str:
    text = HELPER.read_text(encoding="utf-8")
    assert "'''" not in text
    return text


def _name(var: str) -> str:
    return var[:-len("_DIR")].lower()


def apply_input_resolver(nb: dict) -> int:
    """Edit `nb` in place; return the number of resolved inputs (path constants + the competition wheels)."""
    cells = nb["cells"]
    src = ["".join(c["source"]) for c in cells]
    hits = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and ANCHOR in src[i]]
    if len(hits) != 1:
        raise SystemExit(f"REFUSING TO BUILD -- input-resolver anchor {ANCHOR.strip()!r} found in cells {hits}")
    i = hits[0]
    cell, found = src[i], list(ASSIGN.finditer(src[i]))
    if not found:
        raise SystemExit("REFUSING TO BUILD -- no NAME_DIR = '/kaggle/input/...' constant in the paths cell")
    cell = ASSIGN.sub(lambda m: f"{m['n']}{m['sp']}= resolve_input({_name(m['n'])!r}, {m['p']!r}){m['tail']}", cell)
    last = list(re.finditer(r"^.*resolve_input\(.*$\n", cell, re.M))[-1]
    cell = (cell[:last.end()]
            + f"{COMPETITION_NAME} = resolve_input('competition_wheels', {COMPETITION_WHEELS!r})\n" + cell[last.end():])
    cell = cell.replace(ANCHOR, helper_source() + "\n" + ANCHOR, 1)
    src[i] = cell
    n = 0
    for j, s in enumerate(src):
        if j == i or cells[j]["cell_type"] != "code":
            continue
        for q in ("'", '"'):
            lit = f"{q}{COMPETITION_WHEELS}{q}"
            n += s.count(lit)
            s = s.replace(lit, COMPETITION_NAME)
        src[j] = s
    if n != 2:
        raise SystemExit(f"REFUSING TO BUILD -- expected 2 competition wheel literals outside the paths cell, found {n}")
    for c, s in zip(cells, src):
        if s != "".join(c["source"]):     # leave untouched cells in their original encoding
            c["source"] = s.splitlines(True)
    return len(found) + 1


def unresolved_literals(nb: dict, skip=()) -> list[tuple[int, str]]:
    """(cell index, literal) for every /kaggle/input string literal in a code cell that is not an argument
    of resolve_input() -- i.e. a path that would still depend on one mount layout. `skip`: cell indices that
    carry their own resolver (the vLLM launcher)."""
    out = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code" or i in skip:
            continue
        s = "".join(c["source"])
        helper_span = (s.find("# >>> [calamitychasm] INPUT RESOLVER"), s.find("# <<< [calamitychasm] INPUT RESOLVER"))
        for m in LITERAL.finditer(s):
            if helper_span[0] >= 0 and helper_span[0] <= m.start() < helper_span[1]:
                continue
            if re.search(r"resolve_input\('[a-z_]+', $", s[:m.start()]):
                continue
            out.append((i, m.group(2)))
    return out
