"""Tests that read the records of runs v6k and v6l are skipped, with the reason, when those records are absent.

The records of v6k to v6s are not in this archive (README, "What is not in this archive"); they are rebuilt with
tools/rebuild_records.py. The modules test_prototype_v6n to v6s read the records of v6k and v6l when they are
imported, and the fixture `whole` of test_prototype_v6m reads them; both pass synthetic persons through the counts with
the records' structure. With the records present (the working repository, or after a rebuild) every test runs.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NEEDED = [ROOT / "results" / f"prototype_{r}" / "record.json" for r in ("v6k", "v6l")]
MISSING = [p.relative_to(ROOT).as_posix() for p in NEEDED if not p.exists()]
REASON = f"needs {', '.join(MISSING)} (not archived; rebuild with tools/rebuild_records.py)"
IMPORT_READS = [f"test_prototype_{r}.py" for r in ("v6n", "v6o", "v6p", "v6q", "v6r", "v6s")]

collect_ignore = IMPORT_READS if MISSING else []


def pytest_collection_modifyitems(config, items):
    if not MISSING:
        return
    for item in items:
        if item.path.name == "test_prototype_v6m.py" and "whole" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.skip(reason=REASON))


def pytest_report_header(config):
    if MISSING:
        return [f"not collected: {', '.join(IMPORT_READS)}; skipped: tests of test_prototype_v6m.py that use "
                f"the fixture 'whole' -- {REASON}"]
    return []


def pytest_terminal_summary(terminalreporter):
    # shown also with -q, where the header is not printed
    if MISSING:
        terminalreporter.write_line(f"not collected: {', '.join(IMPORT_READS)}; skipped: the tests of "
                                    f"test_prototype_v6m.py that use 'whole' -- {REASON}")
