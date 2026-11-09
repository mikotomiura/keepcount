"""The content digest of a run's record: the SHA-256 of the record without its provenance.

The records of runs v6k to v6s are not in this archive (README, "What is not in this archive"). A record rebuilt from
the conditions, the code and the keystroke data differs from the original file in its provenance only (the time of
the run, the commit, and the SHA-256 of files that name earlier records), so its file's SHA-256 differs; its content
digest does not. `results/record-digests.json` holds, per run, the SHA-256 of the original file (as the conditions of
later runs name it) and its content digest.

Usage:
  python tools/record_digest.py results/prototype_v6k/record.json     # print the content digest
  python tools/record_digest.py --check v6k                           # compare with results/record-digests.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "results" / "record-digests.json"


def content_digest(path: Path) -> str:
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rec, dict) or "provenance" not in rec:
        raise SystemExit(f"{path}: not a run's record (no provenance)")
    rec.pop("provenance")
    text = json.dumps(rec, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def table() -> dict:
    return json.loads(TABLE.read_text(encoding="utf-8"))["runs"]


def check(run: str) -> bool:
    want = table()[run]["content_sha256"]
    got = content_digest(ROOT / "results" / f"prototype_{run}" / "record.json")
    print(f"{run}: {'same content as the original record' if got == want else 'DIFFERS from the original record'}"
          f" ({got})")
    return got == want


def main(argv: list[str]) -> None:
    if len(argv) == 2 and argv[0] == "--check":
        sys.exit(0 if check(argv[1]) else 1)
    if len(argv) == 1:
        print(content_digest(Path(argv[0])))
        return
    sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
