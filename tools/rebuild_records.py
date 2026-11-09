"""Rebuild the records of runs v6k to v6s, in order, on a local branch, and check each against the original's content.

The records of v6k to v6s are not in this archive. Each later run checks the SHA-256 of the earlier records and
conditions named in its conditions ("earlier") and that every file it reads equals the last commit. A rebuilt record
has the same content as the original but a different provenance, so its file's SHA-256 differs from the one named.
This tool, run on a branch other than main with a clean working tree and the keystroke data in external/ (README,
Reproduce):

  1. runs works/prototype_<run>.py;
  2. stops unless the new record's content digest (tools/record_digest.py) equals the original's
     (results/record-digests.json);
  3. in the conditions of the later runs, replaces the original record's SHA-256 by the rebuilt one's, and then the
     SHA-256 of every conditions file so changed by its new one (later conditions name earlier conditions too);
  4. commits the record and the changed conditions on the branch, so that the next run reads committed bytes.

The changes to the conditions are local to the branch and exist only to let the chain run on rebuilt records; no other
byte of a condition is changed (the tool checks, on the bytes, that each changed file differs from its commit only
by those replacements). Do not push or merge the branch: it holds the rebuilt records.

Usage:  git switch -c rebuild && uv run --frozen python tools/rebuild_records.py [--from v6m]
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import record_digest as rd  # noqa: E402

RUNS = ["v6k", "v6l", "v6m", "v6n", "v6o", "v6p", "v6q", "v6r", "v6s"]


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cond(run: str) -> Path:
    return ROOT / "works" / "conditions" / f"prototype_{run}.json"


def replace_hashes(later: list[str], old: str, new: str, done: dict[str, str]) -> list[str]:
    """Replace old by new in the conditions of the later runs, as bytes; cascade to the hashes of the conditions so
    changed (later conditions name earlier conditions too). Every replacement made is added to done (old -> new)."""
    changed: list[str] = []
    pending = [(old, new)]
    while pending:
        o, n = pending.pop(0)
        if o == n:  # nothing changes, and nothing would cascade (Codex 7)
            continue
        done[o] = n
        for run in later:
            p = cond(run)
            raw = p.read_bytes()
            if o.encode() not in raw:
                continue
            before = sha(p)
            p.write_bytes(raw.replace(o.encode(), n.encode()))
            pending.append((before, sha(p)))
            if run not in changed:
                changed.append(run)
    return changed


def final(h: str, done: dict[str, str]) -> str:
    """Where a SHA-256 ends after the replacements (a file changed twice in one cascade gives old -> mid -> new)."""
    seen = set()
    while h in done and h not in seen:
        seen.add(h)
        h = done[h]
    return h


def only_hashes_differ(run: str, done: dict[str, str]) -> None:
    """The conditions differ from their committed bytes only where an old SHA-256 was replaced by its new one
    (bytes compared, so a change of line endings also stops)."""
    import re
    committed = subprocess.run(["git", "show", f"HEAD:works/conditions/prototype_{run}.json"], cwd=ROOT,
                               capture_output=True, check=True).stdout
    now = cond(run).read_bytes()
    hexes = re.compile(rb"[0-9a-f]{64}")
    if hexes.sub(b"#", committed) != hexes.sub(b"#", now):
        raise SystemExit(f"{run}: the conditions differ from their commit beyond SHA-256 values")
    for a, b in zip(hexes.findall(committed), hexes.findall(now)):
        if a != b and final(a.decode(), done) != b.decode():
            raise SystemExit(f"{run}: a SHA-256 changed that this tool did not replace ({a.decode()[:12]})")


def check_chain(start: int, table: dict) -> None:
    """Before any long computation: each later run's conditions name the original records by the SHA-256 of the
    table (or, from --from, the records already rebuilt on this branch by their file's SHA-256)."""
    for i, run in enumerate(RUNS[start:], start):
        earlier = json.loads(cond(run).read_text(encoding="utf-8")).get("earlier", {})
        for prev in RUNS[:i]:
            key = f"results/prototype_{prev}/record.json"
            if key not in earlier:
                continue
            rec = ROOT / key
            want = sha(rec) if rec.exists() else table[prev]["record_sha256"]
            if earlier[key] != want:
                raise SystemExit(f"{run}'s conditions name {key} by a SHA-256 that is neither the original's nor "
                                 f"that of the record on this branch")


def stop(run: str, why: str) -> None:
    raise SystemExit(f"{run}: {why}\nTo go back: git restore works/conditions && rm -rf results/prototype_{run}; "
                     f"then fix the cause and run again with --from {run}")


def main(argv: list[str]) -> None:
    if not argv:
        start = 0
    elif len(argv) == 2 and argv[0] == "--from" and argv[1] in RUNS:
        start = RUNS.index(argv[1])
    else:
        raise SystemExit(__doc__)
    if git("branch", "--show-current").strip() in ("main", ""):
        raise SystemExit("run this on a local branch other than main (git switch -c rebuild)")
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("the working tree has uncommitted changes")
    table = rd.table()
    check_chain(start, table)
    done: dict[str, str] = {}
    for i, run in enumerate(RUNS[start:], start):
        out = ROOT / "results" / f"prototype_{run}"
        if out.exists():
            raise SystemExit(f"{out} exists; this tool rebuilds only records that are absent")
        print(f"== {run}: running works/prototype_{run}.py", flush=True)
        r = subprocess.run([sys.executable, str(ROOT / "works" / f"prototype_{run}.py")], cwd=ROOT)
        if r.returncode != 0:
            stop(run, f"the run stopped (exit {r.returncode})")
        if not rd.check(run):
            stop(run, "the rebuilt record's content differs from the original (a faithful rebuild needs the same OS, "
                      "Python and uv.lock as the original, macOS on Apple silicon); the chain stops here")
        new = sha(out / "record.json")
        changed = replace_hashes(RUNS[i + 1:], table[run]["record_sha256"], new, done)
        for c in changed:
            only_hashes_differ(c, done)
        git("add", "--", str(out.relative_to(ROOT)), *[str(cond(c).relative_to(ROOT)) for c in changed])
        git("commit", "-q", "-m", f"rebuild: {run}'s record (same content as the original) and its SHA-256 in the "
                                  f"conditions of {', '.join(changed) or 'no later run'}")
        print(f"   {run}: committed; conditions changed: {', '.join(changed) or 'none'}", flush=True)
    print("all records rebuilt with the original content")


if __name__ == "__main__":
    main(sys.argv[1:])
