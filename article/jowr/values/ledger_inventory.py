"""Inventory of the record values that the JoWR draft writes (step 1 of the disclosure ledger, decision 38 / 24).

Usage: python3 -I ledger_inventory.py  ->  ledger_inventory.json
- Wraps every record (v6k-v6s, and v6r for the appendix) in containers that log each leaf read, with its JSON path.
- Body: runs paper_numbers.computed() with its single "N = {}" replaced by a dict that cuts the log at each key
  assignment, so every key gets the leaves read since the previous key (over-inclusive: checks read for a key count).
  Figures 2 are their own units.
- Appendix: wraps the table functions that appendix_tables.main() calls; each gets the leaves read inside it.
- A leaf is "written" in a unit when one of its renderings (APA 2 or 3 decimals, "%.3f", or "%g" for integers) is a
  number token of the unit's output text. Coincidental equality only adds leaves (conservative for the ledger).
- Reads the records only; writes nothing but ledger_inventory.json.
"""
import json
import re
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_numbers as pn  # noqa: E402
import appendix_tables as at  # noqa: E402

LOG: list = []  # (path, value)


class TD(dict):
    def __init__(self, d, path):
        super().__init__({k: wrap(v, f"{path}/{k}") for k, v in d.items()})
        self._p = path

    def __getitem__(self, k):
        v = super().__getitem__(k)
        if not isinstance(v, (TD, TL)):
            LOG.append((f"{self._p}/{k}", v))
        return v

    def get(self, k, default=None):
        return self[k] if k in self else default

    def items(self):
        return [(k, self[k]) for k in super().keys()]

    def values(self):
        return [self[k] for k in super().keys()]


class TL(list):
    def __init__(self, xs, path):
        super().__init__([wrap(v, f"{path}/{i}") for i, v in enumerate(xs)])
        self._p = path

    def __getitem__(self, i):
        v = super().__getitem__(i)
        if isinstance(i, slice):
            for j, x in zip(range(*i.indices(len(self))), v):
                if not isinstance(x, (TD, TL)):
                    LOG.append((f"{self._p}/{j}", x))
            return v
        if not isinstance(v, (TD, TL)):
            LOG.append((f"{self._p}/{i % len(self)}", v))
        return v

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def wrap(v, path):
    if isinstance(v, dict) and not isinstance(v, TD):
        return TD(v, path)
    if isinstance(v, list) and not isinstance(v, TL):
        return TL(v, path)
    return v


TOK = re.compile(r"(?<![\w.])-?\d*\.\d+|(?<![\w.])\d+(?![\w.])")


def tokens(text):
    t = text.replace("--", " ").replace("$-$", "-").replace("\\%", "")  # a range's upper end is a token (code-reviewer 3)
    return set(TOK.findall(t))


def renderings(v):
    out = set()
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return out
    if isinstance(v, int):
        out.add(str(v))
    for places in (2, 3):
        try:  # per places: a stop at 2 places must not drop the 3-place rendering (code-reviewer 2)
            s = pn.fmt(v, places)
        except SystemExit:
            continue
        sign = "-" if v < 0 and s.strip(".0") else ""  # as pn.f2: a value rounding to 0 has no sign
        out.add(sign + (s[1:] if s.startswith("0") else s))
        out.add(sign + s)
    out.add(f"{v:.3f}")
    return out


def written(units):
    """A leaf counts as written in a unit that read it if one of its renderings is a number token of that unit's
    output. It also counts as written if a rendering is a token of another unit's output that no leaf read by that
    unit explains: a value read for one key and reused for another is caught (Codex 2), while a token that the other
    unit's own leaves give is a coincidence and is not counted (code-reviewer, round 2). Returns, per unit, the leaves
    written there, with the units where they were caught by the second clause."""
    toks = {name: tokens(text) for name, (text, _) in units.items()}
    explained = {name: set().union(*[renderings(v) for _, v in leaves]) if leaves else set()
                 for name, (_, leaves) in units.items()}
    unexplained = {name: toks[name] - explained[name] for name in units}
    res = {}
    for name, (text, leaves) in units.items():
        hit, elsewhere = {}, {}
        for path, v in leaves:
            r = renderings(v)
            if r & toks[name]:
                hit[path] = v
                continue
            where = sorted(u for u in units if u != name and r & unexplained[u])
            if where:
                hit[path] = v
                elsewhere[path] = where
        res[name] = {"text": text if len(text) < 400 else text[:400] + "...", "n_leaves_read": len(leaves),
                     "written": hit, "caught_elsewhere": elsewhere}
    return res, unexplained


class TrackN(dict):
    def __init__(self):
        super().__init__()
        self.units = {}
        self.mark = 0

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        self.units[k] = (str(v), LOG[self.mark:])
        self.mark = len(LOG)


def body(R):
    src = (HERE / "paper_numbers.py").read_text(encoding="utf-8")
    if src.count("    N = {}\n") != 1:
        sys.exit("paper_numbers.py: expected exactly one 'N = {}' line")
    src = src.replace("    N = {}\n", "    N = _TrackN()\n").replace('if __name__ == "__main__":', "if False:")
    mod = types.ModuleType("pn_tracked")
    mod.__dict__["_TrackN"] = TrackN
    mod.__dict__["__file__"] = str(HERE / "paper_numbers.py")
    exec(compile(src, "paper_numbers_tracked", "exec"), mod.__dict__)
    LOG.clear()
    N = mod.computed(R)
    if dict(N) != json.loads((HERE / "numbers.json").read_text(encoding="utf-8")):
        sys.exit("tracked body numbers differ from numbers.json")
    units = dict(N.units)
    for base, name in (("oddeven", "fig2"), ("halves", "fig2-halves")):
        m = len(LOG)
        text = mod.figure2(R, base)
        if text != (HERE / f"{name}.tex").read_text(encoding="utf-8"):
            sys.exit(f"{name}: differs from the file")
        units[name] = (text, LOG[m:])
    return units


TOPS = ("runs", "predictions", "rlag", "thresholds", "identification", "full_cells", "reliability", "split_dependence",
        "contrast", "doses", "moved", "border", "groups", "margins", "complete", "flow", "writer_threshold",
        "lwp_reading")


def appendix(R):
    units, stack = {}, []

    def wrapper(name, f):
        def g(*a, **kw):
            m = len(LOG)
            stack.append(name)
            try:
                out = f(*a, **kw)
            finally:
                stack.pop()
            key = name if name != "full_cells" else f"full_cells:{a[1]}"
            if stack:
                return out  # nested: the outer unit gets the leaves
            units[key] = (out if isinstance(out, str) else json.dumps(out, default=str), LOG[m:])
            return out
        return g

    for name in TOPS:
        setattr(at, name, wrapper(name, getattr(at, name)))
    for v in R:
        R[v]["_sha"] = pn.SHA[v]
    R["v6r"] = wrap(json.loads((pn.RES / "prototype_v6r" / "record.json").read_text(encoding="utf-8")), "v6r")
    R["v6r"]["_sha"] = "d21b309a60e92b3e1af648049c7719468cf2fc3d23d5ab2773e7056117b1bfdc"
    C = {v: json.loads((pn.ROOT / "works" / "conditions" / f"prototype_{v}.json").read_text(encoding="utf-8"))
         for v in ("v6k", "v6l", "v6m", "v6n", "v6o", "v6p", "v6q", "v6r", "v6s")}
    files = {"runs": "app-runs.tex", "rlag": "app-rlag.tex", "thresholds": "app-thr.tex",
             "identification": "app-id.tex", "reliability": "app-rel.tex", "split_dependence": "app-split.tex",
             "contrast": "app-contrast.tex", "border": "app-border.tex", "groups": "app-groups.tex",
             "margins": "app-margin.tex", "complete": "app-complete.tex", "flow": "app-flow.tex"}
    for name, fn in files.items():
        out = getattr(at, name)(R)
        if out != (HERE / fn).read_text(encoding="utf-8"):
            sys.exit(f"{fn}: differs from the file")
    if at.predictions(R, C) != (HERE / "app-pred.tex").read_text(encoding="utf-8"):
        sys.exit("app-pred.tex differs")
    cells = at.full_cells(R, "v6k", "Rule count: every cell", "tab:cellsk") + \
        at.full_cells(R, "v6m", "Second count: every cell", "tab:cellsm")
    if cells != (HERE / "app-cells.tex").read_text(encoding="utf-8"):
        sys.exit("app-cells.tex differs")
    if at.doses(R) + at.moved(R) != (HERE / "app-dose.tex").read_text(encoding="utf-8"):
        sys.exit("app-dose.tex differs")
    at.writer_threshold(R)
    wthr = (HERE / "app-wthr.tex").read_text(encoding="utf-8")
    if at.lwp_reading(R) not in wthr:
        sys.exit("app-wthr.tex lacks lwp_reading")
    # every app-*.tex the build reads is one of the files checked here (a new table function stops the run: Codex 6)
    checked = set(files.values()) | {"app-pred.tex", "app-cells.tex", "app-dose.tex", "app-wthr.tex", "app-ledger.tex"}
    present = {f.name for f in HERE.glob("app-*.tex")}
    if present != checked:
        sys.exit(f"app files not covered by the inventory: {sorted(present - checked)}; missing: {sorted(checked - present)}")
    t, leaves = units["writer_threshold"]
    units["writer_threshold"] = (wthr, leaves)
    return units


# The archive also holds the conditions of v6l-v6s, which quote earlier records as the reasons for each run (decision
# 28). Each quotation below is recomputed from the record leaves it summarizes
# and must appear verbatim in the conditions file; its leaves count as written in an "archive" unit.
# (run, field, text as the conditions write it, leaves read, how they are written)
def _rel(R, rec, ds, feat, key="mean_rSB"):
    if rec == "v6l":
        return R["v6l"]["reliability"][ds]["features"][feat][key]
    return R["v6n"]["part_3"]["reliability"][ds]["features"][feat][key]


QUOTATIONS: dict = {}  # unit name -> (run, field, the conditions' own words), filled by archive_conditions


def archive_conditions(R):
    C = {v: json.loads((pn.ROOT / "works" / "conditions" / f"prototype_{v}.json").read_text(encoding="utf-8"))
         for v in ("v6l", "v6m", "v6n", "v6o", "v6p", "v6q", "v6r", "v6s")}
    units = {}

    def unit(name, run, field, quoted, text, leaves_from):
        QUOTATIONS[name] = (run, field, quoted)
        if quoted not in C[run][field]:
            sys.exit(f"archive {name}: '{quoted}' not in {run} conditions '{field}'")
        m = len(LOG)
        got = leaves_from()
        if got != text:
            sys.exit(f"archive {name}: recomputed '{got}', the conditions write '{quoted}'")
        units[name] = (text, LOG[m:])

    # v6l purpose: odd/even parts cut 44-92% of bursts at their edges (v6k, median per set and part)
    unit("conditions:v6l", "v6l", "purpose", "cut 44-92% of bursts", ".44--.92",
         lambda: pn.rng([R["v6k"]["writer"][ds]["edge_share_of_bursts_2000_text"]["oddeven"][pt]["median"]
                         for ds in ("kupa_copy", "kupa_essay", "lwp_lwp") for pt in "AB"]))
    # v6m purpose: the two IQRs of U of low split-half reliability in v6l (mean rSB below 0.50)
    unit("conditions:v6m", "v6m", "purpose", "low split-half reliability in v6l (0.16-0.37)", ".16--.37",
         lambda: pn.rng([x for x in (_rel(R, "v6l", ds, f) for ds in ("kupa_copy", "kupa_essay", "lwp_lwp")
                                     for f in ("U@2000[1]", "U@2000[3]")) if x < 0.50]))
    # v6o purpose: KUPA essay inside_100 moves a median of 11 sentence-initial gaps; R_loc's sentence-start number
    # (R[0]) has a split-half reliability of .54 (both also written in the body: nSwap100K, nRlocsentK)
    unit("conditions:v6o", "v6o", "purpose", "split-half reliability of .54", ".54",
         lambda: pn.f2(_rel(R, "v6n", "kupa_essay", "R[0]")))
    # v6q fixed_before_running: I[2] rose in .81-.97 of persons under inside and .94-.995 under cross (v6n part 2)
    sd = R["v6n"]["part_2_signed_dose_rows"]

    def rose():
        xs = {fam: [sd[ds][f"{fam}_{d}"]["I"]["2"]["share_positive"] for ds in ("kupa_essay", "lwp_lwp")
                    for d in (25, 50, 75, 100)] for fam in ("inside", "cross")}
        # the conditions write the cross range's upper end with three places
        return f"{pn.rng(xs['inside'])}; {pn.f2(min(xs['cross']))}--{pn.f2(max(xs['cross']), places=3)}"
    unit("conditions:v6q", "v6q", "fixed_before_running", "I[2] rose in .81-.97 of persons under inside and .94-.995",
         ".81--.97; .94--.995", rose)
    # v6q limits_written_before: I[2]'s part-to-part reliability (v6n part 3, mean rSB: KUPA essay .45, LWP .59)
    unit("conditions:v6q-rel", "v6q", "limits_written_before", "KUPA essay .45, LWP .59", ".45, .59",
         lambda: ", ".join(pn.f2(_rel(R, "v6n", ds, "I[2]")) for ds in ("kupa_essay", "lwp_lwp")))
    # every other decimal or percentage in these conditions is a parameter or an example in a reading rule, written
    # before the values; a quotation is allowed only in its own run's conditions and as often as it is quoted there.
    # Strings are read one by one (json.dumps would write a newline as "\\n" and hide the number after it: code-reviewer)
    params = {".1", "0.1", "0.15", "0.20", "0.25", "0.5", "0.50", "0.60", "0.75", "0.80", "1.00", "2.5", "97.5",
              "83%", "97%", "75%", "95%", "25%", "50%", "100%", "2.5%", "97.5%"}
    quoted = {"v6l": {"44-92%": 1}, "v6m": {"0.16": 1, "0.37": 1}, "v6o": {".54": 1},
              "v6q": {".81": 1, ".97": 1, ".94": 1, ".995": 1, ".45": 1, ".59": 1}}
    num = re.compile(r"(?<![\w.])\d*\.\d+(?![\d.])%?|(?<![\w.])\d+(?:-\d+)?%")  # (?![\d.]): not version strings

    def strings(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield k
                yield from strings(v)
        elif isinstance(o, list):
            for v in o:
                yield from strings(v)
        elif isinstance(o, str):
            yield o

    for v, c in C.items():
        found: dict = {}
        for s in strings(c):
            for tok in num.findall(s):
                found[tok] = found.get(tok, 0) + 1
        new = sorted(t for t in found if t not in params and t not in quoted.get(v, {}))
        if new:
            sys.exit(f"{v} conditions: decimals or percentages neither quoted nor known as parameters: {new}")
        extra = {t: n for t, n in found.items() if t in quoted.get(v, {}) and t not in params
                 and n != quoted[v][t]}
        if extra:
            sys.exit(f"{v} conditions: a quotation appears a different number of times than covered: {extra}")
    return units


# decimals typed by hand in main.tex and appendix.tex (outside comments and \\N{}): thresholds, chance, margins and
# seconds, none a share (code-reviewer round 2, MEDIUM 5). A new one stops the run.
HAND_TYPED = {".005", ".50", ".55", ".60", ".65", "1.18", "4.0", "9.7"}


def hand_typed():
    if (HERE.parent.name, HERE.name) == ("jowr", "values"):
        # the archive (article/jowr/values/) holds the article's outputs but not its source: this check needs it
        print("in the archive: the check of hand-typed decimals in the manuscript's source is skipped")
        return
    found = set()
    for name in ("main.tex", "appendix.tex"):
        for ln in (HERE / name).read_text(encoding="utf-8").splitlines():
            ln = re.sub(r"(?<!\\)%.*", "", ln)
            ln = re.sub(r"\\N\{[^}]*\}", " ", ln)
            found |= {t for t in tokens(ln) if "." in t}
    if found - HAND_TYPED:
        sys.exit(f"decimals typed by hand, not in HAND_TYPED: {sorted(found - HAND_TYPED)}")


def main():
    R = {v: wrap(r, v) for v, r in pn.load().items()}
    b, a, c = body(R), appendix(R), archive_conditions(R)
    units = {**{f"body:{k}": v for k, v in b.items()}, **{f"appendix:{k}": v for k, v in a.items()},
             **{f"archive:{k}": v for k, v in c.items()}}
    w, unexplained = written(units)
    res = {"body": {k[5:]: v for k, v in w.items() if k.startswith("body:")},
           "appendix": {k[9:]: v for k, v in w.items() if k.startswith("appendix:")},
           "archive": {k[8:]: v for k, v in w.items() if k.startswith("archive:")}}
    res["_unexplained_tokens"] = {k: sorted(v) for k, v in unexplained.items() if v}
    # reverse check (code-reviewer round 2, MEDIUM 4): a decimal no leaf read by its unit explains must be known
    known = json.loads((HERE / "ledger_known_unexplained.json").read_text(encoding="utf-8"))["known"]
    new_t = {k: sorted(t for t in v if "." in t and t not in known.get(k, [])) for k, v in unexplained.items()}
    new_t = {k: v for k, v in new_t.items() if v}
    if new_t:
        sys.exit(f"decimals no read leaf explains, not in ledger_known_unexplained.json: {new_t}")
    hand_typed()
    allw = {p for part in ("body", "appendix", "archive") for u in res[part].values() for p in u["written"]}
    res["_summary"] = {"units_body": len(res["body"]), "units_appendix": len(res["appendix"]),
                       "units_archive": len(res["archive"]), "distinct_leaves_written": len(allw)}
    (HERE / "ledger_inventory.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(res["_summary"]))


if __name__ == "__main__":
    main()
