"""The table of the article's values for the archive: per output, the string the article prints and the record cells
read to make it, without the record values (decision 24 and the hand-over of 2026-11-09).

Usage: python3 -I archive_sources.py  ->  sources.json
- Runs the same tracked units as ledger_inventory.py (body keys, Figure 2, appendix tables, the conditions' quotations)
  and writes, per unit, its output (the body key's string, the table's file name, or the quotation) and the sorted
  record paths it read. Record values are never written: an output is the article's own string, and a path is a
  record's key path.
- Stops if a written path is not among the paths read (the ledger and this table disagree), or if a record path holds
  a decimal (a value leaking into a path). The conditions' quotations are written as the conditions write them.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ledger_inventory as li  # noqa: E402

APP_FILES = {"runs": "app-runs.tex", "rlag": "app-rlag.tex", "thresholds": "app-thr.tex",
             "identification": "app-id.tex", "reliability": "app-rel.tex", "split_dependence": "app-split.tex",
             "contrast": "app-contrast.tex", "border": "app-border.tex", "groups": "app-groups.tex",
             "margins": "app-margin.tex", "complete": "app-complete.tex", "flow": "app-flow.tex",
             "predictions": "app-pred.tex", "full_cells:v6k": "app-cells.tex", "full_cells:v6m": "app-cells.tex",
             "doses": "app-dose.tex", "moved": "app-dose.tex", "writer_threshold": "app-wthr.tex",
             "lwp_reading": "app-wthr.tex"}


def main():
    R = {v: li.wrap(r, v) for v, r in li.pn.load().items()}
    b, a, c = li.body(R), li.appendix(R), li.archive_conditions(R)
    inv = json.loads((HERE / "ledger_inventory.json").read_text(encoding="utf-8"))
    numbers = json.loads((HERE / "numbers.json").read_text(encoding="utf-8"))
    out = {"note": "Per output of the article: the string it prints and the record cells (key paths in the records "
                   "results/prototype_<run>/record.json, rebuilt by running the run) read to make it. Values are "
                   "only the article's own strings. Over-inclusive: a key lists every cell read since the previous "
                   "key, checks included.",
           "body": {}, "figures": {}, "appendix": {}, "conditions": {}}
    for k, (text, leaves) in b.items():
        paths = sorted({p for p, _ in leaves})
        if k in ("fig2", "fig2-halves"):
            out["figures"][k] = {"file": f"{k}.tex", "records_read": paths}
        else:
            if numbers.get(k) != text:
                sys.exit(f"body key {k}: tracked string differs from numbers.json")
            out["body"][k] = {"text": text, "records_read": paths}
    for k, (_, leaves) in a.items():
        if k not in APP_FILES:
            sys.exit(f"appendix unit {k} has no file in APP_FILES")
        out["appendix"][k] = {"file": APP_FILES[k], "records_read": sorted({p for p, _ in leaves})}
    for k, (text, leaves) in c.items():
        run, field, quoted = li.QUOTATIONS[k]
        out["conditions"][k.split(":", 1)[1]] = {"run": run, "field": field, "text": quoted,
                                                 "records_read": sorted({p for p, _ in leaves})}
    # the ledger's written leaves are among the paths read here
    for part, mine in (("body", {**out["body"], **out["figures"]}), ("appendix", out["appendix"]),
                       ("archive", {f"conditions:{k}": v for k, v in out["conditions"].items()})):
        for unit, d in inv[part].items():
            if unit not in mine:
                sys.exit(f"ledger unit {part}:{unit} missing here")
            missing = set(d["written"]) - set(mine[unit]["records_read"])
            if missing:
                sys.exit(f"{part}:{unit}: written paths not read here: {sorted(missing)[:3]}")
    # no record value in a path: a path's number tokens are indices, years, thresholds or doses, never a decimal
    for part in ("body", "figures", "appendix", "conditions"):
        for unit, d in out[part].items():
            for p in d["records_read"]:
                if any("." in t for t in li.tokens(p.replace("@", " "))):
                    sys.exit(f"{part}:{unit}: a decimal in a record path: {p}")
    text = json.dumps(out, indent=1, ensure_ascii=False) + "\n"
    (HERE / "sources.json").write_text(text, encoding="utf-8")
    n = sum(len(out[p]) for p in ("body", "figures", "appendix", "conditions"))
    print(json.dumps({"units": n, "distinct_paths": len({x for p in ("body", "figures", "appendix", "conditions")
                                                         for d in out[p].values() for x in d["records_read"]})}))


if __name__ == "__main__":
    main()
