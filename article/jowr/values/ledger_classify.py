"""Step 2 of the disclosure ledger: sort the written leaves (ledger_inventory.json) into linear shares, each with the
per-writer quantity it sums ("variable") and the writers it sums over ("set"), and list the variables written on more
than one set.

Usage: python3 -I ledger_classify.py  ->  ledger_classes.json
- Persons: v6k and v6l count sample A; v6m-v6s count sample B (v6m's persons, overlap 0 with A; records' persons).
- Rows: in sample B, inside/before is inside_100 (v6n conditions, "rows": checked per person in the run).
- R, R_loc and R_mean exist on the same persons in every cell (v6n conditions, R_mean), so a reference's share on all
  writers with R (count_1, gate_0) is one set ("hasR").
- A stage's share on its count-1 persons, on its rule persons and on a group of them is the same per-writer verdict
  (conservative: if not, fewer sets share a variable).
- Not linear in the writers' verdicts (and so not in the ledger): interval ends, AUCs and their intervals, reliabilities,
  medians, counts of writers. Anything else unknown stops the run.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_numbers as pn  # noqa: E402

REFS = ("R", "R_loc", "R_mean", "R_lag")
NONLINEAR = re.compile(r"(_ci/\d+$|median|/one_minus_p_o/(min|max)$|^v6k/main_cells/rule \| .*/(stage|R|R_loc)/1/[01]$|_minus_[^/]*/[12]$|/ci/\d+$|rank_auc|/eer/|/rank_n/|/top1$|mean_rSB|mean_r$|rSB_percentile|/median$|"
                       r"/persons$|/persons/(\w+/)?(high|low|mid)$|/k/\d+$|starts_reaching|count_persons|left_out|ids_looked|tally_of_reading|"
                       r"covariate_median|base_by_group|_minus_|/difference$|lwp_minus_kupa|"
                       r"/common$|/_sha$)")


# the differences of shares the text writes, each checked by hand (record 20): two shares on the same writers, both
# written; the whole difference of two datasets, both written; differences of two datasets' bands (one equation in
# two unknown sums); a median of written agreements. Any other written difference stops the run (Codex 4).
DIFFERENCES = {
    "v6m/part_1/contrast/labelled/I/halves | essay/paired_minus_R_loc/difference",
    "v6q/main_cells/gate_cells/lwp_lwp oddeven inside_100/I[2]_minus_I/0",
    "v6q/main_cells/gate_cells/lwp_lwp halves inside_75/I[2]_minus_I/0",
    "v6q/main_cells/gate_cells/lwp_lwp halves inside_100/I[2]_minus_I/0",
    "v6s/summary/primary/one_minus_p_o/median",
    "v6o/bands/n_s_min/oddeven/cells/R_loc/inside_100/whole/difference",
    "v6o/bands/n_s_min/oddeven/cells/R_loc/inside_100/bands/high/lwp_minus_kupa",
    "v6o/bands/n_s_min/oddeven/cells/R_loc/cross_100/bands/high/lwp_minus_kupa",
}


def sample(rec):
    return "A" if rec in ("v6k", "v6l") else "B"


def row(s, r):
    return "inside_100" if s == "B" and r == "inside/before" else r


def classify(path, R):
    """(variable, set, n) for a linear share, None for a nonlinear leaf; stops on an unknown leaf."""
    p = path.split("/")
    rec, s = p[0], sample(p[0])
    if rec in ("v6k", "v6l", "v6m", "v6n") and p[1] == "rule_persons" and p[-1] == "share_vs_own":
        ds, base, st, x = p[2], p[3], p[-3], p[-2]
        r_ = "/".join(p[4:-3])
        cell = R[rec]["rule_persons"][ds][base][r_][st][x]
        var = (s, ds, base, row(s, r_), x if x in REFS else st)
        return var, (s, ds, base, row(s, r_), "rule", st), cell["persons"]
    if rec == "v6k" and p[1] == "main_cells" and p[2].startswith("rule | ") and p[-1] == "0" \
            and p[-2] in ("stage",) + REFS:  # [share, [lo, hi], class]: index 0 only (index 1 holds the interval)
        if p[2].count("|") != 4:
            sys.exit(f"unexpected v6k main cell: {path}")
        _, st, ds, base, r_ = [x.strip() for x in p[2].split(" | ")]
        r_ = "/".join([r_] + p[3:-2])
        x = p[-2]
        cell = R["v6k"]["rule_persons"][ds][base][r_][st]["stage" if x == "stage" else x]
        var = (s, ds, base, r_, x if x in REFS else st)
        return var, (s, ds, base, r_, "rule", st), cell["persons"]
    if p[1] == "count_1" and p[-1] == "share_vs_own":
        ds, base, x = p[2], p[3], p[-2]
        r_ = "/".join(p[4:-2])
        cell = R[rec]["count_1"][ds][base][r_][x]
        var = (s, ds, base, row(s, r_), x)
        st = ("hasR",) if x in REFS else ("has", x)
        return var, (s, ds, base, row(s, r_), "count1") + st, cell["persons"]
    if rec == "v6n" and p[2] == "gate_0" and p[-2] == "shares":
        ds, base, fam = [x.strip() for x in p[3].split(" | ")]
        x, dose = p[4], p[-1]
        r_ = f"{fam}_{dose}"
        n = R["v6n"]["count_1"][ds][base][r_][x]["persons"]
        return (s, ds, base, r_, x), (s, ds, base, r_, "count1", "hasR"), n
    if rec == "v6m" and p[2] == "contrast" and p[-1] == "share_vs_own":
        st, (base, task), x = p[4], p[5].split(" | "), p[6]
        cell = R["v6m"]["part_1"]["contrast"]["labelled"][st][p[5]][x]
        if x == "own_persons":
            return (s, "kupa", base, f"contrast_{task}", st), (s, "kupa", base, "contrast", "own", st), cell["persons"]
        return (s, "kupa", base, f"contrast_{task}", x if x in REFS else st), \
            (s, "kupa", base, "contrast", "rule", st), cell["persons"]
    if rec == "v6p" and p[1] == "groups" and p[-1] == "share_vs_own":
        if p[-3] != "groups":
            sys.exit(f"unexpected v6p path: {path}")
        cov, ds, base, st, r_, g = p[2], p[3], p[4], p[5], "/".join(p[6:-3]), p[-2]
        cell = R["v6p"]["groups"][cov][ds][base][st][r_]["groups"][g]
        r_ = row(s, r_)
        return (s, ds, base, r_, st), (s, ds, base, r_, "rule", st, cov, g), cell["persons"]
    if rec == "v6q" and p[1] == "cells" and p[-1] == "share_vs_own":
        ds, base, r_, st = p[2], p[3], p[4], p[6]
        cell = R["v6q"]["cells"][ds][base][r_]["rule"][st]
        return (s, ds, base, row(s, r_), st), (s, ds, base, row(s, r_), "rule", "I"), cell["persons"]
    if rec == "v6q" and p[2] == "primary_groups_n_s" and p[-1] == "0":
        ds, base, r_ = p[3].split()
        g, st = p[5], p[6]
        # the groups of I's rule writers by n_s, the same writers as v6p's (v6q conditions reuse v6p's groups)
        n = R["v6p"]["groups"]["n_s"][ds][base]["I"][r_]["groups"][g]["persons"]
        return (s, ds, base, r_, st), (s, ds, base, r_, "rule", "I", "n_s", g), n
    if rec == "v6o" and p[1] == "bands" and p[4] == "cells" and p[7] == "bands" and len(p) == 10 \
            and p[9] in ("kupa_essay", "lwp"):
        cov, base, ref, r_, g, dk = p[2], p[3], p[5], p[6], p[8], p[9]
        ds = {"kupa_essay": "kupa_essay", "lwp": "lwp_lwp"}[dk]
        n = R["v6o"]["bands"][cov][base]["persons"][dk][g]
        return (s, ds, base, r_, ref), (s, ds, base, r_, "count1", "hasR", cov, g), n
    if rec == "v6n" and p[1] == "part_2_signed_dose_rows" and p[-1].startswith("share_"):
        ds, r_, x, feat, sign = p[2], p[3], p[4], p[5], p[6]
        n = R["v6n"]["part_2_signed_dose_rows"][ds][r_][x][feat]["persons"]
        return (s, ds, "whole", r_, f"sign:{x}[{feat}]:{sign}"), (s, ds, "whole", r_, "signed", x), n
    if rec == "v6s" and p[-1] == "1-p_o":
        mc = R["v6s"]["main_cells"][p[2]]
        rest = "/".join(p[3:-1])
        hits = [c for c in mc if rest.startswith(c + "/")]
        if len(hits) != 1:
            sys.exit(f"v6s: cell of {path} not found once")
        cellname = hits[0]
        x = rest[len(cellname) + 1:]
        node = mc[cellname][x]
        return (s, cellname, "both", "agree", x), (s, cellname, "both", "common", x), node["common"]
    if re.search(r"_minus_|/difference$", path) \
            and not re.search(r"(_ci|/ci)/\d+$|_minus_[^/]*/[12]$|median|/one_minus_p_o/(min|max)$", path) \
            and path not in DIFFERENCES:
        sys.exit(f"a written difference not checked by hand (decision record 20): {path}")
    if NONLINEAR.search(path):
        return None
    sys.exit(f"unknown leaf: {path}")


def resolve(var, sets, R):
    """Why a variable written on more than one set gives no sum over 1-4 writers; stops if none of the reasons holds.
    'same writers': every set is a subset of one set (the writers with R of the cell, or the contrast's writers with
    R_loc) and has as many writers as it, so all are that set. 'partition': the whole cell and groups of it, where the
    groups (all of them, written or not) partition the whole and each has 5 writers or more."""
    s, ds, base, r_, x = var
    kinds = {st[4:] for st in sets}
    ns = {e["n"] for e in sets.values()}
    if x in REFS and ds != "kupa" and all(k == ("count1", "hasR") or (k[0] == "rule" and len(k) == 2)
                                          or (k[:2] == ("count1", "hasR") and len(k) == 4) for k in kinds):
        rec = "v6k" if s == "A" else ("v6n" if r_.startswith(("inside_", "cross_")) else "v6m")
        rr = "inside/before" if s == "B" and r_ == "inside_100" and rec == "v6m" else r_
        n_r = R[rec]["count_1"][ds][base][rr][x]["persons"]
        whole = {e["n"] for st, e in sets.items() if len(st) <= 6}
        if whole - {n_r}:
            sys.exit(f"{var}: writers {sorted(whole)} on its rule and R sets, {n_r} with R")
        bands = {st[6:]: e["n"] for st, e in sets.items() if len(st) == 8}
        if not bands:
            return "same writers (rule writers within the writers with R, as many)"
        covs = {c for c, _ in bands}
        if s != "B" or len(covs) != 1:
            sys.exit(f"{var}: bands {sorted(bands)}")
        bp = R["v6o"]["bands"][covs.pop()][base]["persons"]["kupa_essay" if ds == "kupa_essay" else "lwp"]
        if sum(bp.values()) != n_r or min(bp.values()) < 5 or any(bp[g] != n for (_, g), n in bands.items()):
            sys.exit(f"{var}: bands {bp} do not partition the {n_r} writers with R")
        return "same writers and partition (rule writers = writers with R; bands of them, 5 or more each)"
    if x in REFS and ds == "kupa" and all(st[3:5] == ("contrast", "rule") for st in sets):
        cp = R["v6m"]["part_1"]["contrast_persons"]
        task = r_.split("_")[1]
        own_u = R["v6m"]["part_1"]["contrast"]["labelled"]["U@2000"][f"{base} | {task}"]["own_persons"]["persons"]
        n_u = R["v6m"]["part_1"]["contrast"]["labelled"]["U@2000"][f"{base} | {task}"][x]["persons"]
        if own_u != cp or ns != {n_u}:
            sys.exit(f"{var}: U's own contrast writers {own_u} of {cp}, writers {sorted(ns)} and {n_u}")
        return "same writers (contrast rule writers of U are all contrast writers with R_loc, as many)"
    if x in ("I", "I[2]") and all(k[:2] == ("rule", "I") and (len(k) == 2 or k[2] == "n_s") for k in kinds):
        whole = {e["n"] for st, e in sets.items() if st[4:] == ("rule", "I")}
        rr = "inside/before" if r_ == "inside_100" and r_ not in R["v6p"]["groups"]["n_s"][ds][base]["I"] else r_
        gp = R["v6p"]["groups"]["n_s"][ds][base]["I"][rr]["groups"]
        gn = {g: gp[g]["persons"] for g in ("low", "mid", "high")}
        if len(whole) != 1 or sum(gn.values()) != next(iter(whole)) or min(gn.values()) < 5:
            sys.exit(f"{var}: groups {gn} do not partition the whole {whole}")
        for st, e in sets.items():
            if len(st) == 8 and e["n"] is not None and e["n"] != gn[st[7]]:
                sys.exit(f"{var}: group {st[7]} written with {e['n']} writers, the record has {gn[st[7]]}")
        return "partition (groups of the whole, 5 or more each)"
    sys.exit(f"{var}: written on {sorted(kinds)} with no reason that no small set is derivable")


def agreement_keys(R):
    """(set, row, stage) of every cell where v6s wrote an agreement between the splits (primary and shuffles), the row
    normalized by row() (inside/before = inside_100 in sample B)."""
    out = set()
    for part in ("primary", "shuffles"):
        for cellname, stages in R["v6s"]["main_cells"][part].items():
            ds, r_ = cellname.split(" ", 1)
            for st, v in stages.items():
                if isinstance(v, dict) and "1-p_o" in v:
                    out.add((ds, row("B", r_), st))
    return out


def subgroup_key(path):
    """(set, row, stage) of a written share over a group or band of a cell's writers (v6p, v6q, v6o), else None."""
    p = path.split("/")
    if p[0] == "v6p" and p[1] == "groups" and p[-1] == "share_vs_own" and p[-3] == "groups":
        return p[3], row("B", "/".join(p[6:-3])), p[5]
    if p[0] == "v6q" and p[2] == "primary_groups_n_s" and p[-1] == "0":
        ds, _, r_ = p[3].split()
        return ds, row("B", r_), p[6]
    if p[0] == "v6o" and p[1] == "bands" and len(p) == 10 and p[9] in ("kupa_essay", "lwp"):
        return {"kupa_essay": "kupa_essay", "lwp": "lwp_lwp"}[p[9]], row("B", p[6]), p[5]
    return None


def check_agreement_groups(inv, R):
    """Decision 25: where v6s wrote an agreement between the splits, no share over a group or band of that cell's
    writers is written for the same stage, from any record and under any name of the row (code-reviewer 1)."""
    keys = agreement_keys(R)
    for part in ("body", "appendix", "archive"):
        for unit, d in inv[part].items():
            for path in d["written"]:
                k = subgroup_key(path)
                if k is not None and k in keys:
                    sys.exit(f"{unit}: group share {path} written where v6s wrote the agreement of {k}")


def main():
    R = pn.load()
    inv = json.loads((HERE / "ledger_inventory.json").read_text(encoding="utf-8"))
    vars_ = defaultdict(lambda: defaultdict(lambda: {"n": None, "units": set(), "paths": set()}))
    nonlinear = 0
    for part in ("body", "appendix", "archive"):
        for unit, d in inv[part].items():
            for path, val in d["written"].items():
                if isinstance(val, int) and not isinstance(val, bool) and \
                        re.search(r"(share_vs_own|1-p_o|/shares/\d+|share_(positive|negative|zero))$", path):
                    sys.exit(f"a share written as an integer: {path}")  # code-reviewer 8
                if isinstance(val, int) and not isinstance(val, bool) or isinstance(val, str):
                    nonlinear += 1  # counts of writers, cells or events, and labels: not shares (out of the rule)
                    continue
                c = classify(path, R)
                if c is None:
                    nonlinear += 1
                    continue
                var, st, n = c
                e = vars_[var][st]
                if e["n"] is not None and n is not None and e["n"] != n:
                    sys.exit(f"{var} {st}: persons {e['n']} and {n}")
                e["n"] = n if n is not None else e["n"]
                e["units"].add(f"{part}:{unit}")
                e["paths"].add(path)
    multi = {" | ".join(map(str, v)): {" | ".join(map(str, st)): {"n": e["n"], "units": sorted(e["units"]),
                                                                    "paths": sorted(e["paths"])}
                                       for st, e in sets.items()}
             for v, sets in vars_.items() if len(sets) > 1}
    for v, sets in vars_.items():  # every share is on 5 writers or more, known (Codex 1)
        for st, e in sets.items():
            if not isinstance(e["n"], int) or isinstance(e["n"], bool) or e["n"] < 5:
                sys.exit(f"{v} on {st}: writers {e['n']!r}")
    why = {" | ".join(map(str, v)): resolve(v, sets, R) for v, sets in vars_.items() if len(sets) > 1}
    check_agreement_groups(inv, R)
    out = {"linear_variables": len(vars_), "linear_variable_sets": sum(len(s) for s in vars_.values()),
           "nonlinear_leaves": nonlinear, "variables_on_more_than_one_set": len(multi),
           "resolved_by": {k: sum(1 for w in why.values() if w == k) for k in sorted(set(why.values()))},
           "small_sets_derivable": 0, "multi": multi}
    (HERE / "ledger_classes.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "multi"}))


if __name__ == "__main__":
    main()
