"""Write the tables of Appendices A, C, D, E and G from the records of v6k-v6s (app-*.tex, read by appendix.tex).

Usage: python3 -I appendix_tables.py (build.py runs it after paper_numbers.py)
- Records are read through paper_numbers.load() (SHA-256 checked); nothing in results/ is changed.
- Disclosure (decision 19): each quantity is written on one set of writers per
  cell. A summary is written on its own rule writers; a reference is written on one set only (R and R_loc on U's rule
  writers in the rule and second counts, as in the body; R_loc and R_mean on the writers with R in the dose tables).
  Where a reference is read on another summary's writers, only its class is written. Paired differences are written
  as the class of their interval (above 0, across 0, below 0), since their value would give the reference's share on
  the summary's writers. Families the records themselves withheld (v6o-v6s) are not written. The ledger over the body,
  the appendices and the archive is built before submission (Appendix A says so).
- Every class written is recomputed from the interval and must agree with the class the record gives; a value the body
  also gives is spot-checked against numbers.json (B's rank AUC and the swaps at LWP inside 75%); the body's other keys
  are checked in paper_numbers.py.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # run with -I: the script's own directory is not on the path
import paper_numbers as pn  # noqa: E402
SETS = pn.SETS
BASES = pn.BASES
SETNAME = {"kupa_essay": "KUPA essays", "lwp_lwp": "LWP"}
BASENAME = {"oddeven": "odd--even", "halves": "first--second half"}
ROWNAME = {"inside/before": "inside", "shuffled/before": "shuffled", "shuffled_in_halves/before": "shuffled in halves"}
STAGENAME = {"U@2000": "$U$", "B@2000": "$B$", "Bsd@2000": "$B_{sd}$", "I": "$I$", "B_complete@2000": "$B$ complete",
             "Bsd_complete@2000": "$B_{sd}$ complete", "R": "$R$", "R_loc": "$R_{loc}$", "R_mean": "$R_{mean}$"}
LETTER = {"above": "a", "below": "b", "undecided": "u"}


FIXED = {"U@2000": lambda b, r: r == "inside/before",
         "B@2000": lambda b, r: r == "inside/before" or (b == "halves" and r == "shuffled_in_halves/before")}


def fixed(stage, b, r):
    """Cells fixed by construction (v6k conditions: U inside; B inside and shuffled in halves under halves)."""
    return stage in FIXED and FIXED[stage](b, r)


def tex(s):
    """Escape a record string for LaTeX text."""
    out = []
    for ch in str(s):
        out.append({"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "'": r"\textquotesingle{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
                    ">": r"\textgreater{}", "<": r"\textless{}"}.get(ch, ch))
    return "".join(out)


def klass(interval, margin=0.6, what="cell"):
    """The conditions' reading: above if the lower end exceeds the margin, below if the upper end lies under it,
    otherwise undecided (an end equal to the margin is undecided); cls() checks it against the record's class."""
    lo, hi = pn.ci(interval, what)
    if not 0 <= lo <= hi <= 1:
        sys.exit(f"{what}: interval [{lo}, {hi}] is not within [0, 1]")
    return "above" if lo > margin else ("below" if hi < margin else "undecided")


def reading(c):
    """The record's own class of a cell (reading.size or class), as above/below/undecided."""
    if "reading" in c:
        return {"above 0.60": "above", "below 0.60": "below", "undecided": "undecided"}[c["reading"]["size"]]
    return c["class"]


CLASSES = ("above", "below", "undecided")


def rec_class(x, what):
    """A class the record writes without an interval: it must be one of the three."""
    if x not in CLASSES:
        sys.exit(f"{what}: unknown class {x!r}")
    return x


def cls(c, what):
    mine = klass(c["share_vs_own_ci"], what=what)
    if mine != reading(c):
        sys.exit(f"{what}: class {mine} from the interval differs from the record's {reading(c)}")
    return mine


def cell(c, what, n=True):
    """'.72 a (177)': share, class letter, writers."""
    k = cls(c, what)
    s = f"{pn.f2(c['share_vs_own'])}\\,{LETTER[k]}"
    if n:
        p = c.get("persons")
        if isinstance(p, bool) or not isinstance(p, int) or p < 5:
            sys.exit(f"{what}: no valid count of writers")
        s += f" ({p})"
    return s


def sign_class(interval, what):
    """Class of a paired interval against 0 (the record writes no class for it; the body's count uses the same rule)."""
    lo, hi = pn.ci(interval, what)
    return "$>0$" if lo > 0 else ("$<0$" if hi < 0 else "$\\ni 0$")


def rowname(r):
    """'inside_75' -> 'inside 75\\%'; 'shuffled/before' -> 'shuffled'."""
    if r in ROWNAME:
        return ROWNAME[r]
    fam, _, dose = r.partition("_")
    if fam in ("inside", "cross") and dose.isdigit():
        return f"{fam} {dose}\\%"
    if r in ("shuffled", "shuffled_in_halves"):
        return ROWNAME[r + "/before"]
    sys.exit(f"unknown row name {r!r}")


def table(caption, label, cols, head, rows, note=None, size="\\footnotesize"):
    out = ["\\begin{table}[htbp]", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size,
           f"\\begin{{tabularx}}{{\\linewidth}}{{@{{}}{cols}@{{}}}}", "\\toprule", " & ".join(head) + " \\\\",
           "\\midrule"]
    out += [" & ".join(r) + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabularx}"]
    if note:
        out.append(f"\\par\\smallskip\\footnotesize Note. {note}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


def longtable(head, rows, cols, note=None):
    out = ["{\\footnotesize", f"\\begin{{longtable}}{{@{{}}{cols}@{{}}}}", "\\toprule", " & ".join(head) + " \\\\",
           "\\midrule", "\\endhead"]
    out += [" & ".join(r) + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{longtable}"]
    if note:
        out.append(f"\\par\\noindent Note. {note}\\par")
    out.append("}")
    return "\n".join(out) + "\n"


# ---------- Appendix A ----------
def runs(R):
    rows = []
    status = {"v6k": "rule count (part 1); identification and thresholds, descriptions",
              "v6l": "descriptions (same writers as v6k)",
              "v6m": "second count (part 0); not evaluable (part 1); descriptions (parts 2--4)",
              "v6n": "descriptions", "v6o": "descriptions", "v6p": "descriptions", "v6q": "descriptions",
              "v6r": "descriptions (nothing written)", "v6s": "descriptions"}
    for v in ("v6k", "v6l", "v6m", "v6n", "v6o", "v6p", "v6q", "v6r", "v6s"):
        rec = R[v]
        pv = rec["provenance"]
        if pv.get("git_dirty") is not False:
            sys.exit(f"{v}: the run was not from a clean checkout")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T.*", pv["run_at"]) or not re.fullmatch(r"[0-9a-f]{40}", pv["git_commit"]) \
                or not re.fullmatch(r"[0-9a-f]{64}", pv["conditions_sha256"]) or not re.fullmatch(r"[0-9a-f]{64}", rec["_sha"]):
            sys.exit(f"{v}: provenance not in the expected form")
        rows.append([v, pv["run_at"][:10], f"\\texttt{{{pv['git_commit'][:7]}}}", f"\\texttt{{{pv['conditions_sha256'][:8]}}}",
                     f"\\texttt{{{rec['_sha'][:8]}}}", status[v]])
    return table("Runs, commits and statuses", "tab:runs", "lllllX",
                 ["Run", "Run at", "Commit", "Conditions", "Record", "Status"], rows,
                 note="Commit: the commit the code ran from; conditions and record: the first characters of their "
                      "SHA-256. The conditions of each run were committed before the run; commit times are not a "
                      "registration a third party can verify.")


def predictions(R, C):
    rows = []
    for v in ("v6k", "v6l", "v6m", "v6n", "v6o", "v6p", "v6q", "v6r", "v6s"):
        text = C[v].get("prediction") or C[v].get("predictions")
        if not isinstance(text, str):
            sys.exit(f"{v}: the conditions have no prediction text")
        out = R[v]["predictions"]
        if v == "v6s":
            out = {k_: x for k_, x in out.items()}
        rows.append([f"\\multicolumn{{2}}{{@{{}}p{{0.97\\linewidth}}@{{}}}}{{\\textit{{{v}}}: {tex(text)}}}"])
        for key in sorted(out, key=lambda x: int(x[1:])):
            o = out[key]
            o = {True: "held", False: "did not hold"}.get(o, o) if isinstance(o, bool) else o
            rows.append([key, tex(o)])
    return longtable(["", "Outcome (from the record)"], rows, "lp{0.86\\linewidth}",
                     note="Each run's predictions verbatim from its conditions (italic), written before the run, then "
                          "each outcome as the record gives it. Names in the conditions: persons = writers; "
                          "stages = summaries; smallest difference = margin; baselines oddeven and halves = the two "
                          "splits; U@2000 = $U$ at 2~s; I[2] = the between-word mean; n\\_s = sentence starts per "
                          "part; S1 = the earlier sentence summary with its external scale; theta\\_w = the "
                          "writer-specific threshold.")


def rlag(R):
    g = R["v6l"]["rule"]["part_1_by_gate"]
    rows = []
    for s in ("U@2000", "B@2000", "Bsd@2000", "I"):
        labels = [g[gt][s]["label"] for gt in ("R", "R_loc", "R_lag")]
        if not all(isinstance(x, str) for x in labels):
            sys.exit(f"{s}: a label is not a string")
        rows.append([STAGENAME[s]] + [tex(x) for x in labels])
    # the gate R gives v6k's labels (v6l notes); check them against v6k's record
    for s in ("U@2000", "B@2000", "Bsd@2000", "I"):
        if g["R"][s]["label"] != R["v6k"]["main_cells"][f"rule | {s}"]["label"]:
            sys.exit(f"{s}: v6l's label under gate R differs from v6k's")
    return table("Labels of the rule count under three gates (the rule count's writers; descriptions)", "tab:rlag",
                 "lXXX", ["Summary", "Gate $R$", "Gate $R_{loc}$", "Gate $R_{lag}$"], rows,
                 note="$R_{lag}$ is the reference's lag-1 correlation alone; it appears only here. Labels as the "
                      "record writes them (``shows these operations beyond the smallest difference'' = shows).")


# ---------- Appendix C ----------
def thresholds(R):
    k = R["v6k"]
    rows = []
    for st in SETS:
        for b in BASES:
            for r in ("inside/before",) + pn.SHUFFLES:
                c_ = f"{st} | {b} | {r}"
                row = [SETNAME[st], BASENAME[b], ROWNAME[r]]
                for stg, kst in (("U", "U@2000"), ("B", "B@2000"), ("Bsd", "Bsd@2000")):
                    c2000 = cls(k["rule_persons"][st][b][r][kst]["stage"], f"{kst} {c_}")
                    marks = [LETTER[c2000]] + [LETTER[rec_class(k["rule"]["part_2"][stg][t]["classes"][c_], f"{stg} {t} {c_}")]
                                               for t in ("1000", "500", "writer", "writer_fixed")]
                    row.append(" ".join(marks))
                rows.append(row)
    # the classes at the other thresholds have no interval in the record: check them against the record's own count of
    # cells equal to 2 s, recounted here from the 2-s classes recomputed from their intervals
    for stg, kst in (("U", "U@2000"), ("B", "B@2000"), ("Bsd", "Bsd@2000")):
        for t_ in ("1000", "500", "writer", "writer_fixed"):
            same = sum(rec_class(k["rule"]["part_2"][stg][t_]["classes"][f"{st} | {b} | {r}"], stg) ==
                       cls(k["rule_persons"][st][b][r][kst]["stage"], f"{kst} {st} {b} {r}")
                       for st in SETS for b in BASES for r in ("inside/before",) + pn.SHUFFLES)
            if same != k["rule"]["part_2"][stg][t_]["same_as_2000"]:
                sys.exit(f"{stg} {t_}: {same} cells equal to 2 s, the record says {k['rule']['part_2'][stg][t_]['same_as_2000']}")
    return table("Classes at the 2-s, 1-s and 500-ms thresholds and at the writer-specific threshold (rule count's writers)",
                 "tab:thr", "lllXXX", ["Data", "Split", "Operation", "$U$", "$B$", "$B_{sd}$"], rows,
                 note="Per cell the classes at 2~s, 1~s, 500~ms, the writer-specific threshold refitted on each "
                      "record, and the writer-specific threshold kept from the original record "
                      "(a above .60, b below, u undecided). The 2-s class is recomputed from its interval; the "
                      "others are the record's classes (it gives no interval for them), checked against its count of "
                      "cells equal to 2~s. Cells fixed by construction ($U$ and $B$ under inside, "
                      "$B$ under shuffled in halves with the first--second half split) are b at the fixed thresholds; "
                      "a writer-specific threshold refitted on each record can move them. "
                      "Only the classes are written.")


def writer_threshold(R):
    """Median writer-specific threshold; and, per summary, cells whose class kept from the original record equals the
    class at 2 s, and LWP reshuffle cells above .60 with the threshold refitted on each record."""
    k = R["v6k"]
    w = k["writer"]
    out = {"fixed_same": [], "refit_above": []}
    for stg, kst in (("U", "U@2000"), ("B", "B@2000"), ("Bsd", "Bsd@2000")):
        same, above, movable = 0, 0, 0
        for st in SETS:
            for b in BASES:
                for r in ("inside/before",) + pn.SHUFFLES:
                    c_ = f"{st} | {b} | {r}"
                    c2000 = cls(k["rule_persons"][st][b][r][kst]["stage"], f"{kst} {c_}")
                    if not fixed(kst, b, r):  # cells that can change only (as kThrmov in the body)
                        movable += 1
                        same += rec_class(k["rule"]["part_2"][stg]["writer_fixed"]["classes"][c_], c_) == c2000
                    if st == "lwp_lwp" and r in pn.SHUFFLES:
                        above += rec_class(k["rule"]["part_2"][stg]["writer"]["classes"][c_], c_) == "above"
        out["fixed_same"].append(f"{same} of {movable}")
        out["refit_above"].append(above)
    for st in ("kupa_essay", "lwp_lwp"):
        t_ = w[st]["theta_w_text"]["theta_ms"]
        out[st] = (pn.fin(t_["median"]), pn.fin(t_["p25"]), pn.fin(t_["p75"]))
        moved = w[st]["abs_ln_theta_w_row_minus_text"]
        for row in ("inside/before", "shuffled/before"):
            if pn.fin(moved[row]["median"]) <= 0:
                sys.exit(f"{st} {row}: the writer threshold does not move with the operation")
    return out


# ---------- Appendix D ----------
def identification(R):
    c2 = R["v6k"]["count_2"]
    rows = []
    order = ["S0", "R", "S1", "U@2000", "U@1000", "U@500", "U@writer", "B@2000", "B@1000", "B@500", "B@writer",
             "Bsd@2000", "Bsd@1000", "Bsd@500", "Bsd@writer", "I"]
    name = {"S0": "$S_0$", "R": "$R$", "S1": "$S_1$", "I": "$I$"}
    for s in order:
        m = c2["kupa_copy_to_essay"][s]["main"]
        reg = c2["kupa_copy_to_essay"][s]["registration_scale"]
        rev = c2["kupa_essay_to_copy"][s]["main"]
        for x in (m, reg, rev):
            lo, hi = pn.ci(x["rank_auc_ci"], s)
            if not lo <= pn.fin(x["rank_auc"]) <= hi:
                sys.exit(f"{s}: rank AUC outside its interval")
        def auc(x):
            lo, hi = pn.ci(x["rank_auc_ci"])
            return f"{pn.f2(x['rank_auc'])} [{pn.f2(lo)}, {pn.f2(hi)}]"
        rn = m["rank_n"]
        if s in name:
            nm = name[s]
        else:
            base, thr = s.split("@")
            nm = {"U": "$U$", "B": "$B$", "Bsd": "$B_{sd}$"}[base] + " " + {"2000": "2~s", "1000": "1~s", "500": "500~ms",
                                                                           "writer": "writer"}[thr]
        rows.append([nm, auc(m)] + [pn.f2(rn[r_]["rate"], places=3) for r_ in ("1", "5", "10", "20")]
                    + [pn.f2(m["eer"]["eer"], places=3), auc(reg), auc(rev)])  # rates and EER over 200 writers: three decimals
    chance = c2["kupa_copy_to_essay"]["S0"]["main"]["rank_n"]
    ch = ", ".join(pn.f2(chance[r_]["chance"], places=3) for r_ in ("1", "5", "10", "20"))
    return table("Matching writers across tasks: all summaries and thresholds (KUPA, 200 writers; description)", "tab:idfull",
                 "lXllllXXX", ["Summary", "Rank AUC", "Top-1", "Top-5", "Top-10", "Top-20", "EER", "Registration scale",
                               "Essay $\\to$ copy"], rows,
                 note=f"Copy registered and essay matched unless marked; rank AUC with 95\\% percentile intervals over "
                      f"2,000 resamples of writers. Main standardization: each task's values standardized over the "
                      f"writers within that task; registration scale: both tasks standardized with the copy's means "
                      f"and standard deviations. Chance for top-1, -5, -10 and -20: {ch}. EER: equal error rate. "
                      f"$S_1$: the sentence summary of the earlier preprint with its external scale. $I$ is counted "
                      f"on its three numbers.", size="\\scriptsize")


# ---------- Appendix E ----------
def same_ref(refs, what):
    """Decision 19 writes a reference on one set: the reference cells read on the summaries' writers must be the same
    set (same count, same share) as the one written; otherwise stop."""
    if len({(c["persons"], pn.fin(c["share_vs_own"])) for c in refs}) != 1:
        sys.exit(f"{what}: the reference is read on sets that differ; it would be written on two sets")


def full_cells(R, key, title, label):
    rec = R[key]
    rp = rec["rule_persons"]
    ref = "R" if key == "v6k" else "R_loc"
    rows = []
    for st in SETS:
        for b in BASES:
            for r in ("inside/before",) + pn.SHUFFLES:
                c_ = rp[st][b][r]
                row = [SETNAME[st], BASENAME[b], ROWNAME[r]]
                for s in ("U@2000", "B@2000", "Bsd@2000", "I"):
                    if fixed(s, b, r):
                        if pn.fin(c_[s]["stage"]["share_vs_own"]) != 0:
                            sys.exit(f"{key} {s} {st} {b} {r}: a cell fixed by construction has a share")
                        row.append("fixed")
                    else:
                        row.append(cell(c_[s]["stage"], f"{key} {s} {st} {b} {r}"))
                same_ref([c_[s][ref] for s in ("U@2000", "B@2000", "Bsd@2000", "I")], f"{key} {ref} {st} {b} {r}")
                row.append(cell(c_["U@2000"][ref], f"{key} {ref} {st} {b} {r}"))
                rows.append(row)
    return table(title, label, "lllXXXXX", ["Data", "Split", "Operation", "$U$", "$B$", "$B_{sd}$", "$I$",
                                            STAGENAME[ref] + " (on $U$'s writers)"], rows,
                 note="Share, class (a above .60, b below, u undecided) and number of writers. Each summary on the "
                      "writers who have it and the reference; the reference on $U$'s writers, which are the writers of "
                      "every summary written here. Fixed: decided by construction (named, not read): $U$ and $B$ under "
                      "inside, $B$ under shuffled in halves with the first--second half split.", size="\\scriptsize")


def reliability(R):
    rel = R["v6l"]["reliability"]
    names = {"S0": "$S_0$", "U@2000": "$U$", "B@2000": "$B$", "Bsd@2000": "$B_{sd}$", "B_complete@2000": "$B$ complete",
             "Bsd_complete@2000": "$B_{sd}$ complete", "I": "$I$", "R": "$R$"}
    comp = {"S0": [f"quantile {q}" for q in (".05", ".10", ".25", ".50", ".75", ".90", ".95")],
            "U@2000": ["log median short gap, median", "log median short gap, IQR", "long-gap share, median",
                       "long-gap share, IQR"],
            "I": ["all gaps", "within words", "between words"],
            "R": ["sentence start", "subsentence start", "word start", "within word", "sentence minus within",
                  "long time at sentence or subsentence starts", "lag-1 correlation"]}
    rows = []
    for f in rel["lwp_lwp"]["features"]:
        base, _, idx = f.partition("[")
        i_ = int(idx.rstrip("]"))
        if i_ < 0 or (base in comp and i_ >= len(comp[base])):
            sys.exit(f"{f}: index outside the components")
        if base in comp:
            if len(comp[base]) != sum(1 for g in rel["lwp_lwp"]["features"] if g.startswith(base + "[")):
                sys.exit(f"{base}: the component names do not match the record's features")
            row = [f"{names[base]}: {comp[base][i_]}"]
        else:
            if i_ != 0:
                sys.exit(f"{f}: a one-number summary with an index")
            row = [names[base]]
        for st in SETS:
            x = rel[st]["features"][f]
            for v_ in ("mean_r", "mean_rSB"):
                if not -1 <= pn.fin(x[v_]) <= 1:
                    sys.exit(f"{f}: reliability outside [-1, 1]")
            lo, hi = pn.ci(x["rSB_percentile_95"])
            row += [pn.f2(x["mean_r"]), f"{pn.f2(x['mean_rSB'])} [{pn.f2(lo)}, {pn.f2(hi)}]"]
        rows.append(row)
    return table("Split-half reliability of every feature (rule count's writers; description)", "tab:rel", "lllll",
                 ["Feature", "KUPA $r$", "KUPA $r_{SB}$", "LWP $r$", "LWP $r_{SB}$"], rows,
                 note="Mean over 5,000 random splits of each record into two halves of its sentences; $r$ uncorrected, "
                      "$r_{SB}$ with the Spearman--Brown step, with the 2.5 and 97.5 percentiles of $r_{SB}$ over splits "
                      "(no interval over writers). Components in the order of the record (for $U$: the log median "
                      "short gap, then the long-gap share; $R_{loc}$ is $R$ less its lag-1 correlation).", size="\\scriptsize")


def split_dependence(R):
    """v6s's written cells. Each split's class is checked against v6n's class of the same cell (the references on the
    writers with R, a summary on its rule writers); the between-word mean (I[2], not in v6n) only against the three
    classes."""
    s = R["v6s"]["main_cells"]
    n = R["v6n"]
    rows = []
    for part in ("primary", "shuffles"):
        for cname, stages in s[part].items():
            st, row_ = cname.split(" ", 1)
            for stage, v in stages.items():
                if not isinstance(v, dict):
                    continue
                lo, hi = pn.ci(v["1-p_o_ci"])
                if not lo <= pn.fin(v["1-p_o"]) <= hi:
                    sys.exit(f"{cname} {stage}: 1-p_o outside its interval")
                for b in BASES:
                    rec_class(v["class"][b], f"{cname} {stage} {b}")
                    if stage in ("R_loc", "R_mean"):
                        c_ = n["count_1"][st][b][row_][stage]
                    elif stage in n["rule_persons"][st][b][row_]:
                        c_ = n["rule_persons"][st][b][row_][stage]["stage"]
                    else:
                        continue
                    if cls(c_, f"v6n {stage} {st} {b} {row_}") != v["class"][b]:
                        sys.exit(f"{cname} {stage} {b}: v6s's class differs from v6n's")
                # v6s writes 1 - p_o and its interval at three decimals; they are written as recorded
                rows.append([SETNAME[st], rowname(row_) + ("*" if part == "primary" else ""), STAGENAME.get(stage, {"I[2]": "between-word mean"}.get(stage, tex(stage))),
                             str(v["common"]), f"{pn.f2(v['1-p_o'], places=3)} [{pn.f2(lo, places=3)}, {pn.f2(hi, places=3)}]",
                             f"{v['class']['oddeven']} / {v['class']['halves']}"])
    return table("Writers whose verdict differed between the two splits, in the cells the disclosure rule allowed (description)",
                 "tab:split", "llllXX", ["Data", "Operation", "Summary", "Writers", "Differed", "Class (odd--even / halves)"],
                 rows, note="Writers counted under both splits (the common writers); the share whose verdict differed "
                            "($1 - p_o$) with a 95\\% interval; the classes are the record's, checked against the classes of the same "
                            "cells in the dose counts. *Primary cells, named before counting: inside at 75\\% and 100\\% and cross at "
                            "100\\%. No chance level is written (disclosure rule).")


def contrast(R):
    lab = R["v6m"]["part_1"]["contrast"]["labelled"]
    rows = []
    for c_ in ("oddeven | copy", "halves | copy", "oddeven | essay", "halves | essay"):
        b, base = c_.split(" | ")
        row = [BASENAME[b], f"the {base}'s"]
        for s in ("U@2000", "B@2000", "Bsd@2000", "I"):
            x = lab[s][c_]
            row.append(cell(x["stage"], f"contrast {s} {c_}", n=False) + " " + sign_class(x["paired_minus_R_loc"]["ci"], s))
        same_ref([lab[s][c_]["R_loc"] for s in ("U@2000", "B@2000", "Bsd@2000", "I")], f"contrast {c_}")
        row.append(cell(lab["U@2000"][c_]["R_loc"], f"contrast R_loc {c_}"))
        rows.append(row)
    return table("Copy against essay (second count's writers; not evaluable)", "tab:contrast", "llXXXXX",
                 ["Split", "Own variation", "$U$", "$B$", "$B_{sd}$", "$I$", "$R_{loc}$ (on $U$'s writers)"], rows,
                 note="Change: distance between a writer's copy and essay; own variation: the distance between the two "
                      "parts of the copy or of the essay. Share and class of each summary, then the class of the "
                      "paired interval of the summary minus $R_{loc}$ on the summary's writers ($>0$, $\\ni 0$ across 0, "
                      "$<0$). These pair distances with different numbers of features.")


def doses(R):
    n = R["v6n"]
    c1, nrp = n["count_1"], n["rule_persons"]
    rows = []
    for st in SETS:
        for b in BASES:
            for fam in ("inside", "cross"):
                for x in (25, 50, 75, 100):
                    for ref in ("R_loc", "R_mean"):
                        same_ref([c1[st][b][f"{fam}_{x}"][ref]] + [nrp[st][b][f"{fam}_{x}"][s][ref] for s in ("I", "Bsd@2000")],
                                 f"dose {ref} {st} {b} {fam}_{x}")
                for q in ("R_loc", "R_mean", "I", "Bsd@2000"):
                    row = [SETNAME[st], BASENAME[b], fam, STAGENAME[q]]
                    for x in (25, 50, 75, 100):
                        r = f"{fam}_{x}"
                        c_ = c1[st][b][r][q] if q.startswith("R_") else nrp[st][b][r][q]["stage"]
                        row.append(cell(c_, f"dose {q} {st} {b} {r}", n=False))
                    rows.append(row)
    return table(
                 "Shares by dose (second count's writers; descriptions)",
                 "tab:dose", "llllXXXX", ["Data", "Split", "Operation", "Summary", "25\\%", "50\\%", "75\\%", "100\\%"], rows,
                 note="All cells are descriptions. Inside at 100\\% applies the same swaps as the second count's inside "
                      "(Table~\\ref{tab:cellsm}), recounted with the dose rows' resampling keys; cross at 100\\% and "
                      "$R_{mean}$ were never rule cells. The references on all writers who have them; $I$ and $B_{sd}$ "
                      "on the writers who have both the summary and the reference. Share and class (a, b, u).", size="\\scriptsize")


def moved(R):
    t = R["v6n"]["tallies"]
    rows = []
    for st in SETS:
        for fam in ("inside", "cross"):
            row = [SETNAME[st], fam]
            for x in (25, 50, 75, 100):
                d = t[st]["doses"][f"{fam}_{x}"]
                sw = pn.fin(d["swaps"]["median"])
                row.append(f"{sw:g}; {pn.fmt(100 * pn.fin(d['time_moved_share']['median']), 1)}\\%")
            rows.append(row)
    return table("What moved by dose (medians over writers; description)", "tab:moved", "llXXXX",
                 ["Data", "Operation", "25\\%", "50\\%", "75\\%", "100\\%"], rows,
                 note="Median number of swaps (inside: sentence-initial gaps swapped; cross: long sentence-initial "
                      "gaps swapped), then the median share of the record's time moved (inside: the sum over swaps of "
                      "the absolute difference of the two gaps; cross: the sum of the moved long gaps; each over the "
                      "record's total time); the two medians are taken separately.")


def border(R):
    """Cells of the dose rows where R_loc's class differs between the summaries' writers (classes only)."""
    nrp = R["v6n"]["rule_persons"]
    rows, same = [], []
    for st in SETS:
        for b in BASES:
            for r in [x for x in nrp[st][b] if x.startswith(("inside_", "cross_"))]:
                got = {}
                for s in ("U@2000", "B@2000", "Bsd@2000", "I"):
                    c_ = nrp[st][b][r][s]["R_loc"]
                    got[s] = (cls(c_, f"border {s} {st} {b} {r}"), c_["persons"])
                if len({g[0] for g in got.values()}) > 1:
                    rows.append([SETNAME[st], BASENAME[b], r.replace("_", " ") + "\\%"]
                                + [f"{g[0]} ({g[1]})" for g in got.values()])
                    # each summary's writers are a subset of the writers with R; equal in number to them, they are the
                    # same writers
                    n_r = R["v6n"]["count_1"][st][b][r]["R_loc"]["persons"]
                    same.append(all(g[1] == n_r for g in got.values()) and
                                len({pn.fin(nrp[st][b][r][s]["R_loc"]["share_vs_own"]) for s in got}) == 1)
    if not rows:
        sys.exit("no border cell found; the body names one")
    if not all(same):
        sys.exit("a border cell is read on sets of different size or with different shares; the note says the same writers")
    return table("Border cells: cells where the class of $R_{loc}$ differs with the resampling key",
                 "tab:border", "lllXXXX", ["Data", "Split", "Dose", "Key of $U$", "Key of $B$", "Key of $B_{sd}$", "Key of $I$"],
                 rows, note="Class and number of writers. In these cells the four summaries' writers are the same "
                            "writers, with the same share of the reference; the reference is resampled with each "
                            "summary's key, and the class differs with the key. Only the classes are written.")


def groups(R):
    o, p = R["v6o"], R["v6p"]
    rows = []
    for st in SETS:
        for b in BASES:
            g = o["groups"]["n_s"][st][b]
            if not g["written"]:
                continue
            med = g["covariate_median_by_group"]
            own = o["main_cells"]["groups_n_s"][f"{st} {b}"]["base_by_group"]["R_loc"] if f"{st} {b}" in o["main_cells"]["groups_n_s"] else None
            row = [SETNAME[st], BASENAME[b], ", ".join(f"{x:g}" for x in g["cuts"]),
                   " / ".join(str(g["persons"][x]) for x in ("low", "mid", "high")),
                   " / ".join(f"{med[x]:g}" if med[x] != int(med[x]) else f"{med[x]:.1f}" for x in ("low", "mid", "high"))]
            row.append(" / ".join(pn.fmt(own[x], 2) for x in ("low", "mid", "high")) if own else "--")
            rows.append(row)
    t1 = table("Groups by sentence starts per part (writers with $R$; description)", "tab:groups", "llllXX",
               ["Data", "Split", "Cuts", "Writers", "Median sentence starts", "Median own variation ($R_{loc}$)"],
               rows, note="Sentence starts per part: the mean over the two parts of the number of sentence-initial gaps. Groups "
                         "low / mid / high, cut at the two values given.")
    rows = []
    agree = agreement_cells(R)
    for st in SETS:
        for b in BASES:
            fam = p["groups"]["n_s"][st][b]["I"]
            for r in ("inside_75", "inside_100", "cross_100"):
                f_ = fam[r]
                if not f_.get("written"):
                    rows.append([SETNAME[st], BASENAME[b], r.replace("_", " ") + "\\%", "withheld", "", ""])
                    continue
                row = [SETNAME[st], BASENAME[b], r.replace("_", " ") + "\\%"]
                for g in ("low", "mid", "high"):
                    x = f_["groups"][g]
                    k_ = klass(x["share_vs_own_ci"], what=f"group {st} {b} {r} {g}")
                    if k_ != x["class"]:
                        sys.exit(f"group {st} {b} {r} {g}: class differs from the record")
                    st_ = {"gate": "g", "gate in group only": "gg", "R_mean only": "m", "not read": "n"}[x["state"]]
                    if (st, r) in agree:  # decision 25: the share is given as the number of writers
                        n_ = x["persons"]
                        if not isinstance(n_, int) or isinstance(n_, bool) or n_ < 5:
                            sys.exit(f"group {st} {b} {r} {g}: persons {n_!r}")
                        row.append(f"[{n_}]\\,{LETTER[k_]}, {st_}")
                        WITHHELD.append(f"I group {st} {b} {r} {g}")
                    else:
                        row.append(f"{pn.f2(x['share_vs_own'])}\\,{LETTER[k_]}, {st_}")
                rows.append(row)
    t2 = table("$I$ by group of sentence starts (description)", "tab:igroups", "lllXXX",
               ["Data", "Split", "Row", "Low", "Mid", "High"], rows,
               note="Share and class of $I$ (a above .60, b below, u undecided), then the state of the reference in the "
                    "group: g the cell's gate reached and the group's $R_{loc}$ above; gg only the group's $R_{loc}$ "
                    "above; m only the group's $R_{mean}$ above; n neither. Where the agreement of the verdicts between "
                    "the two splits is given for $I$ (Table~\\ref{tab:split}), the share is replaced by the number of writers in "
                    "brackets (Appendix~A, disclosure rule).")
    return t1 + t2


WITHHELD: list = []  # cells the ledger gives as counts of writers (decision 25), filled by groups()


def agreement_cells(R):
    """(set, row) of the primary cells where v6s wrote I's agreement between the splits: there the groups' shares
    under both splits would join v6s's one minus agreement, a combination v6s's check did not cover (decision 25)."""
    out = set()
    for cellname, stages in R["v6s"]["main_cells"]["primary"].items():
        st, r = cellname.split()
        if isinstance(stages.get("I"), dict):
            out.add((st, r))
    if not out <= {(s_, r_) for s_ in SETS for r_ in ("inside_75", "inside_100", "cross_100")}:
        sys.exit(f"v6s wrote I's agreement in a cell outside the group table: {sorted(out)}")
    return out


# ---------- Appendix G ----------
RULE_LABEL = {"shows these operations beyond the smallest difference": "shows",
              "does not show them beyond it": "does not show",
              "not beyond, partly undecided": "does not show, partly undecided",
              "beyond, partly undecided": "shows, partly undecided",
              "undecided": "undecided", "mixed directions": "mixed directions", "not evaluable": "not evaluable"}
MARGINS = (0.55, 0.60, 0.65)


def rule_label(classes):
    """The rule of the conditions (v6k rule.part_1) over the evaluable cells that can change, in the record's words."""
    if any(c_ not in CLASSES for c_ in classes):
        sys.exit(f"rule_label: unknown class in {classes!r}")
    if len(classes) < 4:
        return "not evaluable"
    s_ = set(classes)
    if s_ == {"above"}:
        return "shows these operations beyond the smallest difference"
    if s_ == {"below"}:
        return "does not show them beyond it"
    if s_ == {"undecided"}:
        return "undecided"
    if "above" in s_ and "below" in s_:
        return "mixed directions"
    return "not beyond, partly undecided" if "below" in s_ else "beyond, partly undecided"


def at(c, mg, what):
    """Class of a cell at a margin; at .60 it must agree with the record."""
    return cls(c, what) if mg == 0.60 else klass(c["share_vs_own_ci"], mg, what)


def margins(R):
    cells = [(st, b, r) for st in SETS for b in BASES for r in ("inside/before",) + pn.SHUFFLES]
    stages = ("U@2000", "B@2000", "Bsd@2000", "I")
    # (1) classes of the rule cells, the summaries and the gates (gates on U's writers, as in Tables cellsk and cellsm)
    rows = []
    for key, ref, who in (("v6k", "R", "rule count"), ("v6m", "R_loc", "second count")):
        rp = R[key]["rule_persons"]
        for s in stages + (ref,):
            row = [STAGENAME[s] + (" (gate)" if s == ref else ""), who]
            for mg in MARGINS:
                cnt = {"above": 0, "below": 0, "undecided": 0}
                for st, b, r in cells:
                    if s != ref and fixed(s, b, r):
                        continue
                    c_ = rp[st][b][r]["U@2000"][ref] if s == ref else rp[st][b][r][s]["stage"]
                    cnt[at(c_, mg, f"margin {key} {s} {st} {b} {r}")] += 1
                row.append(f"{cnt['above']} / {cnt['below']} / {cnt['undecided']}")
            rows.append(row)
    t1 = table("Classes of the rule cells at three margins (recomputed from the recorded intervals; description)",
               "tab:margin", "llXXX", ["Summary", "Count", ".55", ".60", ".65"], rows,
               note="Cells above / below / undecided over the 12 rule cells (inside and the two reshuffles at the full "
                    "dose, two datasets, two splits), cells fixed by construction left out. Each summary on its own "
                    "writers; the gate on $U$'s writers. The gate is not applied in this table.")
    # (2) the rule's labels when the margin of both the gate and the summary is moved
    rows, lab60, labels = [], {}, {}
    for key, ref in (("v6k", "R"), ("v6m", "R_loc")):
        rp = R[key]["rule_persons"]
        for s in stages:
            row = [STAGENAME[s], {"v6k": "rule count", "v6m": "second count"}[key]]
            for mg in MARGINS:
                got = []
                for st, b, r in cells:
                    if fixed(s, b, r):
                        continue
                    c_ = rp[st][b][r][s]
                    if at(c_[ref], mg, f"label gate {key} {s} {st} {b} {r}") == "above":
                        got.append(at(c_["stage"], mg, f"label {key} {s} {st} {b} {r}"))
                lb = rule_label(got)
                labels[(key, s, mg)] = lb
                if mg == 0.60:
                    lab60[(key, s)] = lb
                row.append(f"{RULE_LABEL[lb]} ({len(got)})")
            rows.append(row)
    rec_lab = {("v6k", s): R["v6k"]["main_cells"][f"rule | {s}"]["label"] for s in stages}
    rec_lab.update({("v6m", s): R["v6m"]["part_0"]["by_gate"]["R_loc"][s]["label"] for s in stages})
    if lab60 != rec_lab:
        sys.exit(f"the rule's labels recomputed at .60 differ from the records: {lab60} vs {rec_lab}")
    # Appendix G: at .55, U changes label in both counts and B_sd in the rule count only (above in one reshuffle cell,
    # LWP odd-even shuffled); at .65 no label changes
    changed = {(key, s, mg) for (key, s), lb in lab60.items() for mg in (0.55, 0.65) if labels[(key, s, mg)] != lb}
    if changed != {("v6k", "U@2000", 0.55), ("v6m", "U@2000", 0.55), ("v6k", "Bsd@2000", 0.55)}:
        sys.exit(f"labels changed with the margin: {sorted(changed)}; the text names U in both counts and B_sd in the rule count at .55")
    ab = [(st, b, r) for st, b, r in cells if at(R["v6k"]["rule_persons"][st][b][r]["Bsd@2000"]["stage"], 0.55, "bsd .55") == "above"]
    if ab != [("lwp_lwp", "oddeven", "shuffled/before")] or labels[("v6k", "Bsd@2000", 0.55)] != "mixed directions":
        sys.exit(f"B_sd at .55 in the rule count: above in {ab}; the text names one reshuffle cell and mixed directions")
    t2 = table("The rule's labels at three margins (the margin moved for the gate and the summaries together; description)",
               "tab:marginlabels", "llXXX", ["Summary", "Count", ".55", ".60", ".65"], rows,
               note="In brackets the number of cells the label is given over (the gate above, the summary not fixed by "
                    "construction). At .60 the labels are those of the records. Gate $R$ in the rule count and $R_{loc}$ "
                    "in the second count, each read on the summary's writers.")
    # (3) the reach of R_loc by dose on the writers with R (the reference of Figure 1)
    c1 = R["v6n"]["count_1"]
    rows, reach = [], {}
    for st in SETS:
        for b in BASES:
            for fam in ("inside", "cross"):
                row = [SETNAME[st], BASENAME[b], fam]
                for mg in MARGINS:
                    ks = [at(c1[st][b][f"{fam}_{x}"]["R_loc"], mg, f"reach {st} {b} {fam} {x}") for x in (25, 50, 75, 100)]
                    reach[(st, b, fam, mg)] = ks
                    row.append(" ".join(LETTER[k_] for k_ in ks))
                rows.append(row)
    # what the body says of the reach (Section 5.2): at .55 KUPA inside 100% is reached under both splits, and LWP cross
    # 100% under first-second half; at .65 LWP inside 75% is no longer reached; inside 50% is reached at no margin
    if not all(reach[("kupa_essay", b, "inside", 0.55)][3] == "above" and reach[("kupa_essay", b, "inside", 0.60)][3] != "above"
               for b in BASES):
        sys.exit("at .55 KUPA inside 100% is not reached under both splits")
    if not all(reach[("lwp_lwp", b, "inside", 0.65)][2] != "above" and reach[("lwp_lwp", b, "inside", 0.60)][2] == "above"
               and reach[("lwp_lwp", b, "inside", 0.65)][3] == "above" for b in BASES):
        sys.exit("at .65 LWP inside 75% is still reached, or 100% is not")
    if reach[("lwp_lwp", "halves", "cross", 0.55)][3] != "above" or reach[("lwp_lwp", "oddeven", "cross", 0.55)][3] == "above":
        sys.exit("at .55 the reach of cross is not LWP first-second half 100% only")
    if any(reach[(st, b, fam, mg)][x] == "above" for st in SETS for b in BASES for fam in ("inside", "cross")
           for mg in MARGINS for x in (0, 1)):
        sys.exit("a dose of 25% or 50% is reached at some margin")
    # the whole set of cells reached at each margin, as the body and the paragraph of Appendix G name them
    lwp_in = {("lwp_lwp", b, "inside", x) for b in BASES for x in (75, 100)}
    want = {0.55: lwp_in | {("kupa_essay", b, "inside", 100) for b in BASES} | {("lwp_lwp", "halves", "cross", 100)},
            0.60: lwp_in, 0.65: {("lwp_lwp", b, "inside", 100) for b in BASES}}
    for mg in MARGINS:
        got = {(st, b, fam, x) for st in SETS for b in BASES for fam in ("inside", "cross")
               for i_, x in enumerate((25, 50, 75, 100)) if reach[(st, b, fam, mg)][i_] == "above"}
        if got != want[mg]:
            sys.exit(f"cells reached at {mg}: {sorted(got)}; the text names {sorted(want[mg])}")
    t3 = table("Reach of $R_{loc}$ by dose at three margins (second count's writers, all who have $R$; description)",
               "tab:marginreach", "lllXXX", ["Data", "Split", "Operation", ".55", ".60", ".65"], rows,
               note="Classes at the doses 25, 50, 75 and 100\\% (a above the margin, b below, u undecided).")
    # (4) the cells read through the gate in the dose rows, and the summaries' classes there
    nrp = R["v6n"]["rule_persons"]
    rd = ("U@2000", "B@2000", "Bsd@2000", "I", "B_complete@2000", "Bsd_complete@2000")
    rows, n60, read = [], None, {}
    for mg in MARGINS:
        cnt = {"above": 0, "below": 0, "undecided": 0}
        where = set()
        read[mg] = []
        for st in SETS:
            for b in BASES:
                for fam in ("inside", "cross"):
                    for x in (25, 50, 75, 100):
                        for s in rd:
                            if fam == "inside" and s in ("U@2000", "B@2000"):
                                continue  # fixed by construction under inside
                            c_ = nrp[st][b][f"{fam}_{x}"][s]
                            if at(c_["R_loc"], mg, f"read gate {st} {b} {fam} {x} {s}") == "above":
                                cnt[at(c_["stage"], mg, f"read {st} {b} {fam} {x} {s}")] += 1
                                read[mg].append((st, b, fam, x, s))
                                where.add(f"{SETNAME[st]} {fam}")
        if cnt["above"]:
            sys.exit(f"a summary is above the margin {mg} in a cell read through the gate; the text says none is")
        n = sum(cnt.values())
        if mg == 0.60:
            n60 = n
        rows.append([f"{mg:.2f}"[1:], str(n), f"{cnt['above']} / {cnt['below']} / {cnt['undecided']}",
                     ", ".join(sorted(where))])
    if n60 != 15:
        sys.exit(f"at .60, {n60} cells are read through the gate; the body names 15")
    # the note names every cell where reading on a summary's writers differs from the reach on all writers
    diffs = set()
    for mg in MARGINS:
        if len(set(read[mg])) != len(read[mg]):
            sys.exit(f"a cell is read twice at {mg}")
        want_ = {(st, b, fam, x, s) for st in SETS for b in BASES for fam in ("inside", "cross")
                 for i_, x in enumerate((25, 50, 75, 100)) if reach[(st, b, fam, mg)][i_] == "above"
                 for s in rd if not (fam == "inside" and s in ("U@2000", "B@2000"))}
        diffs |= {(mg,) + w_ for w_ in set(read[mg]) ^ want_}
    if diffs != {(0.55, "kupa_essay", "halves", "inside", 100, "Bsd_complete@2000"),
                 (0.60, "lwp_lwp", "oddeven", "inside", 75, "I"),
                 (0.65, "lwp_lwp", "halves", "inside", 75, "Bsd_complete@2000")}:
        sys.exit(f"the cells named in the note of the read table changed: {sorted(diffs)}")
    t4 = table("Cells of the dose rows read through the gate at three margins (second count's writers; description)",
               "tab:marginread", "lllX", ["Margin", "Cells read", "Above / below / undecided", "Where"], rows,
               note="Summaries that can move ($B_{sd}$, $I$ and the complete-burst variants under inside; also $U$ and "
                    "$B$ under cross), at every dose, read where $R_{loc}$ on the summary's own writers is above the "
                    "margin; the margin moved for the gate and the summary together. On a summary's writers the "
                    "reference is resampled with that summary's key, so the cells read can differ from the reach on all "
                    "writers (Table~\\ref{tab:marginreach}). These are three cells: at .60 LWP odd--even inside 75\\% for $I$ "
                    "is not read (the border cell of Section~\\ref{sec:inside}); at .65 one of the nine is LWP "
                    "first--second half inside 75\\% for $B_{sd}$ complete, read; at .55 KUPA essays under "
                    "first--second half inside 100\\% for $B_{sd}$ complete is not read.")
    return t1 + t2 + t3 + t4


# ---------- Appendix E: complete bursts ----------
def complete(R):
    """Complete-burst variants: the rule cells of both counts and the LWP inside cells of the dose rows."""
    rows = []
    k_, m_ = R["v6k"]["rule_persons"], R["v6m"]["rule_persons"]
    for st in SETS:
        for b in BASES:
            for r in ("inside/before",) + pn.SHUFFLES:
                row = [SETNAME[st], BASENAME[b], ROWNAME[r]]
                for rp, s, ref, tag in ((k_, "Bsd_complete@2000", "R", "k"), (m_, "B_complete@2000", "R_loc", "m"),
                                        (m_, "Bsd_complete@2000", "R_loc", "m")):
                    c_ = rp[st][b][r][s]
                    row.append(cell(c_["stage"], f"complete {tag} {s} {st} {b} {r}") + "; "
                               + LETTER[cls(c_[ref], f"complete gate {tag} {s} {st} {b} {r}")])
                rows.append(row)
    t1 = table("Complete-burst variants in the rule cells (description beside the rule)", "tab:complete", "lllXXX",
               ["Data", "Split", "Operation", "$B_{sd}$ complete (rule count)", "$B$ complete (second count)",
                "$B_{sd}$ complete (second count)"], rows,
               note="Bursts bounded by long gaps at both ends only; a part with fewer than two such bursts has no "
                    "$B_{sd}$ complete. Share, class and number of writers, then the class of the gate ($R$ in the rule "
                    "count, $R_{loc}$ in the second count) on the same writers (class only). $B$ complete was not "
                    "counted in the rule count; it was counted afterwards on the rule count's writers as a description, "
                    "not given here. Neither variant is fixed by construction.", size="\\scriptsize")
    nrp = R["v6n"]["rule_persons"]
    rows, n_read = [], 0
    for b in BASES:
        for x in (75, 100):
            row = [BASENAME[b], f"{x}\\%"]
            for s in ("Bsd@2000", "I", "B_complete@2000", "Bsd_complete@2000"):
                c_ = nrp["lwp_lwp"][b][f"inside_{x}"][s]
                if cls(c_["R_loc"], f"gate cells {s} {b} {x}") == "above":
                    n_read += 1
                    row.append(cell(c_["stage"], f"gate cells {s} {b} {x}"))
                else:
                    row.append("gate not above")
            rows.append(row)
    # the body counts 15 such cells over the four summaries (25% and 50% reach no gate)
    for b in BASES:
        for x in (25, 50):
            if any(cls(nrp["lwp_lwp"][b][f"inside_{x}"][s]["R_loc"], f"gate {s} {b} {x}") == "above"
                   for s in ("Bsd@2000", "I", "B_complete@2000", "Bsd_complete@2000")):
                sys.exit(f"the gate is above at LWP inside {x}%; the breakdown table leaves that dose out")
    if n_read != 15:
        sys.exit(f"{n_read} gate cells in the breakdown; the body names 15")
    notup = [(s, b, x, cls(nrp["lwp_lwp"][b][f"inside_{x}"][s]["R_loc"], "gate")) for b in BASES for x in (75, 100)
             for s in ("Bsd@2000", "I", "B_complete@2000", "Bsd_complete@2000")
             if cls(nrp["lwp_lwp"][b][f"inside_{x}"][s]["R_loc"], "gate") != "above"]
    if notup != [("I", "oddeven", 75, "undecided")]:
        sys.exit(f"cells without the gate: {notup}; the note names I at odd-even 75% (undecided)")
    t2 = table("The 15 cells read through the gate: LWP, inside (second count's writers; description)", "tab:gatecells",
               "llXXXX", ["Split", "Dose", "$B_{sd}$", "$I$", "$B$ complete", "$B_{sd}$ complete"], rows,
               note="Share, class and number of writers where $R_{loc}$ on the summary's writers was above .60. At "
                    "odd--even 75\\% the reference was undecided on $I$'s writers (a border cell, Table~\\ref{tab:border}).")
    return t1 + t2


# ---------- Appendix A: writers ----------
def flow(R):
    rows = []
    for key, who, frm in (("v6k", "rule count", {"kupa": 437, "lwp": 204}), ("v6m", "second count", {"kupa": 656, "lwp": 411})):
        rec = R[key]
        tal = rec["writer"] if key == "v6k" else rec["tallies"]
        for ds, st in (("kupa", "kupa_essay"), ("lwp", "lwp_lwp")):
            looked = rec["persons"][ds][f"ids_looked_at_from_{frm[ds]}"]
            t = tal[st]
            for x in (looked, t["count_persons"], t["left_out_fewer_than_8_or_part_without_gap"]):
                if isinstance(x, bool) or not isinstance(x, int) or x < 0:
                    sys.exit(f"{key} {st}: a count of writers is not a non-negative integer")
            if looked < 200:
                sys.exit(f"{key} {st}: fewer identifiers looked at than writers drawn")
            eligible = t["count_persons"] + t["left_out_fewer_than_8_or_part_without_gap"]
            if eligible != 200:
                sys.exit(f"{key} {st}: {eligible} eligible writers, the conditions draw 200")
            if ds == "kupa" and tal["kupa_copy"]["count_persons"] != 200:
                sys.exit(f"{key}: not every KUPA copy was counted")
            per = [rec["rule_persons"][st][b][r][s]["stage"]["persons"] for b in BASES for r in ("inside/before",) + pn.SHUFFLES
                   for s in ("U@2000", "B@2000", "Bsd@2000", "I")]
            ref = "R" if key == "v6k" else "R_loc"
            for b in BASES:
                # the writers left out of a cell lack the reference: every summary has all counted writers, the
                # reference fewer, and the cell's writers are as many as the reference's
                c1 = rec["count_1"][st][b]["shuffled/before"]
                if any(c1[s]["persons"] != t["count_persons"] for s in ("U@2000", "B@2000", "Bsd@2000", "I")):
                    sys.exit(f"{key} {st} {b}: a summary lacks counted writers; the note blames the reference")
                for r in ("inside/before",) + pn.SHUFFLES:
                    cc = rec["rule_persons"][st][b][r]
                    ns = {cc[s]["stage"]["persons"] for s in ("U@2000", "B@2000", "Bsd@2000", "I")}
                    if len(ns) != 1:
                        sys.exit(f"{key} {st} {b} {r}: the four summaries have different writers")
                    c_ = cc["U@2000"]["stage"]
                    if c_["persons"] + c_["left_out_no_vector"] != t["count_persons"] or \
                            rec["count_1"][st][b][r][ref]["persons"] != c_["persons"]:
                        sys.exit(f"{key} {st} {b} {r}: writers in the cell, left out and with the reference do not agree")
            byb = {b: rec["rule_persons"][st][b]["shuffled/before"]["U@2000"]["stage"]["persons"] for b in BASES}
            rows.append([who, {"kupa": "KUPA essays", "lwp": "LWP"}[ds], f"{frm[ds]}", str(looked),
                         str(looked - 200), str(t["count_persons"]), str(t["left_out_fewer_than_8_or_part_without_gap"]),
                         f"{byb['oddeven']} / {byb['halves']}", f"{min(per)}--{max(per)}"])
    return table("Writers from the providers' orders to the cells", "tab:flow", "llrrrrrXX",
                 ["Count", "Data", "From", "Looked at", "Not eligible", "Counted", "Left out", "Per cell (odd--even / halves)",
                  "Range over cells"], rows,
                 note="From: the first position in the fixed order looked at. Eligible (200 per dataset and count, as the "
                      "conditions draw): at least 5 sentences and 5 full 10-s windows (the last counted press at least 50~s after the first), and every sentence holding at least "
                      "one gap whether the gap before its first press is counted in it, in the previous sentence or in "
                      "neither (in KUPA in both tasks); this rule was carried over from the earlier preprint's draws. "
                      "Counted: at least 8 sentences and a gap in every part under both splits; left out: the others. "
                      "Per cell: writers with the summary and the reference on both parts (the same for $U$, $B$, $B_{sd}$ "
                      "and $I$; the writers left out had a part without a gap at one of the reference's boundaries); range over cells: over "
                      "the six rule cells of the dataset and the four summaries. The complete-burst variants have as many writers or fewer "
                      "(Table~\\ref{tab:complete}). All 200 KUPA copies were counted for identification.",
                 size="\\scriptsize")


def lwp_reading(R):
    """LWP's reading tally (Appendix H): the same file was read for both counts."""
    t = [R[v]["persons"]["lwp"]["tally_of_reading"] for v in ("v6k", "v6m")]
    for x in t:
        for k_ in ("events", "decreasing_down_time", "not_counted", "dropped_equal_times"):
            if isinstance(x[k_], bool) or not isinstance(x[k_], int) or x[k_] < 0:
                sys.exit(f"LWP tally {k_} is not a count")
    if t[0] != t[1]:
        sys.exit("the two counts read LWP's file differently")
    t = t[0]
    if t["decreasing_down_time"] != 1:
        sys.exit("Appendix H writes 'event' in the singular")
    return ("\\newcommand{\\LWPevents}{" + f"{t['events']:,}" + "}\n"
            "\\newcommand{\\LWPdecr}{" + str(t["decreasing_down_time"]) + "}\n"
            "\\newcommand{\\LWPequal}{" + str(t["dropped_equal_times"]) + "}\n")


def main():
    R = pn.load()
    for v in R:
        R[v]["_sha"] = pn.SHA[v]
    rpath = pn.RES / "prototype_v6r" / "record.json"
    R["v6r"] = json.loads(rpath.read_text(encoding="utf-8"))
    R["v6r"]["_sha"] = "d21b309a60e92b3e1af648049c7719468cf2fc3d23d5ab2773e7056117b1bfdc"  # the SHA-256 the notes give
    pn.check_record("v6r", rpath, R["v6r"]["_sha"])
    C = {}
    for v in ("v6k", "v6l", "v6m", "v6n", "v6o", "v6p", "v6q", "v6r", "v6s"):
        cp = pn.ROOT / "works" / "conditions" / f"prototype_{v}.json"
        if hashlib.sha256(cp.read_bytes()).hexdigest() != R[v]["provenance"]["conditions_sha256"]:
            sys.exit(f"{v}: the conditions file differs from the one the run recorded")
        C[v] = json.loads(cp.read_text(encoding="utf-8"))
    N = json.loads((HERE / "numbers.json").read_text(encoding="utf-8"))
    out = {
        "app-runs.tex": runs(R), "app-pred.tex": predictions(R, C), "app-rlag.tex": rlag(R),
        "app-thr.tex": thresholds(R), "app-id.tex": identification(R),
        "app-cells.tex": full_cells(R, "v6k", "Rule count: every cell", "tab:cellsk")
        + full_cells(R, "v6m", "Second count: every cell", "tab:cellsm"),
        "app-rel.tex": reliability(R), "app-split.tex": split_dependence(R), "app-contrast.tex": contrast(R),
        "app-dose.tex": doses(R) + moved(R), "app-border.tex": border(R), "app-groups.tex": groups(R),
        "app-margin.tex": margins(R), "app-complete.tex": complete(R), "app-flow.tex": flow(R),
    }
    # spot checks against the body's numbers
    k = R["v6k"]
    if pn.f2(k["count_2"]["kupa_copy_to_essay"]["B@2000"]["main"]["rank_auc"]) != N["kAucB"]:
        sys.exit("identification table and body disagree on B")
    t75 = R["v6n"]["tallies"]["lwp_lwp"]["doses"]["inside_75"]
    if f"{pn.fin(t75['swaps']['median']):g}" != N["nSwap75L"]:
        sys.exit("moved table and body disagree")
    wt = writer_threshold(R)
    out["app-wthr.tex"] = ("\\newcommand{\\WthrK}{" + f"{wt['kupa_essay'][0] / 1000:.1f}~s (quartiles {wt['kupa_essay'][1] / 1000:.1f} and {wt['kupa_essay'][2] / 1000:.1f}~s)" + "}\n"
                           "\\newcommand{\\WthrL}{" + f"{wt['lwp_lwp'][0] / 1000:.1f}~s (quartiles {wt['lwp_lwp'][1] / 1000:.1f} and {wt['lwp_lwp'][2] / 1000:.1f}~s)" + "}\n"
                           "\\newcommand{\\WfixSame}{" + "{} ($U$), {} ($B$) and {} ($B_{{sd}}$)".format(*wt["fixed_same"]) + "}\n"
                           "\\newcommand{\\WrefitAbove}{" + "{}, {} and {}".format(*wt["refit_above"]) + "}\n"
                           + lwp_reading(R))
    # decision 25: the cells the ledger gives as counts of writers (3 cells x 2 splits x 3 groups of I in KUPA essays)
    if len(WITHHELD) != 18 or len(set(WITHHELD)) != 18:
        sys.exit(f"the ledger withholds {len(WITHHELD)} group shares, not the 18 of decision 25")
    out["app-ledger.tex"] = "\\newcommand{\\LedgerWithheld}{" + str(len(WITHHELD)) + "}\n"
    for name, text in out.items():
        (HERE / name).write_text(text, encoding="utf-8")
    print(f"{len(out)} appendix files written")


if __name__ == "__main__":
    main()
