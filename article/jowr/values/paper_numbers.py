"""Read the numbers of the JoWR draft from the records of v6k-v6s and write numbers.json.

Usage: python3 -I paper_numbers.py (named so that it does not shadow the standard module "numbers")
- Each key is computed from a record (path in the function) and compared with the value expected from the notes and
  outline v15 (EXPECTED). A key whose computed string differs from its expected string stops the run, and so does a
  record whose SHA-256 differs from the one the notes give.
- Values bounded by 1 (shares, AUCs, reliabilities, differences of shares) are written in APA style without a leading
  zero (".72"); ranges are written with an en dash ("--").
- The records are read only; nothing in results/ is changed.
"""
import hashlib
import math
from decimal import ROUND_HALF_UP, Decimal
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results"
SHA = {
    "v6k": "edfcce3f7c51911cd20af011ab862d458ea290e1b5e2e553b79e94bd4e553205",
    "v6l": "034c00971d82610fa7cddb85240f8c820cb29fe463e395631bb867760331f9fe",
    "v6m": "440f252bee4436f908c9ad399472295ff2ae88816435af8bf2812d5a0af9b751",
    "v6n": "738f61ad2f8c0abc3dc2efd74cc7938eed44d56d182278d7501c48c9b615c4ba",
    "v6o": "60cd29c6a3742293f4c3e8e510cfe3329ca2dddb2b06838c534540b16712e888",
    "v6p": "6073c525dbd7e163f89e0fe720c8d00aa45156d0294cce88094a3facc1d3944b",
    "v6q": "057c0c5bdf931608e424fde1b47efd214e5d338f5b152ab41ceb9f1897bd8554",
    "v6s": "d5d2e7c9a675fcb3213585a36a0ac0456d20f0776b6bff6f45df1380b23d4219",
}
SETS = ("kupa_essay", "lwp_lwp")
BASES = ("oddeven", "halves")
SHUFFLES = ("shuffled/before", "shuffled_in_halves/before")


def check_record(v, p, sha):
    """The record is the original file (its SHA-256), or, in the archive, a rebuilt record with the original content:
    the SHA-256 of the record without its provenance equals the one results/record-digests.json gives
    (tools/rebuild_records.py). Anything else stops."""
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    if got == sha:
        return
    table = RES / "record-digests.json"
    if table.exists():
        t = json.loads(table.read_text(encoding="utf-8"))["runs"].get(v, {})
        rec = json.loads(p.read_text(encoding="utf-8"))
        rec.pop("provenance", None)
        text = json.dumps(rec, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        if t.get("record_sha256") == sha and hashlib.sha256(text.encode("utf-8")).hexdigest() == t.get(
                "content_sha256"):
            print(f"{v}: a rebuilt record with the original content (its provenance differs)", file=sys.stderr)
            return
    sys.exit(f"{p}: SHA-256 {got} is not {sha}, and its content is not the original's")


def load():
    out = {}
    for v, sha in SHA.items():
        p = RES / f"prototype_{v}" / "record.json"
        check_record(v, p, sha)
        out[v] = json.loads(p.read_text(encoding="utf-8"))
    return out


def fin(x, what="value"):
    """Stop on a value that is not a finite number (NaN would pass every comparison silently)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        sys.exit(f"{what}: not a finite number: {x!r}")
    return x


def fmt(x, places=2):
    """Half-up rounding of the record's value (its repr), once. A value already rounded to places + 1 decimals with
    a final 5 is ambiguous (the unrounded value may lie on either side), so it stops the run."""
    d = Decimal(repr(abs(fin(x))))
    if d.as_tuple().exponent == -(places + 1) and d.as_tuple().digits[-1] == 5:
        sys.exit(f"{x!r} is already rounded with a final 5; read the unrounded value from the record")
    q = Decimal(1).scaleb(-places)
    return str(Decimal(repr(abs(fin(x)))).quantize(q, rounding=ROUND_HALF_UP))


def f2(x, sign=False, places=2):
    """APA style for a value bounded by 1: '.72', '1.00', '-.14'; sign=True writes '+.19'."""
    s = fmt(x, places)
    if s.startswith("0"):
        s = s[1:]
    if x < 0 and s.strip(".0"):
        return "$-$" + s
    return ("+" if sign and s.strip(".0") else "") + s


def rng(xs, sign=False, what="range", places=2):
    xs = [fin(x, what) for x in xs]
    if not xs:
        sys.exit(f"{what}: empty set of cells")
    lo, hi = min(xs), max(xs)
    a, b = f2(lo, sign, places), f2(hi, sign, places)
    return a if a == b else f"{a}--{b}"


def ci(pair, what="interval"):
    lo, hi = (fin(x, what) for x in pair)
    if lo > hi:
        sys.exit(f"{what}: interval ends in the wrong order: {pair!r}")
    return lo, hi


def share(cell):
    return cell["share_vs_own"]


def computed(R):
    k, l, m, n, o, p, q, s = (R[v] for v in ("v6k", "v6l", "v6m", "v6n", "v6o", "v6p", "v6q", "v6s"))
    N = {}
    kmc = k["main_cells"]

    def kcell(stage, st, base, row):
        return kmc[f"rule | {stage} | {st} | {base} | {row}"]

    shuffle_cells = [(st, b, r) for st in SETS for b in BASES for r in SHUFFLES]
    # v6k part 1 (rule persons of each stage; R on U's rule persons)
    N["kRshuf"] = rng([kcell("U@2000", *c)["R"][0] for c in shuffle_cells])
    N["kIshuf"] = rng([kcell("I", *c)["stage"][0] for c in shuffle_cells])
    N["kBsdshuf"] = rng([kcell("Bsd@2000", *c)["stage"][0] for c in shuffle_cells])
    krp = k["rule_persons"]  # unrounded (main_cells holds values rounded to 3 decimals)
    N["kUshuf"] = rng([share(krp[st][b][r]["U@2000"]["stage"]) for st, b, r in shuffle_cells])
    rh = k["count_1"]
    N["kUrandhalf"] = " and ".join(f2(rh[st]["oddeven"]["random_half"]["U@2000"]["share_vs_own"]) for st in SETS)
    N["kRrandhalf"] = " and ".join(f2(rh[st]["oddeven"]["random_half"]["R"]["share_vs_own"]) for st in SETS)
    ins = {st: [krp[st][b]["inside/before"] for b in BASES] for st in SETS}  # rule count, inside at the full dose
    N["kRinsK"] = rng([c["I"]["R"]["share_vs_own"] for c in ins["kupa_essay"]])
    N["kRinsL"] = rng([c["I"]["R"]["share_vs_own"] for c in ins["lwp_lwp"]])
    N["kIinsL"] = rng([c["I"]["stage"]["share_vs_own"] for c in ins["lwp_lwp"]])
    N["kBsdinsL"] = rng([c["Bsd@2000"]["stage"]["share_vs_own"] for c in ins["lwp_lwp"]])
    b_read = k["rule"]["part_1"]["B@2000"]["read"]
    N["kBread"] = rng([kmc[f"rule | B@2000 | {c}"]["stage"][0] for c in b_read])
    N["kBreadn"] = str(len(b_read))
    # v6l: R_loc on U's rule persons, the 8 shuffle cells
    N["lRlocshuf"] = rng([share(l["rule_persons"][st][b][r]["U@2000"]["R_loc"]) for st, b, r in shuffle_cells])
    lrel = l["reliability"]
    N["lBsdrelK"] = f2(lrel["kupa_essay"]["features"]["Bsd@2000[0]"]["mean_rSB"])
    N["lBsdrelL"] = f2(lrel["lwp_lwp"]["features"]["Bsd@2000[0]"]["mean_rSB"])
    N["lIwordrelL"] = f2(lrel["lwp_lwp"]["features"]["I[2]"]["mean_rSB"])
    # v6m part 0 (new persons)
    mrp = m["rule_persons"]
    N["mRlocshuf"] = rng([share(mrp[st][b][r]["U@2000"]["R_loc"]) for st, b, r in shuffle_cells])
    N["mIshuf"] = rng([share(mrp[st][b][r]["I"]["stage"]) for st, b, r in shuffle_cells])
    N["mUshuf"] = rng([share(mrp[st][b][r]["U@2000"]["stage"]) for st, b, r in shuffle_cells])
    N["mBsdshuf"] = rng([share(mrp[st][b][r]["Bsd@2000"]["stage"]) for st, b, r in shuffle_cells])
    N["mBread"] = rng([share(mrp[st][b][r]["B@2000"]["stage"]) for st, b, r in
                       (c.split(" | ") for c in b_read)])
    # v6m part 1 (copy against essay)
    lab = m["part_1"]["contrast"]["labelled"]
    rl = {c: share(lab["U@2000"][c]["R_loc"]) for c in lab["U@2000"]}
    N["mRloccopy"] = f"{f2(rl['oddeven | copy'])} and {f2(rl['halves | copy'])}"
    N["mRlocessay"] = f"{f2(rl['oddeven | essay'])} and {f2(rl['halves | essay'])}"
    if [lab["U@2000"][c]["R_loc"]["reading"]["size"] for c in ("oddeven | copy", "halves | copy", "oddeven | essay",
                                                              "halves | essay")] != ["above 0.60"] * 2 + ["undecided"] * 2:
        sys.exit("contrast gate classes changed")
    N["mIessayhalvesdiff"] = f2(lab["I"]["halves | essay"]["paired_minus_R_loc"]["difference"], sign=True)
    # v6n: reach of the gate R_loc on the persons with R (gate 0)
    g0 = n["part_1"]["gate_0"]
    cls0 = {c: g0[c]["R_loc"]["classes"] for c in g0}
    want0 = {"kupa_essay | oddeven | inside": "bbbu", "kupa_essay | halves | inside": "bbbu",
             "kupa_essay | oddeven | cross": "bbbb", "kupa_essay | halves | cross": "bbbb",
             "lwp_lwp | oddeven | inside": "buaa", "lwp_lwp | halves | inside": "buaa",
             "lwp_lwp | oddeven | cross": "bbbu", "lwp_lwp | halves | cross": "bbuu"}
    for c, w in want0.items():
        got = "".join(cls0[c][s_][0] for s_ in ("25", "50", "75", "100"))
        if got != w:
            sys.exit(f"gate classes of {c} are {got}, the text assumes {w}")
    N["nLWPin75"] = f"{f2(g0['lwp_lwp | oddeven | inside']['R_loc']['shares']['75'])} and " \
                    f"{f2(g0['lwp_lwp | halves | inside']['R_loc']['shares']['75'])}"
    N["nKUPAin100"] = rng([g0[f"kupa_essay | {b} | inside"]["R_loc"]["shares"]["100"] for b in BASES])
    N["nLWPcross100"] = f"{f2(g0['lwp_lwp | oddeven | cross']['R_loc']['shares']['100'])} and " \
                        f"{f2(g0['lwp_lwp | halves | cross']['R_loc']['shares']['100'])}"
    # v6n: cells where the gate reaches on the stage's own rule persons (LWP, inside steps)
    nrp = n["rule_persons"]
    steps = ("inside_25", "inside_50", "inside_75", "inside_100")
    gate = []
    for stage in ("Bsd@2000", "I", "B_complete@2000", "Bsd_complete@2000"):
        for b in BASES:
            for r in steps:
                c = nrp["lwp_lwp"][b][r][stage]
                if c["R_loc"]["reading"]["size"] == "above 0.60":
                    gate.append(c)
    N["nGatecells"] = str(len(gate))
    if max(ci(c["paired_minus_R_loc"]["ci"])[1] for c in gate) >= 0:
        sys.exit("a gate cell has a paired upper end at or above 0")
    if any(c["stage"]["reading"]["size"] == "above 0.60" for c in gate):
        sys.exit("a gate cell has a stage above 0.60")
    i_gate = [(b, r) for b in BASES for r in steps
              if nrp["lwp_lwp"][b][r]["I"]["R_loc"]["reading"]["size"] == "above 0.60"]
    if i_gate != [("oddeven", "inside_100"), ("halves", "inside_75"), ("halves", "inside_100")]:
        sys.exit(f"the gate cells of I are not the three the text names: {i_gate}")
    N["nIgaten"] = str(len(i_gate))
    N["nIgate"] = ", ".join(f2(share(nrp["lwp_lwp"][b][r]["I"]["stage"])) for b, r in i_gate)
    N["nIgateci"] = "; ".join(", ".join(f2(x) for x in ci(nrp["lwp_lwp"][b][r]["I"]["stage"]["share_vs_own_ci"]))
                              for b, r in i_gate)
    # border cell: LWP odd/even inside_75, R_loc class on the rule persons of each stage
    cls = {st: nrp["lwp_lwp"]["oddeven"]["inside_75"][st]["R_loc"]["reading"]["size"]
           for st in ("U@2000", "I", "B@2000", "Bsd@2000")}
    N["nBorder"] = f2(share(nrp["lwp_lwp"]["oddeven"]["inside_75"]["I"]["R_loc"]))
    if not (cls["U@2000"] == cls["I"] == "undecided" and cls["B@2000"] == cls["Bsd@2000"] == "above 0.60"):
        sys.exit(f"border cell classes changed: {cls}")
    # v6n: cells reached only by R_mean (R_loc not above, R_mean above), on the seven stages' rule persons;
    # U@2000, B@2000 and U_med are fixed by construction under inside and are left out there
    rm, rmwhere = [], set()
    for stage in ("U@2000", "B@2000", "Bsd@2000", "I", "U_med", "B_complete@2000", "Bsd_complete@2000"):
        for st in SETS:
            for b in BASES:
                for r in [x for x in nrp[st][b] if x.startswith(("inside_", "cross_"))]:
                    c = nrp[st][b][r][stage]
                    if r.startswith("inside") and stage in ("U@2000", "B@2000", "U_med"):
                        continue  # fixed by construction under inside
                    if c["R_mean"]["reading"]["size"] == "above 0.60" and \
                            c["R_loc"]["reading"]["size"] != "above 0.60":
                        rm.append(c)
                        rmwhere.add((st, b, r, stage if (st, r) == ("lwp_lwp", "inside_75") else ""))
    N["nRmeanonly"] = str(len(rm))
    named = {(s_, b_, r_) for s_, b_, r_, _ in rmwhere}
    want = {("kupa_essay", b_, "inside_100") for b_ in BASES} | {("lwp_lwp", b_, r_) for b_ in BASES
                                                                for r_ in ("cross_75", "cross_100")} \
        | {("lwp_lwp", "oddeven", "inside_75")}
    border = {x[3] for x in rmwhere if x[3]}
    if named != want or border != {"I"}:
        sys.exit(f"R_mean-only cells are not the ones the text names: {sorted(rmwhere)}")
    if max(ci(c["paired_minus_R_mean"]["ci"])[1] for c in rm) >= 0:
        sys.exit("an R_mean-only cell has a paired upper end at or above 0")
    # v6n part 2: directions over persons at step 25 (sentence-start number, both references) and count 1 at 25
    sg = n["part_2_signed_dose_rows"]
    dirs = [sg[st][f"{fam}_25"][ref]["0"]["share_negative"]
            for st in SETS for fam in ("inside", "cross") for ref in ("R_loc", "R_mean")]
    N["nDir25"] = rng(dirs)
    # v0.5: the direction is largely fixed by the swaps: R_loc's sentence-start number rose for few writers under inside
    # and for none under cross
    N["nRoseInside"] = rng([sg[st][f"inside_{x}"]["R_loc"]["0"]["share_positive"] for st in SETS for x in (25, 50, 75, 100)])
    if any(fin(sg[st][f"cross_{x}"][ref]["0"]["share_positive"]) != 0 for st in SETS for x in (25, 50, 75, 100)
           for ref in ("R_loc", "R_mean")):
        sys.exit("a writer's sentence-start number rose under cross")
    c25 = [g0[f"{st} | {b} | {fam}"][ref]["shares"]["25"]
           for st in SETS for b in BASES for fam in ("inside", "cross") for ref in ("R_loc", "R_mean")]
    N["nCount25"] = rng(c25)
    iw = [sg[st][f"inside_{x}"]["I"]["2"]["share_positive"] for st in SETS for x in (25, 50, 75, 100)]
    N["nIwordup"] = rng(iw)
    ic = [share(nrp[st][b][f"inside_{x}"]["I"]["stage"]) for st in SETS for b in BASES for x in (25, 50, 75, 100)]
    N["nIinside"] = rng(ic)
    # v6n part 3: reliability on v6m's persons
    rel = n["part_3"]["reliability"]
    N["nBrel"] = rng([rel[st]["features"]["B@2000[0]"]["mean_rSB"] for st in ("kupa_essay", "kupa_copy", "lwp_lwp")])
    N["nIrelL"] = ", ".join(f2(rel["lwp_lwp"]["features"][f"I[{i}]"]["mean_rSB"]) for i in range(3))
    N["nRlocsentK"] = f2(rel["kupa_essay"]["features"]["R[0]"]["mean_rSB"])
    # v6o: own distance lower in the high group than in the low group
    gs = o["main_cells"]["groups_n_s"]
    lower = sum(gs[c]["base_by_group"][ref]["high"] < gs[c]["base_by_group"][ref]["low"]
                for c in gs for ref in ("R_loc", "R_mean"))
    N["oOwnlower"] = f"{lower} of {2 * len(gs)}"
    # v6p: I by group at LWP odd/even inside_100
    pg = p["groups"]["n_s"]["lwp_lwp"]["oddeven"]["I"]["inside_100"]["groups"]  # unrounded
    N["pIlow"] = f2(pg["low"]["share_vs_own"])
    N["pImidhigh"] = f"{f2(pg['mid']['share_vs_own'])} and {f2(pg['high']['share_vs_own'])}"
    # v6q: I[2] in the three gate cells and by group
    qg = q["main_cells"]["gate_cells"]
    cells = ("lwp_lwp oddeven inside_100", "lwp_lwp halves inside_75", "lwp_lwp halves inside_100")
    qc = [q["cells"]["lwp_lwp"][c.split()[1]][c.split()[2]]["rule"]["I[2]"] for c in cells]  # unrounded
    if any(x["class"] != "undecided" for x in qc):
        sys.exit("I[2] in a gate cell is not undecided")
    N["qI2"] = ", ".join(f2(x["share_vs_own"]) for x in qc)
    N["qI2lo"] = rng([ci(x["share_vs_own_ci"], "I[2]")[0] for x in qc])
    gains = [fin(qg[c]["I[2]_minus_I"][0]) for c in cells]
    if min(gains) <= 0:
        sys.exit("I[2] - I is not positive in every gate cell")
    N["qGain"] = rng(gains)  # the text writes "rose ... by" the range
    if any(fin(qg[c]["I[2]_minus_R_loc"][2]) >= 0 for c in cells):
        sys.exit("an I[2] - R_loc upper end is at or above 0")
    qgr = q["main_cells"]["primary_groups_n_s"]["lwp_lwp oddeven inside_100"]["groups"]
    if [qgr[g]["I[2]"][1] for g in ("mid", "high")] != ["above", "above"]:
        sys.exit("I[2] groups are not above")
    if [pg[g]["class"] for g in ("low", "mid", "high")] != ["below", "undecided", "undecided"] or \
            [pg[g]["state"] for g in ("mid", "high")] != ["gate", "gate"] or pg["low"]["state"] == "gate":
        sys.exit("v6p group classes or states changed")
    N["qI2midhigh"] = f"{f2(qgr['mid']['I[2]'][0])} and {f2(qgr['high']['I[2]'][0])}"
    # v6s: one minus agreement
    sm = s["summary"]
    allv = []
    for part in ("primary", "shuffles"):
        for cellname, stages in s["main_cells"][part].items():
            for stage, v in stages.items():
                if isinstance(v, dict):
                    allv.append((cellname, stage, v["1-p_o"]))
    N["sWritten"] = str(len(allv))
    if sum(sm[x]["written"] for x in ("primary", "shuffles")) != len(allv):
        sys.exit("v6s: written cells in the summary differ from the cells listed")
    N["sCounted"] = str(sum(sm[x]["written"] + sm[x]["withheld_narrows"] + sm[x]["withheld_fewer_than_5"] + sm[x]["no_count"]
                            for x in ("primary", "shuffles")))
    nfixed = sum(sm[x]["fixed"] for x in ("primary", "shuffles"))
    ncells = sum(len(s["main_cells"][x][c]) for x in ("primary", "shuffles") for c in s["main_cells"][x])
    if int(N["sCounted"]) + nfixed != ncells:
        sys.exit(f"v6s: counted {N['sCounted']} + fixed {nfixed} is not the {ncells} cells listed")
    for part in ("primary", "shuffles"):
        for cellname, stages in s["main_cells"][part].items():
            for stage, v in stages.items():
                if isinstance(v, dict) and v["class"]["oddeven"] != v["class"]["halves"]:
                    sys.exit(f"v6s: classes differ between splits in {cellname} {stage}")
    N["sRange"] = rng([x[2] for x in allv])
    N["sMedPrimary"] = f2(sm["primary"]["one_minus_p_o"]["median"])
    N["sRefsKUPA"] = rng([x[2] for x in allv if x[0].startswith("kupa_essay") and x[1] in ("R_loc", "R_mean")
                          and "shuffled" not in x[0]])
    N["sIKUPA"] = rng([x[2] for x in allv if x[0].startswith("kupa_essay") and x[1] == "I"])
    N["sILWPshuf"] = " and ".join(f2(x[2]) for x in allv if x[0].startswith("lwp_lwp") and x[1] == "I"
                                  and "shuffled" in x[0])
    # v6k count 2 (identification, KUPA copy -> essay)
    c2 = k["count_2"]["kupa_copy_to_essay"]
    for key, st in (("S0", "S0"), ("R", "R"), ("U", "U@2000"), ("B", "B@2000"), ("Bsd", "Bsd@2000"), ("I", "I")):
        N[f"kAuc{key}"] = f2(c2[st]["main"]["rank_auc"])
        N[f"kAuc{key}ci"] = ", ".join(f2(x) for x in ci(c2[st]["main"]["rank_auc_ci"], f"{st} AUC"))
        lo = ci(c2[st]["main"]["rank_auc_ci"])[0]
        hi = ci(c2[st]["main"]["rank_auc_ci"])[1]
        N[f"kAuc{key}cls"] = "above" if lo > 0.6 else ("below" if hi < 0.6 else "undecided")
    for key, st in (("U", "U@2000"), ("B", "B@2000"), ("Bsd", "Bsd@2000"), ("I", "I")):
        rs = c2[st]["registration_scale"]
        lo_, hi_ = ci(rs["rank_auc_ci"], f"{st} registration AUC")
        if not lo_ <= fin(rs["rank_auc"]) <= hi_:
            sys.exit(f"{st}: registration AUC outside its interval")
        N[f"kAuc{key}reg"] = f"{f2(rs['rank_auc'])} [{f2(lo_)}, {f2(hi_)}]"
    # the text says that above-chance matching of B and Bsd depends on the standardization: their registration-side
    # lower ends lie at or below .50, those of U and I above it
    # the text says the registration side lowers U, B and Bsd and leaves I about the same (within .02)
    if not all(fin(c2[s_]["registration_scale"]["rank_auc"]) < fin(c2[s_]["main"]["rank_auc"]) for s_ in ("U@2000", "B@2000", "Bsd@2000")) \
            or abs(fin(c2["I"]["registration_scale"]["rank_auc"]) - fin(c2["I"]["main"]["rank_auc"])) > 0.02:
        sys.exit("registration-side AUCs no longer lower U, B and Bsd and leave I about the same")
    if not all(ci(c2[s_]["registration_scale"]["rank_auc_ci"])[0] <= 0.5 for s_ in ("B@2000", "Bsd@2000")) or \
            not all(ci(c2[s_]["registration_scale"]["rank_auc_ci"])[0] > 0.5 for s_ in ("U@2000", "I")):
        sys.exit("registration-side lower ends no longer split B and Bsd from U and I at .50")
    N["kAucB500"] = f2(c2["B@500"]["main"]["rank_auc"])
    tops = [c2[st]["main"]["top1"] for st in ("U@2000", "B@2000", "Bsd@2000", "I")]
    N["kTop1"] = rng(tops, what="top-1", places=3)
    N["kTop1S0"] = f2(c2["S0"]["main"]["top1"], places=3)
    N["kTop1R"] = f2(c2["R"]["main"]["top1"], places=3)
    N["kTop1all"] = f2(max(fin(c2[st]["main"]["top1"]) for st in c2 if st not in ("paired", "S0", "S1", "R")))
    # thresholds: cells (of 12 main cells) whose class is the same at 2 s, 1 s and 500 ms at once, per stage U, B, Bsd
    agree = []
    for stg, kst in (("U", "U@2000"), ("B", "B@2000"), ("Bsd", "Bsd@2000")):
        base = {f"{st} | {b} | {r}": kcell(kst, st, b, r)["stage"] for st in SETS for b in BASES
                for r in ("inside/before",) + SHUFFLES}
        cl = {thr: k["rule"]["part_2"][stg][thr]["classes"] for thr in ("1000", "500")}
        same = 0
        for c, (v, iv) in base.items():
            lo, hi = ci(iv, c)
            if 0.6 in (lo, hi):
                sys.exit(f"{c}: an interval end of the 2-s cell equals 0.60 at the record's rounding")
            c2000 = "above" if lo > 0.6 else ("below" if hi < 0.6 else "undecided")
            same += cl["1000"][c] == c2000 == cl["500"][c]
        agree.append(same)
    N["kThragree"] = f"{min(agree)}--{max(agree)}"
    # the same agreement over the cells that can change: the cells fixed by construction (conditions of v6k,
    # rule.part_1_pause_location: U inside 4; B inside 4 and shuffled_in_halves under halves 2) agree trivially
    fixed = {"U": lambda b, r: r == "inside/before",
             "B": lambda b, r: r == "inside/before" or (b == "halves" and r == "shuffled_in_halves/before"),
             "Bsd": lambda b, r: False}
    mov = {}
    for stg, kst in (("U", "U@2000"), ("B", "B@2000"), ("Bsd", "Bsd@2000")):
        cl = {thr: k["rule"]["part_2"][stg][thr]["classes"] for thr in ("1000", "500")}
        n_, same = 0, 0
        for st in SETS:
            for b in BASES:
                for r in ("inside/before",) + SHUFFLES:
                    if fixed[stg](b, r):
                        if fin(k["rule_persons"][st][b][r][kst]["stage"]["median_change"]) != 0:
                            sys.exit(f"{stg} {st} {b} {r}: a cell taken as fixed by construction has a change")
                        continue
                    c = f"{st} | {b} | {r}"
                    c2000 = {"above 0.60": "above", "below 0.60": "below", "undecided": "undecided"}[
                        k["rule_persons"][st][b][r][kst]["stage"]["reading"]["size"]]
                    n_ += 1
                    same += cl["1000"][c] == c2000 == cl["500"][c]
        mov[stg] = (same, n_)
    N["kThrmovU"], N["kThrmovB"], N["kThrmovBsd"] = (f"{a} of {b}" for a, b in (mov[x] for x in ("U", "B", "Bsd")))
    N["kThrmov"] = f"{sum(a for a, _ in mov.values())} of {sum(b for _, b in mov.values())}"
    N["kThrchanged"] = f"{sum(b - a for a, b in mov.values())} of {sum(b for _, b in mov.values())}"
    # writers (v6k: rule persons drawn from the fixed orders; v6m: second count)
    w = k["writer"]
    N["kNkupa"] = str(w["kupa_copy"]["count_persons"] + w["kupa_copy"]["left_out_fewer_than_8_or_part_without_gap"])
    N["kNessay"] = str(w["kupa_essay"]["count_persons"])
    N["kNlwp"] = str(w["lwp_lwp"]["count_persons"] + w["lwp_lwp"]["left_out_fewer_than_8_or_part_without_gap"])
    N["kNlwpcount"] = str(w["lwp_lwp"]["count_persons"])
    N["mNessay"] = str(m["tallies"]["kupa_essay"]["count_persons"])
    N["mNlwp"] = str(m["tallies"]["lwp_lwp"]["count_persons"])
    # added for v0.3
    # writers per rule cell (U, B, Bsd, I on their rule persons; shuffles and inside, both sets and splits)
    rows = ("inside/before",) + SHUFFLES
    for key, rec in (("kNrule", k), ("mNrule", m)):
        ns = [rec["rule_persons"][st][b][r][stg]["stage"]["persons"] for st in SETS for b in BASES for r in rows
              for stg in ("U@2000", "B@2000", "Bsd@2000", "I")]
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 5 for x in ns):
            sys.exit(f"{key}: a rule cell has no valid count of writers")
        N[key] = f"{min(ns)}--{max(ns)}"
    # Table 4 gives the classes of the rule count's eight shuffle cells per summary (B: its cells that can move)
    # (classes from the unrounded rule_persons intervals, which must agree with the record's own reading)
    def klass(stage, cells):
        out = {"above": 0, "below": 0, "undecided": 0}
        for st, b, r in cells:
            c = krp[st][b][r][stage]["stage"]
            lo, hi = ci(c["share_vs_own_ci"], f"{stage} {st} {b} {r}")
            if 0.6 in (lo, hi):
                sys.exit(f"{stage} {st} {b} {r}: an interval end equals 0.60")
            mine = "above" if lo > 0.6 else ("below" if hi < 0.6 else "undecided")
            rec_ = {"above 0.60": "above", "below 0.60": "below", "undecided": "undecided"}[c["reading"]["size"]]
            if mine != rec_:
                sys.exit(f"{stage} {st} {b} {r}: class {mine} differs from the record's {rec_}")
            out[mine] += 1
        return out
    b_cells = [tuple(c.split(" | ")) for c in b_read]
    b_want = [c for c in shuffle_cells if not (c[1] == "halves" and c[2] == "shuffled_in_halves/before")]
    if len(set(b_cells)) != len(b_cells) or set(b_cells) != set(b_want):
        sys.exit(f"B's cells that can move are not the shuffle cells less the fixed ones: {b_cells}")
    want_cls = {"I": ({"above": 8, "below": 0, "undecided": 0}, shuffle_cells),
                "U@2000": ({"above": 0, "below": 8, "undecided": 0}, shuffle_cells),
                "Bsd@2000": ({"above": 0, "below": 4, "undecided": 4}, shuffle_cells),
                "B@2000": ({"above": 0, "below": 6, "undecided": 0}, b_cells)}
    for stg, (w, cells) in want_cls.items():
        if klass(stg, cells) != w:
            sys.exit(f"Table 4 classes of {stg} changed: {klass(stg, cells)}")
    # random_half (odd/even): Bsd beside U and R
    N["kBsdrandhalf"] = " and ".join(f2(rh[st]["oddeven"]["random_half"]["Bsd@2000"]["share_vs_own"]) for st in SETS)
    # rule count, R inside in KUPA essays: one cell below, one undecided (the text names both classes)
    if [ins["kupa_essay"][i]["I"]["R"]["reading"]["size"] for i in (0, 1)] != ["below 0.60", "undecided"]:
        sys.exit("classes of R inside in KUPA essays changed")
    # v6p: LWP first-second half, I by group at inside_75 and inside_100, the groups read through the gate
    ph = p["groups"]["n_s"]["lwp_lwp"]["halves"]["I"]
    hg = [ph[r]["groups"][g] for r in ("inside_75", "inside_100") for g in ("low", "mid", "high")
          if ph[r]["groups"][g]["state"] == "gate"]
    if len(hg) != 5 or any(x["class"] != "below" for x in hg):
        sys.exit("v6p: the first-second half gate groups of I are not the five below")
    N["pIhalvesgate"] = rng([x["share_vs_own"] for x in hg])
    # the text adds that the highest of them is undecided when inside at the full dose is counted with the other key
    # (row inside/before: the same swaps as inside_100, another bootstrap key)
    top = max(hg, key=lambda x: x["share_vs_own"])
    if top is not ph["inside_100"]["groups"]["high"] or ph["inside/before"]["groups"]["high"]["class"] != "undecided" \
            or ph["inside/before"]["groups"]["high"]["share_vs_own"] != top["share_vs_own"]:
        sys.exit("v6p: the highest first-second half gate group is no longer undecided under inside/before")
    # the text gives the interval width for .29-.45
    if not all(0.285 <= fin(share(nrp["lwp_lwp"][b][r]["I"]["stage"])) < 0.455 for b, r in i_gate):
        sys.exit("nCIwidth: a gate cell's share lies outside .29-.45")
    # width of the 95% intervals of I in the three gate cells (writers about 175-200)
    N["nCIwidth"] = rng([ci(nrp["lwp_lwp"][b][r]["I"]["stage"]["share_vs_own_ci"])[1]
                         - ci(nrp["lwp_lwp"][b][r]["I"]["stage"]["share_vs_own_ci"])[0] for b, r in i_gate])
    # added for v0.4
    # share of bursts cut by a part's edge (v6k writer, text at 2 s): medians over persons, KUPA essays and LWP
    ed = k["writer"]
    N["kEdgeOE"] = rng([ed[st]["edge_share_of_bursts_2000_text"]["oddeven"][pt]["median"] for st in SETS for pt in "AB"])
    N["kEdgeHalves"] = rng([ed[st]["edge_share_of_bursts_2000_text"]["halves"][pt]["median"] for st in SETS for pt in "AB"])
    # second count, LWP inside at the full dose (v6m part 0, gate R_loc): I and Bsd, all below; R_loc above
    mi = [mrp["lwp_lwp"][b]["inside/before"] for b in BASES]
    def cls_ci(c):
        lo, hi = ci(c["share_vs_own_ci"])
        if 0.6 in (lo, hi):
            sys.exit("an interval end equals 0.60")
        return "above 0.60" if lo > 0.6 else ("below 0.60" if hi < 0.6 else "undecided")
    if any(cls_ci(c[s_]["stage"]) != "below 0.60" or c[s_]["stage"]["reading"]["size"] != "below 0.60"
           for c in mi for s_ in ("I", "Bsd@2000")) or \
            any(cls_ci(c[s_]["R_loc"]) != "above 0.60" or c[s_]["R_loc"]["reading"]["size"] != "above 0.60"
                for c in mi for s_ in ("I", "Bsd@2000")):
        sys.exit("v6m: LWP inside classes changed")
    if any(cls_ci(c[s_]["R_loc"]) != "above 0.60" for c in mi for s_ in ("U@2000", "B@2000")) or \
            any(cls_ci(mrp["kupa_essay"][b]["inside/before"][s_]["R_loc"]) == "above 0.60"
                for b in BASES for s_ in ("U@2000", "B@2000", "Bsd@2000", "I")):
        sys.exit("v6m: R_loc at inside is not above in LWP for U and B, or is above in KUPA essays")
    # copy base: B and Bsd below the reference (paired upper end below 0), as the text says
    if any(ci(lab[s_][c]["paired_minus_R_loc"]["ci"])[1] >= 0 for s_ in ("B@2000", "Bsd@2000")
           for c in ("oddeven | copy", "halves | copy")):
        sys.exit("v6m: B or Bsd is not below R_loc against the copy's variation")
    N["mIinsL"] = rng([c["I"]["stage"]["share_vs_own"] for c in mi])
    N["mBsdinsL"] = rng([c["Bsd@2000"]["stage"]["share_vs_own"] for c in mi])
    # reliability of I's three components in LWP on the rule count's writers (v6l)
    N["lIrelK"] = ", ".join(f2(lrel["kupa_essay"]["features"][f"I[{i}]"]["mean_rSB"]) for i in range(3))
    N["lIrelL"] = ", ".join(f2(lrel["lwp_lwp"]["features"][f"I[{i}]"]["mean_rSB"]) for i in range(3))
    # copy against essay, essay's own variation: every summary above R_loc's share; paired upper/lower ends
    ess = [lab[st][c] for st in ("U@2000", "B@2000", "Bsd@2000", "I") for c in ("oddeven | essay", "halves | essay")]
    if any(fin(x["stage"]["share_vs_own"], "contrast stage") <= fin(x["R_loc"]["share_vs_own"], "contrast R_loc") for x in ess):
        sys.exit("v6m: a summary is not above R_loc against the essay's variation")
    N["mEssaySum"] = rng([x["stage"]["share_vs_own"] for x in ess])
    pos = sum(ci(x["paired_minus_R_loc"]["ci"])[0] > 0 for x in ess)
    N["mEssayPos"] = f"{pos} of {len(ess)}"
    N["nIgatecibr"] = ", ".join("[" + ", ".join(f2(x) for x in ci(nrp["lwp_lwp"][b][r]["I"]["stage"]["share_vs_own_ci"])) + "]"
                                for b, r in i_gate)
    N["nRlocsentL"] = f2(rel["lwp_lwp"]["features"]["R[0]"]["mean_rSB"])
    # added for v0.5: uncorrected split-half r beside the corrected values (v6l, v6n part 3)
    N["lBsdrawK"] = f2(lrel["kupa_essay"]["features"]["Bsd@2000[0]"]["mean_r"])
    N["lBsdrawL"] = f2(lrel["lwp_lwp"]["features"]["Bsd@2000[0]"]["mean_r"])
    N["lIrawL"] = ", ".join(f2(lrel["lwp_lwp"]["features"][f"I[{i}]"]["mean_r"]) for i in range(3))
    N["nIrawL"] = ", ".join(f2(rel["lwp_lwp"]["features"][f"I[{i}]"]["mean_r"]) for i in range(3))
    for st_, rr in (("v6l", lrel), ("v6n", rel)):
        for set_ in ("kupa_essay", "lwp_lwp"):
            for f_, x in rr[set_]["features"].items():
                if fin(x["mean_r"]) > fin(x["mean_rSB"]) + 1e-12 and x["mean_r"] > 0:
                    sys.exit(f"{st_} {set_} {f_}: uncorrected r above the corrected value")
    # what moved at LWP inside_75 (v6n tallies, medians over persons): sentence-initial gaps swapped, share of time
    d75 = n["tallies"]["lwp_lwp"]["doses"]["inside_75"]
    sw = fin(d75["swaps"]["median"])
    if sw != int(sw):
        sys.exit("the median number of swaps is not whole")
    N["nSwap75L"] = str(int(sw))
    pct = Decimal(repr(fin(d75["time_moved_share"]["median"]))).scaleb(2)  # exact shift, then the one rounding of fmt
    N["nTime75L"] = fmt(float(pct), 1) + "\\%"
    # sentence starts per part (v6o, groups by n_s under odd/even): group medians, and LWP writers above KUPA's upper cut
    def meds(st):
        gm = o["groups"]["n_s"][st]["oddeven"]["covariate_median_by_group"]
        if gm != o["groups"]["n_s"][st]["halves"]["covariate_median_by_group"]:
            sys.exit(f"v6o: group medians of {st} differ between the splits")
        v = [fin(gm[g]) for g in ("low", "mid", "high")]
        if v != sorted(v):
            sys.exit(f"v6o: group medians of {st} are not increasing")
        s_ = [f"{x:g}" if x != int(x) else f"{x:.1f}" for x in v]
        return f"{s_[0]}, {s_[1]} and {s_[2]}"
    N["oMedL"], N["oMedK"] = meds("lwp_lwp"), meds("kupa_essay")
    # the band family itself is withheld in v6o (its shares would let a set of 3 LWP writers be derived); only its
    # counts of writers are read here: counts are not under the rule, v6o's notes publish them (reading (d)), and the
    # text gives the high band only (159 of 181), from which the low band (3) cannot be told
    bd = o["bands"]["n_s"]["oddeven"]
    if bd.get("written") is not False:
        sys.exit("v6o: the band family is now written; the comment above no longer describes it")
    if bd["cuts_from_kupa_essay"] != o["groups"]["n_s"]["kupa_essay"]["oddeven"]["cuts"]:
        sys.exit("v6o: the bands do not use KUPA essays' cuts")
    lp = bd["persons"]["lwp"]
    if sum(lp.values()) != sum(o["groups"]["n_s"]["lwp_lwp"]["oddeven"]["persons"].values()):
        sys.exit("v6o: LWP band persons do not add up to LWP's group persons")
    N["oAboveK"] = f"{lp['high']} of {sum(lp.values())}"
    hb = ph["inside/before"]["groups"]["high"]
    N["pIhighrule"] = f2(hb["share_vs_own"]) + ", interval [" + ", ".join(f2(x) for x in ci(hb["share_vs_own_ci"])) + "]"
    # added for v0.6
    # what moved at KUPA inside_100, where the gate did not reach (v6n tallies): beside LWP inside_75 (R1 4)
    d100 = n["tallies"]["kupa_essay"]["doses"]["inside_100"]
    sw = fin(d100["swaps"]["median"])
    if sw != int(sw):
        sys.exit("the median number of swaps is not whole")
    N["nSwap100K"] = str(int(sw))
    N["nTime100K"] = fmt(float(Decimal(repr(fin(d100["time_moved_share"]["median"]))).scaleb(2)), 1) + "\\%"
    # writers of I in its three gate cells (the text gave "about 175--200"; audit 2-4)
    N["nIgatepersons"] = f"{min(nrp['lwp_lwp'][b][r]['I']['stage']['persons'] for b, r in i_gate)}--" \
                         f"{max(nrp['lwp_lwp'][b][r]['I']['stage']['persons'] for b, r in i_gate)}"
    # split dependence over the primary cells only (inside 75 and 100, cross 100), beside its median
    prim = [v["1-p_o"] for stages in s["main_cells"]["primary"].values() for v in stages.values() if isinstance(v, dict)]
    N["sRangePrimary"] = rng(prim)
    sp_ = sm["primary"]["one_minus_p_o"]
    if len(prim) != sm["primary"]["written"] or fin(sp_["min"]) != round(min(prim), 3) or fin(sp_["max"]) != round(max(prim), 3):
        sys.exit("v6s: the primary cells listed differ from the record's summary of them (count, min, max)")
    # rank AUC of the four summaries under the main standardization (abstract, (v), conclusion)
    N["kAucSum"] = rng([c2[st]["main"]["rank_auc"] for st in ("U@2000", "B@2000", "Bsd@2000", "I")])
    # added for v0.7
    def civ(c, what):
        """'.xx [.xx, .xx]': share and its 95% interval. Each end is given with the fewest decimals (2 to 4) at which it
        does not round to one of the margins the text reads (.55, .60, .65), so that its class at each margin can be read."""
        lo, hi = ci(c["share_vs_own_ci"], what)
        if not 0 <= lo <= fin(c["share_vs_own"]) <= hi <= 1:
            sys.exit(f"{what}: share or interval outside [0, 1] or out of order")
        def end(x):
            for pl in (2, 3, 4):
                if fmt(x, pl) not in ("0." + "55".ljust(pl, "0"), "0." + "60".ljust(pl, "0"), "0." + "65".ljust(pl, "0")):
                    return f2(x, places=pl)
            sys.exit(f"{what}: an interval end {x!r} lies within 0.00005 of a margin")
        return f"{f2(c['share_vs_own'])} [{end(lo)}, {end(hi)}]"
    c1 = n["count_1"]
    if any(fin(c1["lwp_lwp"][b]["inside_75"]["R_loc"]["share_vs_own"]) != fin(g0[f"lwp_lwp | {b} | inside"]["R_loc"]["shares"]["75"])
           or fin(c1["lwp_lwp"][b]["cross_100"]["R_loc"]["share_vs_own"]) != fin(g0[f"lwp_lwp | {b} | cross"]["R_loc"]["shares"]["100"])
           for b in BASES):
        sys.exit("count_1's R_loc differs from gate_0's")
    N["nLWPin75ci"] = " and ".join(civ(c1["lwp_lwp"][b]["inside_75"]["R_loc"], f"in75 {b}") for b in BASES)
    N["nLWPcross100ci"] = " and ".join(civ(c1["lwp_lwp"][b]["cross_100"]["R_loc"], f"cr100 {b}") for b in BASES)
    # LWP first-second half, I in every group of the inside cells read through the gate (v6p), all groups
    ag = [ph[r]["groups"][g] for r in ("inside_75", "inside_100") for g in ("low", "mid", "high")]
    for x in ag:  # class recomputed from the interval and checked against the record's
        lo_, hi_ = ci(x["share_vs_own_ci"], "v6p group")
        if not hi_ < 0.6 or x["class"] != "below":
            sys.exit("v6p: a group of the first-second half inside cells is not below")
    N["pIhalvesall"] = rng([x["share_vs_own"] for x in ag])
    # v6q: the between-word mean in the same groups read through the gate: all undecided
    qh = [q["main_cells"]["primary_groups_n_s"][f"lwp_lwp halves {r}"]["groups"][g]
          for r in ("inside_75", "inside_100") for g in ("low", "mid", "high")]
    states = {(r, g): q["main_cells"]["primary_groups_n_s"][f"lwp_lwp halves {r}"]["groups"][g]["state"]
              for r in ("inside_75", "inside_100") for g in ("low", "mid", "high")}
    if {k_ for k_, v in states.items() if v == "gate"} != {("inside_75", "mid"), ("inside_75", "high"),
                                                         ("inside_100", "low"), ("inside_100", "mid"), ("inside_100", "high")}:
        sys.exit(f"v6q: the groups read through the gate changed: {states}")
    qh = [x for x in qh if x["state"] == "gate"]
    if len(qh) != 5 or any(x["I[2]"][1] != "undecided" for x in qh):
        sys.exit("v6q: the first-second half gate groups of I[2] are not the five undecided")
    N["qI2halves"] = rng([x["I[2]"][0] for x in qh])
    # v6o: bands of n_s_min (odd-even, written) - LWP minus KUPA essays for R_loc, high band and all writers
    bm = o["bands"]["n_s_min"]["oddeven"]
    if bm.get("written") is not True:
        sys.exit("v6o: the n_s_min bands are not written")
    def dci(x, key):
        """LWP minus KUPA essays with its interval; 'whole' holds 'difference', a band 'lwp_minus_kupa' (= lwp - kupa)."""
        lo, hi = ci(x["ci"], "band")
        d = fin(x[key])
        if key == "lwp_minus_kupa" and abs(d - (fin(x["lwp"]) - fin(x["kupa_essay"]))) > 1e-12:
            sys.exit("v6o: a band's difference is not LWP minus KUPA essays")
        if not lo <= d <= hi:
            sys.exit("v6o: a difference outside its interval")
        return f"{f2(d, sign=True)} [{f2(lo)}, {f2(hi)}]"
    cb = bm["cells"]["R_loc"]
    N["oMinIn100"] = dci(cb["inside_100"]["whole"], "difference")
    N["oMinHighIn100"] = dci(cb["inside_100"]["bands"]["high"], "lwp_minus_kupa")
    N["oMinHighCr100"] = dci(cb["cross_100"]["bands"]["high"], "lwp_minus_kupa")
    # Appendix E: under first-second half a band had fewer than five LWP writers
    if min(o["bands"]["n_s_min"]["halves"]["persons"]["lwp"].values()) >= 5:
        sys.exit("v6o: no first-second half band has fewer than five LWP writers")
    mc = bm["median_covariate"]["high"]
    N["oMinHighMed"] = f"{mc['kupa_essay']:g} and {mc['lwp']:g}"
    N["oMinHighN"] = f"{bm['persons']['kupa_essay']['high']} and {bm['persons']['lwp']['high']}"
    # v6k writer-specific threshold: records with two components, and starts that reached the iteration limit
    wk = k["writer"]
    for st in SETS:
        t_ = wk[st]["theta_w_text"]
        if not set(t_["k"]) <= {"1", "2", "3"} or any(isinstance(v, bool) or not isinstance(v, int) or v < 0
                                                    for v in list(t_["k"].values()) + [t_["starts_reaching_max_iterations"]]):
            sys.exit("v6k: the fit counts are not non-negative integers over k = 1-3")
    N["kFitK2"] = " and ".join(f"{wk[st]['theta_w_text']['k']['2']} of {wk[st]['count_persons']}" for st in SETS)
    if any(sum(wk[st]["theta_w_text"]["k"].values()) != wk[st]["count_persons"] for st in SETS):
        sys.exit("v6k: the chosen k do not add up to the writers counted")
    N["kFitMaxIter"] = " and ".join(str(wk[st]["theta_w_text"]["starts_reaching_max_iterations"]) for st in SETS)
    return N


def figure2(R, base="oddeven"):
    """pgfplots source of Figure 2 (and of its first-second half version): shares by dose with 95% intervals (v6n). R_loc on the writers with
    R (count_1, whose share equals part_1.gate_0's); I and Bsd on their rule writers. R_mean is in Appendix E."""
    n = R["v6n"]
    g0, nrp, c1 = n["part_1"]["gate_0"], n["rule_persons"], n["count_1"]
    panels = []
    for st, name in (("lwp_lwp", "LWP"), ("kupa_essay", "KUPA essays")):
        for fam in ("inside", "cross"):
            lines = []
            series = [("black,thick,mark=*", [c1[st][base][f"{fam}_{x}"]["R_loc"] for x in (25, 50, 75, 100)])]
            for x in (25, 50, 75, 100):
                if c1[st][base][f"{fam}_{x}"]["R_loc"]["share_vs_own"] != g0[f"{st} | {base} | {fam}"]["R_loc"]["shares"][str(x)]:
                    sys.exit("figure 2: count_1's R_loc share differs from gate_0's")
            for stage, style in (("I", "gray,thick,mark=square*"), ("Bsd@2000", "gray,mark=triangle")):
                series.append((style, [nrp[st][base][f"{fam}_{x}"][stage]["stage"] for x in (25, 50, 75, 100)]))
            for style, cells in series:
                pts = []
                for x, c in zip((25, 50, 75, 100), cells):
                    v = fin(c["share_vs_own"])
                    lo, hi = ci(c["share_vs_own_ci"])
                    if not lo <= v <= hi:
                        sys.exit(f"figure 2: share {v} outside its interval [{lo}, {hi}]")
                    pts.append(f"({x},{v:.3f}) -= (0,{v - lo:.3f}) += (0,{hi - v:.3f})")
                lines.append(f"\\addplot[{style},error bars/.cd,y dir=both,y explicit] coordinates {{{' '.join(pts)}}};")
            panels.append((f"{name}, {fam}", lines))
    out = ["\\begin{tikzpicture}", "\\begin{groupplot}[group style={group size=2 by 2,horizontal sep=14mm,vertical sep=24mm},"
           "width=62mm,height=46mm,xmin=20,xmax=105,ymin=0,ymax=1,xtick={25,50,75,100},"
           "ytick={0,.2,.4,.6,.8,1},yticklabel style={/pgf/number format/fixed},label style={font=\\footnotesize},"
           "tick label style={font=\\footnotesize},title style={font=\\footnotesize},"
           "xlabel={Dose (\\% of eligible swaps)},ylabel={Share of writers}]"]
    for i, (title, lines) in enumerate(panels):
        out.append(f"\\nextgroupplot[title={{{title}}}" + (",legend to name=doselegend,legend columns=3" if i == 0 else "") + "]")
        out += lines
        out.append("\\addplot[dashed,domain=20:105] {0.6};")
        if i == 0:
            out.append("\\legend{Reference $R_{loc}$,Composite $I$,$B_{sd}$}")
    out += ["\\end{groupplot}", "\\end{tikzpicture}", "", "\\ref{doselegend}"]
    if base != "oddeven":
        out = [x.replace("doselegend", f"doselegend{base}") for x in out]
    return "\n".join(out) + "\n"


# The strings printed by the article, per key. Each was checked against the records before its version of the draft.
EXPECTED = {
    "kRshuf": ".89--.98", "kIshuf": ".72--.97", "kBsdshuf": ".39--.64", "kBread": ".25--.36", "kBreadn": "6",
    "lRlocshuf": ".87--.97", "lBsdrelK": ".72", "lBsdrelL": ".86", "lIwordrelL": ".35",
    "mRlocshuf": ".87--.97", "mIshuf": ".80--.97", "mBsdshuf": ".36--.62", "mBread": ".25--.32",
    "mRloccopy": ".84 and .83", "mRlocessay": ".59 and .58", "mIessayhalvesdiff": "+.12",
    "nLWPin75": ".67 and .72", "nKUPAin100": ".63", "nLWPcross100": ".59 and .64",
    "nGatecells": "15", "nIgaten": "3", "nIgate": ".45, .29, .37", "nBorder": ".67",
    "nRmeanonly": "37", "nDir25": ".80--.99", "nCount25": ".06--.28", "nIwordup": ".81--.97",
    "nIinside": ".01--.45", "nBrel": ".85--.90", "nIrelL": ".13, .05, .59", "nRlocsentK": ".54",
    "oOwnlower": "8 of 8", "pIlow": ".30", "pImidhigh": ".53 and .55",
    "qI2": ".64, .55, .62", "qI2lo": ".47--.57", "qGain": ".19--.26", "qI2midhigh": ".73 and .72",
    "sWritten": "15", "sRange": ".09--.39", "sMedPrimary": ".23", "sRefsKUPA": ".29--.39", "sIKUPA": ".10--.16",
    "sILWPshuf": ".09 and .14",
    "kAucS0": ".90", "kAucR": ".81", "kAucU": ".68", "kAucB": ".60", "kAucBsd": ".59", "kAucI": ".57",
    "kNkupa": "200", "kNessay": "178", "kNlwp": "200", "kNlwpcount": "197", "mNessay": "175",
    "mNlwp": "197", "kAucB500": ".73", "kAucS0ci": ".88, .92", "kAucRci": ".77, .84", "kAucUci": ".64, .72",
    "kAucBci": ".56, .64", "kAucBsdci": ".54, .63", "kAucIci": ".53, .61",
    "kAucS0cls": "above", "kAucRcls": "above", "kAucUcls": "above", "kAucBcls": "undecided",
    "kAucBsdcls": "undecided", "kAucIcls": "undecided",  "kTop1": ".005--.015", "kTop1all": ".05", "kThragree": "11--12",
    # added for v0.2
    "kThrmovU": "8 of 8", "kThrmovB": "6 of 6", "kThrmovBsd": "11 of 12", "kThrmov": "25 of 26",
    "kTop1S0": ".165", "kTop1R": ".097",
    "kUshuf": ".32--.48", "mUshuf": ".31--.49", "kUrandhalf": ".74 and .78", "kRrandhalf": ".25 and .40", "sCounted": "60",
    "kRinsK": ".52--.53", "kRinsL": ".70--.73", "kIinsL": ".33--.37", "kBsdinsL": ".16--.25",
    "nIgateci": ".38, .52; .21, .36; .30, .44",
    # added for v0.3
    "kBsdrandhalf": ".71 and .72", "pIhalvesgate": ".26--.46", "kNrule": "161--177", "mNrule": "157--181",
    "nCIwidth": ".14",
    # added for v0.4
    "kEdgeOE": ".44--.50", "kEdgeHalves": ".05--.11", "mIinsL": ".37--.45", "mBsdinsL": ".15--.24",
    "lIrelL": ".24, .02, .35", "mEssaySum": ".68--.92", "mEssayPos": "7 of 8",
    "nIgatecibr": "[.38, .52], [.21, .36], [.30, .44]", "nRlocsentL": ".75", "lIrelK": ".58, .35, .45", "pIhighrule": ".46, interval [.32, .61]",
    # added for v0.5
    "lBsdrawK": ".56", "lBsdrawL": ".75", "lIrawL": ".14, .01, .21", "nIrawL": ".07, .02, .43",
    "nSwap75L": "18", "nTime75L": "5.2\\%", "oMedL": "8.5, 13.0 and 19.25", "oMedK": "4.5, 6.0 and 9.5",
    "oAboveK": "159 of 181", "nRoseInside": ".00--.07",
    "kAucUreg": ".63 [.59, .67]", "kAucBreg": ".52 [.48, .56]", "kAucBsdreg": ".52 [.48, .56]", "kAucIreg": ".58 [.54, .62]",
    # added for v0.6
    "nSwap100K": "11", "nTime100K": "5.7\\%", "nIgatepersons": "177--181", "sRangePrimary": ".10--.39",
    "kAucSum": ".57--.68", "kThrchanged": "1 of 26",
    # added for v0.7
    "nLWPin75ci": ".67 [.602, .73] and .72 [.6497, .79]", "nLWPcross100ci": ".59 [.52, .66] and .64 [.57, .71]",
    "pIhalvesall": ".25--.46", "qI2halves": ".52--.70", "oMinIn100": "+.15 [.05, .24]",
    "oMinHighIn100": ".00 [$-$.12, .14]", "oMinHighCr100": "+.16 [.01, .32]", "oMinHighMed": "9 and 13",
    "oMinHighN": "50 and 159", "kFitK2": "132 of 178 and 124 of 197", "kFitMaxIter": "77 and 70",
}


def main():
    R = load()
    N = computed(R)
    if set(N) != set(EXPECTED):
        sys.exit(f"keys without an expected value: {sorted(set(N) - set(EXPECTED))}; "
                 f"expected keys not computed: {sorted(set(EXPECTED) - set(N))}")
    bad = [(key, N.get(key), want) for key, want in EXPECTED.items() if N.get(key) != want]
    if bad:
        for key, got, want in bad:
            print(f"{key}: computed {got!r}, expected {want!r}", file=sys.stderr)
        sys.exit(f"{len(bad)} key(s) differ from the expected values")
    out = Path(__file__).resolve().parent / "numbers.json"
    figs = {"fig2.tex": figure2(R), "fig2-halves.tex": figure2(R, "halves")}  # every check runs before any file is written
    out.write_text(json.dumps(N, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    for name, text in figs.items():
        (out.parent / name).write_text(text, encoding="utf-8")
    print(f"{len(N)} keys written to {out.name}; all {len(EXPECTED)} expected values matched")


if __name__ == "__main__":
    main()
