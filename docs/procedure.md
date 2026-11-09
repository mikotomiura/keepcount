# Using the procedure: running the code, the inputs, and random splits

This note accompanies Appendix H of the article *Can a pause summary show where one writer paused? An audit procedure
for keystroke logs*. It gives three things that the appendix points to: how to run the code of this archive, how the
fields of the KLiCKe corpus stand against the inputs the procedure needs, and how the variation of the per-writer
verdicts over random two-part splits could be counted. The last was not counted for the article.

## 1. Running the code

Environment: [uv](https://docs.astral.sh/uv/) and git, as in the README (`uv sync --frozen`). Every run was made on
macOS, Apple silicon.

Each run `vX` has its conditions in `works/conditions/prototype_vX.json` (committed before the run's code was first
run), its code in `works/prototype_vX.py` and its tests in `tests/test_prototype_vX.py`. A run reads the keystroke
data at the paths its conditions name (README, Reproduce), stops if a file's SHA-256 differs, refuses to overwrite its
results, and checks that every file it reads equals the last commit.

The runs of the article are v6k to v6s, in this order; each reads the records of the runs before it:

| Run | What it counts | Outputs of the article that read its record (`sources.json`) |
|---|---|---|
| v6k | the summaries U, B, B_sd and I at fixed thresholds and at a writer's threshold; the rule count and identification | 52 body values; app-cells, app-complete, app-flow, app-id, app-margin, app-pred, app-rlag, app-runs, app-thr, app-wthr |
| v6l | split-half reliability; R split into R_loc and R_lag; B_complete; the trimmed writer's threshold | 10 body values; app-pred, app-rel, app-rlag, app-runs |
| v6m | the rule recounted on new writers; copy against essay; doses of inside and cross | 14 body values; app-cells, app-complete, app-contrast, app-flow, app-margin, app-pred, app-runs, app-wthr |
| v6n | the reference built from means (R_mean); the 75% doses; direction over writers; reliability on v6m's writers | 28 body values; Figure 2; app-border, app-complete, app-dose, app-margin, app-pred, app-runs, app-split |
| v6o | the reach of R_loc and R_mean by the number of sentence starts per part | 8 body values; app-groups, app-pred, app-runs |
| v6p | the rule stages by the number of sentence starts per part | 7 body values; app-groups, app-pred, app-runs |
| v6q | I's between-word mean I[2] alone | 5 body values; app-pred, app-runs |
| v6r | the agreement of the two fixed splits as a full table (every cell withheld by the disclosure rule) | app-pred, app-runs |
| v6s | the agreement share p_o of the two fixed splits | 6 body values; app-groups, app-pred, app-runs, app-split |

The records of v6k to v6s are not in this archive (README, "What is not in this archive"). To rebuild them:

```bash
ln -s /path/to/your/external external      # the keystroke data at the paths of the README
git switch -c rebuild
uv run --frozen python tools/rebuild_records.py
```

`tools/rebuild_records.py` runs v6k to v6s in order and stops unless each rebuilt record has the content of the
original (the SHA-256 of the record without its provenance, `results/record-digests.json`; `tools/record_digest.py`).
A rebuilt record differs from the original file in its provenance only (time, commit, and hashes of files), so the
SHA-256 that later conditions name for it no longer matches; on the local branch, the tool replaces that SHA-256 (and
the SHA-256 of each conditions file so changed) in the conditions of the later runs, checks that nothing else in them
changed, and commits the record with those conditions so that the next run reads committed bytes. Use `--from vX` to
continue after a stop. With the records of v6k and v6l present, `uv run --frozen pytest` also runs the tests that read
them (otherwise they are skipped, with the reason).

The values written in the article are made from the records by the scripts in `article/jowr/values/`:

```bash
uv run --frozen python article/jowr/values/paper_numbers.py     # numbers.json, fig2.tex, fig2-halves.tex
uv run --frozen python article/jowr/values/appendix_tables.py   # app-*.tex
uv run --frozen python article/jowr/values/ledger_inventory.py  # the disclosure ledger, step 1
uv run --frozen python article/jowr/values/ledger_classify.py    # the disclosure ledger, step 2
```

They check each record by the SHA-256 of the original file and, failing that, by its content digest
(`results/record-digests.json`), so they run on rebuilt records too; only the table of runs (`app-runs.tex`), which
prints each record's time, commit and SHA-256, then differs from the article's. The ledger scripts also write `ledger_inventory.json`, which holds the record values of the cells written, and
`ledger_classes.json`; neither is in this archive (`.gitignore`). The outputs as the article prints them are in
the same folder, and `sources.json` lists, per output, the record cells read to make it (key paths, no values).

## 2. KLiCKe's fields and the inputs of the procedure

The procedure needs, per press: the writer, its place in the order of typing, its time, and a key category (letter or
digit, space, subsentence mark, sentence end, other) from which sentences, subsentences and words are cut (Section 3
and Appendix H of the article). The table gives the fields our code reads from the two datasets it was run on, and the
fields of the KLiCKe corpus as its files name them (Tian, Crossley and Van Waes, 2025). KLiCKe was not run.

| Input | KUPA-KEYS (`works/prototype_v6b.py`, `read_kupa_b`) | LWP (`works/prototype_v6e.py`, `read_lwp`) | KLiCKe (not run) |
|---|---|---|---|
| Writer | `id` | `id` | not checked |
| Order of typing | the file's order | `event_id` (checked to increase within a writer) | Event ID |
| Time of the press | `time`, rows with `type` = down and `is_repeat` false | `down_time` | Down Time |
| Key | `key` | `down_event` (`Space` read as a space) | no key field; the category could be read from Text Change and Activity (not checked) |
| Events not counted | rows other than presses; keys of no category | `down_event` of no category | not checked |

Fields that our code does not read: LWP's `up_time`, `action_time`, `activity` (read only by a sensitivity count of
v6e), `up_event`, `text_change`, `cursor_position` and `word_count`; KLiCKe's Up Time, Pause Time, Action Time,
Position and Word Count. After reading, the code keeps the counted presses in the file's order, sorts them stably by
time, drops a press at the same millisecond as the previous counted press, stops if a time is not finite, and then cuts
the gaps and the sentences (Appendix H).

## 3. Counting the verdicts over random two-part splits (not counted)

The article counts each writer's verdict under two fixed splits of the sentences into two parts, odd/even sentences and
first/second half, and reports how often the two verdicts agree (Section 5.3; run v6s). It does not count how the
verdict varies over random splits. One way to count it, which we have not run:

1. Fix before counting: the number of random splits S (for example as many as v6l's permutation splits), the random
   key per writer and split, and whether a split must keep the two parts' sizes equal (as the halves split does).
2. For each writer, cell and split s = 1, ..., S: split the writer's sentences at random into two parts with the key;
   compute the summary's own distance between the two parts and the change of the operation's row; record the verdict
   (1 if the change exceeds the own distance, 1/2 for a tie, 0 otherwise; the count 1 of the article).
3. Per writer, the share of splits with each verdict; per cell, the distribution of those shares over the writers, and
   the share of writers whose verdict is the same in every split.
4. Before writing any of these values, apply the disclosure rule of Appendix A: list every quantity with the set of
   writers it is computed on and withhold the cells from which a sum over fewer than five writers could be derived.

The functions that compute a summary's distance and the count 1 for a given split are those of the runs above
(v6k to v6s use the fixed splits; v6l's reliability uses random splits of the sentences into two halves).
