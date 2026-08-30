"""How many outputs would a conditional deep-retry pass fire on?

Mirrors the selection policy used by the public notebook
`yusuketogashi/arc-baseline-rebuild` (31.81% public LB, rank 147 on
2026-08-30), which runs a cheap primary pass over every task and then re-runs
only a small set of "starved" tasks with an expensive configuration.

Its trigger is sharper than "empty candidate set": an output is STARVED when it
has fewer than 2 *unique* candidate grids, because with fewer than 2 you cannot
fill both pass@2 attempts -- the second attempt is structurally wasted. MISSING
is the subset with zero candidates.

Tasks are ranked by (most missing outputs, most starved outputs, fewest
candidate files) and the top N are retried.

This script sizes that lever against a run we already paid for: how many of our
outputs are starved, how many of those are currently wrong, and therefore what
the ceiling on a deep pass actually is. No GPU needed.

    python3 submission/analyze_starvation.py --dir /tmp/run/inference_outputs
"""

import argparse
import bz2
import json
import os
import pickle
from collections import defaultdict

import numpy as np


def fingerprint(grid):
    a = np.asarray(grid, dtype=int)
    return (a.shape, a.tobytes())


def load(cand_dir):
    """Group candidates by base key (task_testindex), mirroring ArcDecoder."""
    cands, files = defaultdict(list), defaultdict(list)
    for name in sorted(os.listdir(cand_dir)):
        base = name.split(".")[0]
        files[base].append(name)
        try:
            with bz2.BZ2File(os.path.join(cand_dir, name)) as f:
                cands[base].extend(pickle.load(f))
        except Exception as exc:
            print(f"  !! corrupt {name}: {type(exc).__name__}: {exc}")
    return cands, files


def kgmon(entries):
    """NVARC's ranker, as in analyze_candidates.py. Higher is better."""
    return len(entries) - float(np.mean([np.mean(e["score_aug"]) for e in entries]))


def main():
    p = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--dir", required=True)
    p.add_argument("--solutions", default=os.path.join(
        here, os.pardir, "example", "arc-agi_evaluation_solutions.json"))
    p.add_argument("--challenges", default=os.path.join(
        here, os.pardir, "example", "arc-agi_evaluation_challenges.json"))
    p.add_argument("--max-deep-tasks", type=int, default=24)
    a = p.parse_args()

    sol = json.load(open(a.solutions))
    chal = json.load(open(a.challenges))
    cands, files = load(a.dir)

    # Seed from the challenge file so outputs that produced no file at all are
    # still counted -- those are the worst case and would otherwise vanish.
    expected_by_task = {
        tid: [f"{tid}_{i}" for i in range(len(t["test"]))]
        for tid, t in sorted(chal.items())
    }
    expected = [bk for bks in expected_by_task.values() for bk in bks]

    uniq, correct, solved = {}, {}, {}
    for bk in expected:
        entries = cands.get(bk, [])
        groups = defaultdict(list)
        for e in entries:
            groups[fingerprint(e["solution"])].append(e)
        uniq[bk] = len(groups)

        tid, idx = bk.rsplit("_", 1)
        truth = np.asarray(sol[tid][int(idx)], dtype=int)
        tfp = fingerprint(truth)
        correct[bk] = tfp in groups
        top2 = sorted(groups.items(), key=lambda kv: -kgmon(kv[1]))[:2]
        solved[bk] = any(k == tfp for k, _ in top2)

    n = len(expected)
    missing = [bk for bk in expected if uniq[bk] == 0]
    starved = [bk for bk in expected if uniq[bk] < 2]

    print(f"\n=== outputs: {n} over {len(expected_by_task)} tasks "
          f"({len(cands)} with at least one file) ===")
    print(f"solved (pass@2)            : {sum(solved.values()):>4} "
          f"({sum(solved.values())/n:.1%})")
    print(f"correct grid generated     : {sum(correct.values()):>4} "
          f"({sum(correct.values())/n:.1%})")
    print(f"\nMISSING  (0 unique cands)  : {len(missing):>4} ({len(missing)/n:.1%})")
    print(f"STARVED  (<2 unique cands) : {len(starved):>4} ({len(starved)/n:.1%})")

    print("\n--- unique-candidate distribution ---")
    hist = defaultdict(int)
    for bk in expected:
        hist[min(uniq[bk], 10)] += 1
    for k in sorted(hist):
        label = f"{k}" if k < 10 else "10+"
        bar = "#" * int(50 * hist[k] / n)
        print(f"  {label:>3} unique: {hist[k]:>4}  {bar}")

    # A retry can only help where we are currently wrong. Split the starved set.
    starved_wrong = [bk for bk in starved if not solved[bk]]
    print(f"\n--- upside available to a deep retry ---")
    print(f"starved AND currently wrong: {len(starved_wrong):>4} "
          f"({len(starved_wrong)/n:.1%} of outputs)  <- the reachable ceiling")
    print(f"starved but already solved : {len(starved) - len(starved_wrong):>4}"
          f"  (retry here cannot help, and must not regress)")

    # Where is the mass? Starved vs non-starved failure.
    nonstarved_wrong = [bk for bk in expected
                        if uniq[bk] >= 2 and not solved[bk]]
    print(f"NOT starved but wrong      : {len(nonstarved_wrong):>4} "
          f"({len(nonstarved_wrong)/n:.1%})  <- a deep retry never sees these")

    # Reproduce the public notebook's task ranking.
    missing_by_task, starved_by_task = defaultdict(list), defaultdict(list)
    for tid, bks in expected_by_task.items():
        for bk in bks:
            if uniq[bk] == 0:
                missing_by_task[tid].append(bk)
            if uniq[bk] < 2:
                starved_by_task[tid].append(bk)
    ranked = sorted(starved_by_task, key=lambda t: (
        -len(missing_by_task.get(t, [])),
        -len(starved_by_task.get(t, [])),
        len(files.get(t, [])),
        t,
    ))
    picked = ranked[:a.max_deep_tasks]
    reachable = sum(1 for t in picked for bk in expected_by_task[t]
                    if not solved[bk])
    print(f"\n--- selector at max_deep_tasks={a.max_deep_tasks} ---")
    print(f"tasks starved (eligible)   : {len(ranked):>4}")
    print(f"tasks selected             : {len(picked):>4}")
    print(f"their outputs still wrong  : {reachable:>4} "
          f"-> +{reachable/n:.1%} if every one were fixed (it will not be)")
    print(f"selected: {', '.join(picked[:12])}"
          f"{' ...' if len(picked) > 12 else ''}")


if __name__ == "__main__":
    main()
