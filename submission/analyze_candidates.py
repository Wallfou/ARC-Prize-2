"""Failure taxonomy over saved DFS candidates. No GPU needed.

Answers the question PLAN.md asks and nothing else could: when we get a puzzle
wrong, did the search never produce the right grid (a GENERATION problem), or
did it produce it and rank it below the two attempts we submit (a SELECTION
problem)? Those have completely different fixes.

Download a run's output first, then point this at it:

    kaggle kernels output wallfou/arc2-nvarc-2026 -p /tmp/run
    python3 submission/analyze_candidates.py --dir /tmp/run/inference_outputs

Only meaningful for an eval-mode run, since it needs ground-truth answers.
Remember the public eval split is contaminated, so absolute rates read high;
the *shape* of the failure is what transfers.
"""

import argparse
import bz2
import json
import os
import pickle
from collections import Counter, defaultdict

import numpy as np


def load(cand_dir):
    """Group candidate files by test output, mirroring ArcDecoder."""
    cands, views = defaultdict(list), defaultdict(list)
    for name in os.listdir(cand_dir):
        base = name.split(".")[0]
        views[base].append(name)
        with bz2.BZ2File(os.path.join(cand_dir, name)) as f:
            cands[base].extend(pickle.load(f))
    return cands, views


def kgmon(entries):
    """NVARC's ranker: DFS hit count minus mean augmented NLL. Higher is better."""
    return len(entries) - float(np.mean([np.mean(e["score_aug"]) for e in entries]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="inference_outputs from a run")
    p.add_argument("--solutions",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        os.pardir, "example",
                                        "arc-agi_evaluation_solutions.json"))
    p.add_argument("--views", type=int, default=16,
                   help="augmented views decoded per output")
    p.add_argument("--batch", type=int, default=4, help="views per DFS batch")
    a = p.parse_args()

    sol = json.load(open(a.solutions))
    cands, views = load(a.dir)

    found, missed, ranks, miss_views = 0, 0, Counter(), []
    for base, entries in cands.items():
        task, idx = base.rsplit("_", 1)
        if task not in sol:
            continue
        truth = np.array(sol[task][int(idx)])

        by_grid = defaultdict(list)
        for e in entries:
            by_grid[tuple(map(tuple, e["solution"]))].append(e)

        same = [g for g in by_grid if np.array_equal(np.array(g), truth)]
        if not same:
            missed += 1
            miss_views.append(len(views[base]))
            continue

        found += 1
        order = sorted(by_grid, key=lambda g: -kgmon(by_grid[g]))
        ranks[min(order.index(same[0]), 5)] += 1

    total = found + missed
    if not total:
        raise SystemExit("no outputs matched the solutions file")

    print(f"outputs analysed          : {total}")
    print(f"correct grid generated    : {found} ({100*found/total:.1f}%)")
    print(f"never generated           : {missed} ({100*missed/total:.1f}%)"
          "   <- GENERATION gap")

    top2 = ranks[0] + ranks[1]
    print(f"\nrank of the correct grid (0 becomes attempt_1):")
    for r in sorted(ranks):
        print(f"   {'rank 5+' if r >= 5 else f'rank {r}  '} : {ranks[r]}")
    print(f"\nsubmitted correctly      : {top2} ({100*top2/total:.1f}%)")
    print(f"generated but ranked out : {found-top2} "
          f"({100*(found-top2)/total:.1f} points)   <- SELECTION gap")

    # A view that produced no file means the threshold pruned everything, or a
    # timeout killed its batch. Timeouts drop whole batches, so a view count
    # that is not a multiple of the batch size implicates the threshold.
    if miss_views:
        whole = sum(1 for v in miss_views if v % a.batch == 0)
        print(f"\non missed outputs, views returning candidates "
              f"(of {a.views}): median {int(np.median(miss_views))}")
        print(f"   view count a multiple of {a.batch} : {whole}"
              "   (consistent with a batch timeout)")
        print(f"   not a multiple            : {len(miss_views)-whole}"
              "   (threshold pruned single views)")


if __name__ == "__main__":
    main()
