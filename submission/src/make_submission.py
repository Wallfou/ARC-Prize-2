"""Aggregate per-puzzle DFS candidates into submission.json.

This is NVARC's notebook cell 9, split out so it can run standalone and so a
valid submission always exists on disk even if the solver never finishes.
"""

import os
import json
import argparse

import numpy as np

import arc_config
from arc_loader import ArcDataset
from arc_decoder import ArcDecoder, score_kgmon, score_full_probmul_3


SELECTORS = {
    # NVARC's post-deadline formula (paper 3.4): DFS hit count + mean augmented
    # log-prob. This is what the vendored notebook defaults to, and what the
    # 29.72% run used.
    "kgmon": score_kgmon,
    # The original ARChitects product-of-experts score.
    "probmul": score_full_probmul_3,
}


def write_fallback(challenges_path=None, out_path=None):
    """Emit a format-valid submission covering every task, before any GPU work.

    Kaggle scores a missing or malformed submission.json as a hard failure, so
    write the all-[[0]] skeleton first and overwrite it later with real answers.
    """
    challenges_path = challenges_path or arc_config.TEST_CHALLENGES
    out_path = out_path or arc_config.SUBMISSION_PATH

    data = ArcDataset.from_file(challenges_path)
    submission = data.get_submission()
    with open(out_path, "w") as f:
        json.dump(submission, f)
    print(f"*** Wrote fallback submission for {len(submission)} tasks -> {out_path}")
    return submission


def validate_format(submission, challenges):
    """Fail loudly on the ways a submission silently scores zero."""
    problems = []

    missing = set(challenges) - set(submission)
    extra = set(submission) - set(challenges)
    if missing:
        problems.append(f"{len(missing)} task ids missing, e.g. {sorted(missing)[:3]}")
    if extra:
        problems.append(f"{len(extra)} unexpected task ids, e.g. {sorted(extra)[:3]}")

    for key, task in challenges.items():
        entry = submission.get(key)
        if entry is None:
            continue
        if not isinstance(entry, list) or len(entry) != len(task["test"]):
            problems.append(
                f"{key}: expected {len(task['test'])} outputs, got "
                f"{len(entry) if isinstance(entry, list) else type(entry).__name__}"
            )
            continue
        for i, attempts in enumerate(entry):
            for name in ("attempt_1", "attempt_2"):
                grid = attempts.get(name)
                if grid is None:
                    problems.append(f"{key}[{i}]: {name} absent")
                    continue
                if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
                    problems.append(f"{key}[{i}].{name}: not a 2D list")
                    continue
                widths = {len(r) for r in grid}
                if len(widths) != 1 or 0 in widths:
                    problems.append(f"{key}[{i}].{name}: ragged or empty rows")
                elif len(grid) > 30 or max(widths) > 30:
                    problems.append(f"{key}[{i}].{name}: exceeds 30x30")
                elif not all(isinstance(c, int) and 0 <= c <= 9 for r in grid for c in r):
                    problems.append(f"{key}[{i}].{name}: cells outside 0-9 int")

    return problems


def build(selector="kgmon", challenges_path=None, solutions_path=None, out_path=None):
    challenges_path = challenges_path or arc_config.TEST_CHALLENGES
    out_path = out_path or arc_config.SUBMISSION_PATH

    data = ArcDataset.from_file(challenges_path)
    if solutions_path:
        data = data.load_replies(solutions_path)

    decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)
    decoder.load_decoded_results(arc_config.OUTPUT_DIR)
    print(f"*** Loaded candidates for {len(decoder.decoded_results)} test outputs")

    submission = data.get_submission(decoder.run_selection_algo(SELECTORS[selector]))

    with open(challenges_path) as f:
        challenges = json.load(f)
    problems = validate_format(submission, challenges)
    if problems:
        print(f"!!! {len(problems)} FORMAT PROBLEMS")
        for p in problems[:20]:
            print("   ", p)
        raise SystemExit(1)
    print(f"*** Format OK: {len(submission)} tasks, "
          f"{sum(len(v) for v in submission.values())} outputs, 2 attempts each")

    with open(out_path, "w") as f:
        json.dump(submission, f)
    print(f"*** Wrote {out_path}")

    if solutions_path:
        score = data.validate_submission(submission)
        n = len(data.keys)
        print(f"*** Local score: {score:.2f}/{n} = {100 * score / n:.2f}%")

    return submission


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selector", default="kgmon", choices=sorted(SELECTORS))
    p.add_argument("--challenges", default=None)
    p.add_argument("--solutions", default=None, help="score locally if given")
    p.add_argument("--out", default=None)
    p.add_argument("--fallback-only", action="store_true")
    a = p.parse_args()

    if a.fallback_only:
        write_fallback(a.challenges, a.out)
    else:
        build(a.selector, a.challenges, a.solutions, a.out)
