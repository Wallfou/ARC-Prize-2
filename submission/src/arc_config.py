"""Central configuration for the ARC Prize 2026 port of the NVARC solution.

Everything that differs between NVARC's original ARC-AGI-1 notebook and this
2026 submission lives here, so `arc_loader.py` and `arc_decoder.py` stay
byte-identical to the originals and `arc_solver.py` keeps a small diff.
"""

import os
import glob
import time


# --- Which model to run ------------------------------------------------------
# Attach the Kaggle model, then set MODEL_SLUG to its directory name.
#   sorokin/qwen3_2b_grids15_sft141 -> 22.22% public LB in 6h10min (paper Table 2)
#   sorokin/qwen3_4b_grids15_sft139 -> 29.72% public LB in 12h     (paper Table 2)
MODEL_SLUG = os.environ.get("ARC_MODEL_SLUG", "qwen3_2b_grids15_sft141")


# --- Paths -------------------------------------------------------------------
COMPETITION = "arc-prize-2026-arc-agi-2"
INPUT_DIR = os.environ.get("ARC_INPUT_DIR", "/kaggle/input")
WORK_DIR = os.environ.get("ARC_WORK_DIR", "/kaggle/working")

TEST_FILE = "arc-agi_test_challenges.json"


def find_data_dir():
    """Locate the competition data by filename rather than by mount path.

    Kaggle's mount layout is not one thing: datasets land at
    /kaggle/input/<slug>/, notebook outputs at
    /kaggle/input/notebooks/<owner>/<slug>/, and the competition directory name
    does not always equal the competition slug. Search for the file instead.
    """
    if os.environ.get("ARC_DATA_DIR"):
        return os.environ["ARC_DATA_DIR"]

    direct = os.path.join(INPUT_DIR, COMPETITION)
    if os.path.exists(os.path.join(direct, TEST_FILE)):
        return direct

    hits = glob.glob(os.path.join(INPUT_DIR, "**", TEST_FILE), recursive=True)
    if hits:
        return os.path.dirname(sorted(hits, key=len)[0])

    raise FileNotFoundError(
        f"Could not find {TEST_FILE} anywhere under {INPUT_DIR}. Attach the "
        f"'{COMPETITION}' competition to this notebook.\n"
        f"Present: {sorted(glob.glob(os.path.join(INPUT_DIR, '*')))}"
    )


DATA_DIR = find_data_dir() if os.path.isdir(INPUT_DIR) else os.path.join(
    INPUT_DIR, COMPETITION)

TEST_CHALLENGES = os.path.join(DATA_DIR, TEST_FILE)
EVAL_CHALLENGES = os.path.join(DATA_DIR, "arc-agi_evaluation_challenges.json")
EVAL_SOLUTIONS = os.path.join(DATA_DIR, "arc-agi_evaluation_solutions.json")


# --- Which task set to solve -------------------------------------------------
# "test" is the scored path. "eval" solves the 120 public evaluation tasks,
# whose answers ship with the competition, so a commit run yields a real
# accuracy number without spending one of the 1-per-day submissions.
#
# Comparable published baseline: NVARC's paper (Table 2) reports 25/120 on this
# same split for the 2B checkpoint, and 30/120 for the 4B. Both were measured
# with the same contamination we have -- 55% of their training data was seeded
# from these very puzzles' descriptions -- so the comparison is apples to
# apples even though the absolute number reads high.
TASK_SET = os.environ.get("ARC_TASK_SET", "test")
assert TASK_SET in ("test", "eval"), TASK_SET


def is_rerun():
    """True during the scored rerun against the 240 hidden tasks."""
    return bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN"))


# A notebook left in eval mode and then submitted would write a 120-key
# submission, miss every hidden task id, and score 0.00 -- while
# validate_format printed "Format OK" (it validates against the eval file) and
# the solved-count guard passed. Silent zero. Force the scored path instead of
# asserting, because an assert here fires before the fallback submission is
# written, which would leave no submission at all.
if TASK_SET != "test" and is_rerun():
    print(f"!!! ARC_TASK_SET was '{TASK_SET}' on a scored rerun -- forcing "
          f"'test'. An eval-mode submission scores 0.00 silently.")
    TASK_SET = "test"

CHALLENGES = TEST_CHALLENGES if TASK_SET == "test" else EVAL_CHALLENGES
SOLUTIONS = None if TASK_SET == "test" else EVAL_SOLUTIONS

# 0 = every task. Set lower to sample the split for a faster signal.
TASK_LIMIT = int(os.environ.get("ARC_TASK_LIMIT", 0))

# NVARC wrote these to '../inference_outputs' and '../worker{rank}', which on
# Kaggle resolve to /kaggle/ -- not writable. Keep them under /kaggle/working.
OUTPUT_DIR = os.path.join(WORK_DIR, "inference_outputs")
WORKER_FLAG_DIR = os.path.join(WORK_DIR, "worker_flags")
SUBMISSION_PATH = os.path.join(WORK_DIR, "submission.json")
# One JSONL per worker: structured per-puzzle timings, memory and
# failures. Retrievable for commit and eval runs (not for scored
# reruns, whose only output is submission.json), and far easier to
# analyse than regexing the log.
RUN_LOG_DIR = os.path.join(WORK_DIR, "run_logs")
# One subdirectory per swept value. Deliberately NOT under OUTPUT_DIR: the
# decoder lists that directory and bz2-opens every entry, and it is kept
# byte-identical to NVARC's, so a nested directory there would crash it.
SWEEP_DIR = os.path.join(WORK_DIR, "sweep_outputs")


def model_path():
    """Resolve an attached Kaggle model to its weights directory.

    Verified layout (2026-08):
        /kaggle/input/models/<owner>/<slug>/<framework>/<variation>/<version>/
    The owner prefix and version both vary, so search for config.json under any
    directory whose path contains the slug rather than assuming a depth.
    """
    if os.environ.get("ARC_MODEL_DIR"):
        return os.environ["ARC_MODEL_DIR"]

    hits = [p for p in glob.glob(os.path.join(INPUT_DIR, "**", "config.json"),
                                 recursive=True)
            if MODEL_SLUG in p]
    if hits:
        # Shortest path wins: avoids nested subdirs like ./checkpoint/config.json
        return os.path.dirname(sorted(hits, key=len)[0])

    raise FileNotFoundError(
        f"No config.json for '{MODEL_SLUG}' under {INPUT_DIR}. Attach the Kaggle "
        f"model 'sorokin/{MODEL_SLUG}' to this notebook, or set ARC_MODEL_DIR.\n"
        f"Present: {sorted(glob.glob(os.path.join(INPUT_DIR, '*')))}"
    )


# --- Run mode ----------------------------------------------------------------
# is_rerun() is defined above, before the TASK_SET guard that depends on it.
# On a commit run Kaggle ships a placeholder arc-agi_test_challenges.json that
# is 240 *training* tasks (verified: 240/240 overlap with the training file,
# 0/240 with the eval set), so any score from it is meaningless.


# Tasks solved during a non-rerun commit, purely to exercise the pipeline.
SMOKE_TEST_KEYS = ["00576224", "007bbfb7", "009d5c81", "00d62c1b"]


# --- Time budget -------------------------------------------------------------
# Kaggle hard-kills at 12h and obfuscates reported runtime by up to ~10 min, so
# leave more headroom than NVARC's 600s. Anchored at import time.
TOTAL_BUDGET_S = float(os.environ.get("ARC_TOTAL_BUDGET_S", 12 * 3600))
SAFETY_MARGIN_S = float(os.environ.get("ARC_SAFETY_MARGIN_S", 1200))

# A shortened budget is an experiment-only setting. Left on by accident during a
# scored rerun it would stop the solver hours early and quietly submit mostly
# fallbacks -- the same silent-zero shape as an eval-mode submission, which is
# why TASK_SET is force-corrected above. Force this one too, for the same reason:
# the guard must cover every knob that can truncate a scored run, not just the
# one that has burned us before.
if is_rerun():
    if TOTAL_BUDGET_S < 12 * 3600:
        print(f"!!! ARC_TOTAL_BUDGET_S was {TOTAL_BUDGET_S:.0f}s on a scored "
              f"rerun -- forcing {12 * 3600}s. A short budget silently submits "
              f"fallbacks for every task the solver never reaches.")
        TOTAL_BUDGET_S = 12 * 3600
    if SAFETY_MARGIN_S < 1200:
        print(f"!!! ARC_SAFETY_MARGIN_S was {SAFETY_MARGIN_S:.0f}s on a scored "
              f"rerun -- forcing 1200s. Kaggle hard-kills at 12h and misreports "
              f"runtime by up to ~10min; too small a margin loses the whole run.")
        SAFETY_MARGIN_S = 1200.0

# Per-puzzle wall clock before we stop decoding and move on (NVARC used 1200).
PUZZLE_BUDGET_S = float(os.environ.get("ARC_PUZZLE_BUDGET_S", 1200))
# Per-DFS-call cap (NVARC used 540).
DFS_BUDGET_S = float(os.environ.get("ARC_DFS_BUDGET_S", 540))

NUM_WORKERS = int(os.environ.get("ARC_NUM_WORKERS", 4))  # one per L4

# NVARC ran with gradient checkpointing OFF -- they had memory to spare on an L4.
# The 4B is ~2.6x the 2B's parameters and we lack flash-attn, so headroom is
# thinner. Checkpointing is numerically identical, just recomputes activations
# instead of storing them: roughly 30% slower for a large memory saving. Flip to
# "1" if a run OOMs; we have hours of unused budget to pay for it.
GRAD_CKPT = os.environ.get("ARC_GRAD_CKPT", "0") == "1"

# One-off decoder correctness probe. Costs one puzzle's worth of extra
# decoding on rank 0, so keep it off for scoring runs.
PROBE_DFS = os.environ.get("ARC_PROBE_DFS", "0") == "1"

# Depth-first search keeps any completed grid whose TOTAL probability exceeds
# this. NVARC used 0.2. Our failure analysis says this floor is what prunes the
# search on the puzzles we get wrong: on failures a median of 9 of 16 augmented
# views return nothing at all, and the shortfall is not divisible by the batch
# size, so it is the floor rather than a timeout.
#
# Lowering it is strictly more search, so a low-floor run is a SUPERSET of a
# high-floor one -- filter the saved candidates by beam_score afterwards to
# reconstruct any higher floor offline. One run, whole sweep.
MIN_PROB = float(os.environ.get("ARC_MIN_PROB", 0.2))

# --- Conditional deep retry on starved outputs -------------------------------
# An output is STARVED when the search produced fewer than two *unique* candidate
# grids for it. Below two you cannot even fill both pass@2 attempts, so the
# second attempt is structurally wasted.
#
# Lowering MIN_PROB for every puzzle was measured harmful -- 19.9h against an
# 11.6h budget, pass@2 flat -- because it pays the cost on the puzzles that
# already work. Retrying only the starved outputs pays it exactly where the
# score is currently zero, which is what that global sweep could not see.
#
# Measured on our 120-task 4B eval run (submission/analyze_starvation.py):
#   24 of 172 outputs starved (14.0%), 9 of them with no candidate at all;
#   20 of the 24 are currently wrong -- the reachable ceiling, 11.6%;
#   109 outputs (63.4%) are wrong while NOT starved, and a retry never sees them.
# Starved puzzles are *cheaper* than the rest (median 392s against 575s): they
# finish early because nothing clears the floor, so they have headroom.
#
# MEASURED WORTHLESS 2026-08-30, on that notebook's own published run output.
# `yusuketogashi/arc-baseline-rebuild` (31.81% LB, rank 147) publishes both its
# primary-only and its deep-augmented submission for the same 120-task eval run.
# Scoring both against the eval solutions: 43/172 either way. Zero net flips,
# zero gains, zero losses -- from a deep pass STRICTLY MORE aggressive than this
# one (floor 0.08 AND 24 inference origins AND 16 scoring views AND a 2400s cap,
# 1.9h over 18 tasks, deep_status=COMPLETED).
#
# It did work mechanically: outputs with no candidate at all fell 10 -> 4 and 13
# submitted answers changed. None of the recovered grids was correct. Starvation
# is the model not knowing the answer, not the floor pruning a known one -- which
# is exactly the generation bottleneck, and no search setting fixes it.
# DEFAULT OFF: measured worthless 2026-08-30, see below. Kept because the code
# and its analysis tooling are correct and cheap to re-enable if the generator
# ever improves enough that a starved output is a threshold artifact rather
# than the model simply not knowing the answer.
RETRY_STARVED = os.environ.get("ARC_RETRY_STARVED", "0") == "1"
RETRY_MIN_PROB = float(os.environ.get("ARC_RETRY_MIN_PROB", 0.08))
RETRY_MIN_UNIQUE = int(os.environ.get("ARC_RETRY_MIN_UNIQUE", 2))

# Extra wall clock one puzzle may spend on its retry, on top of PUZZLE_BUDGET_S.
RETRY_BUDGET_S = float(os.environ.get("ARC_RETRY_BUDGET_S", 300))

# The retry must not buy candidates for early puzzles by dropping late ones.
# A 240-task scored run has ~20 minutes of slack in 11.6h, so the retry only
# fires when this many puzzles beyond those still queued would fit before the
# deadline at the worker's current pace. On a 120-task eval commit run, which
# uses 5.2h of the same budget, the slack is large and it fires freely.
RETRY_PACE_MARGIN = int(os.environ.get("ARC_RETRY_PACE_MARGIN", 8))


# --- A2: within-run learning-rate sweep ---------------------------------------
# Test-time fine-tuning is the only lever that touches the 67.4% generation gap,
# and lr is the only part of the A2 bundle that is free: it does not change the
# step count, where `num_train_epochs` 1->2 and `augment(n=)` 16->32 each add
# ~3.2h to a 240-task run against ~1.2h of slack.
#
# The problem is measurement, not the change. The eval loop's own noise floor is
# 3.47pp (measured v40 vs baseline on outputs nothing touched), and unlike the
# A3 retry an lr change has no inert subset to partition on -- it touches every
# output. Two separate runs cannot resolve the ~1pp an lr change might be worth.
#
# So sweep WITHIN one run: enqueue each task once per value and write each
# variant to its own output directory. Every pair is then compared on the same
# machine, in the same run, under the same clock -- which removes the run-level
# variance that dominates (§5a: machine speed -> truncation) and leaves only
# per-task batched-decode nondeterminism. Costs one extra fine-tune+decode per
# value, so N values is N times the run.
#
# ARC_SWEEP_PARAM names any key of `train_args`, so the same harness covers
# learning_rate (v47, null) and num_train_epochs (never tested -- it has been
# costed twice and discussed as part of the A2 bundle, but `num_train_epochs=1`
# has not changed since the original port).
#
# Empty = off, and the sweep must never reach a scored run: it would multiply
# the work and cover only 240/N tasks. Guarded below.
SWEEP_PARAM = os.environ.get("ARC_SWEEP_PARAM", "learning_rate")
SWEEP = [float(x) for x in
         os.environ.get("ARC_SWEEP", "").replace(" ", "").split(",") if x]

# Short directory prefix per knob, so arms read `epochs_2` rather than
# `num_train_epochs_2`. Unknown knobs fall back to their own name.
SWEEP_TAG = {"learning_rate": "lr",
             "num_train_epochs": "epochs"}.get(SWEEP_PARAM, SWEEP_PARAM)


def sweep_dir(value):
    """Output directory for one arm. `:g` keeps 2.0 as `epochs_2`."""
    return os.path.join(SWEEP_DIR, f"{SWEEP_TAG}_{value:g}")


if SWEEP and is_rerun():
    print(f"!!! ARC_SWEEP was {SWEEP} on {SWEEP_PARAM} during a scored rerun -- "
          f"forcing off. A sweep enqueues every task once per value, so it "
          f"would cover only 240/{len(SWEEP)} tasks and submit fallbacks for "
          f"the rest.")
    SWEEP = []


def retry_slack(queued, elapsed_s, num_done, remaining_s, num_workers=None):
    """Puzzles of headroom beyond those still queued, at the current pace.

    Positive means this worker could finish its remaining share and still have
    room to spare; the retry is only allowed to spend that surplus. Kept here,
    pure and free of torch, so it can be tested without a GPU.
    """
    workers = num_workers or NUM_WORKERS
    avg = elapsed_s / max(1, num_done)
    if avg <= 0:
        return 0.0
    fit = remaining_s / avg
    mine = max(0, queued) / max(1, workers)
    return fit - mine

_START = time.time()


def deadline():
    return _START + TOTAL_BUDGET_S - SAFETY_MARGIN_S
