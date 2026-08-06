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

DATA_DIR = os.path.join(INPUT_DIR, COMPETITION)

TEST_CHALLENGES = os.path.join(DATA_DIR, "arc-agi_test_challenges.json")
EVAL_CHALLENGES = os.path.join(DATA_DIR, "arc-agi_evaluation_challenges.json")
EVAL_SOLUTIONS = os.path.join(DATA_DIR, "arc-agi_evaluation_solutions.json")

# NVARC wrote these to '../inference_outputs' and '../worker{rank}', which on
# Kaggle resolve to /kaggle/ -- not writable. Keep them under /kaggle/working.
OUTPUT_DIR = os.path.join(WORK_DIR, "inference_outputs")
WORKER_FLAG_DIR = os.path.join(WORK_DIR, "worker_flags")
SUBMISSION_PATH = os.path.join(WORK_DIR, "submission.json")


def model_path():
    """Resolve an attached Kaggle model to its weights directory.

    Kaggle mounts models at /kaggle/input/<slug>/<framework>/<variation>/<ver>/,
    and the version number changes between attachments, so glob for it rather
    than hardcoding. Falls back to a plain directory for local testing.
    """
    direct = os.path.join(INPUT_DIR, MODEL_SLUG)
    hits = sorted(glob.glob(os.path.join(direct, "*", "*", "*", "config.json")))
    if hits:
        return os.path.dirname(hits[-1])
    if os.path.exists(os.path.join(direct, "config.json")):
        return direct
    raise FileNotFoundError(
        f"No config.json under {direct}. Attach the Kaggle model "
        f"'sorokin/{MODEL_SLUG}' to this notebook, or set ARC_MODEL_SLUG."
    )


# --- Run mode ----------------------------------------------------------------
def is_rerun():
    """True during the scored rerun against the 240 hidden tasks.

    On the interactive/commit run Kaggle supplies a placeholder
    arc-agi_test_challenges.json. In 2026 that placeholder is 240 *training*
    tasks (verified: 240/240 overlap with arc-agi_training_challenges.json,
    0/240 with the evaluation set), so any score computed from it is
    meaningless. Solve only a handful there to keep commits fast.
    """
    return bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN"))


# Tasks solved during a non-rerun commit, purely to exercise the pipeline.
SMOKE_TEST_KEYS = ["00576224", "007bbfb7", "009d5c81", "00d62c1b"]


# --- Time budget -------------------------------------------------------------
# Kaggle hard-kills at 12h and obfuscates reported runtime by up to ~10 min, so
# leave more headroom than NVARC's 600s. Anchored at import time.
TOTAL_BUDGET_S = float(os.environ.get("ARC_TOTAL_BUDGET_S", 12 * 3600))
SAFETY_MARGIN_S = float(os.environ.get("ARC_SAFETY_MARGIN_S", 1200))

# Per-puzzle wall clock before we stop decoding and move on (NVARC used 1200).
PUZZLE_BUDGET_S = float(os.environ.get("ARC_PUZZLE_BUDGET_S", 1200))
# Per-DFS-call cap (NVARC used 540).
DFS_BUDGET_S = float(os.environ.get("ARC_DFS_BUDGET_S", 540))

NUM_WORKERS = int(os.environ.get("ARC_NUM_WORKERS", 4))  # one per L4

_START = time.time()


def deadline():
    return _START + TOTAL_BUDGET_S - SAFETY_MARGIN_S
