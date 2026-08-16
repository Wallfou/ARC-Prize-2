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

# NVARC wrote these to '../inference_outputs' and '../worker{rank}', which on
# Kaggle resolve to /kaggle/ -- not writable. Keep them under /kaggle/working.
OUTPUT_DIR = os.path.join(WORK_DIR, "inference_outputs")
WORKER_FLAG_DIR = os.path.join(WORK_DIR, "worker_flags")
SUBMISSION_PATH = os.path.join(WORK_DIR, "submission.json")


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
