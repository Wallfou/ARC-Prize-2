"""Offline checks for the NVARC port. No GPU, no model weights required.

Run this before every Kaggle commit. It exercises the parts of the pipeline
that fail silently on Kaggle: submission format, augmentation invertibility,
the ordering the DFS batching depends on, and sequence-length headroom.

    python3 submission/validate_local.py
"""

import os
import sys
import json
import types
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

# arc_loader imports AutoTokenizer purely for a type annotation. None of the
# checks below tokenize anything, so stub it rather than require a working
# transformers install on the dev machine. On Kaggle the real import wins.
try:
    from transformers import AutoTokenizer  # noqa: F401
except Exception:
    _stub = types.ModuleType("transformers")
    _stub.AutoTokenizer = object
    sys.modules["transformers"] = _stub

from arc_loader import ArcDataset, QwenFormatter  # noqa: E402
from make_submission import validate_format  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def load(data_dir, name):
    with open(os.path.join(data_dir, name)) as f:
        return json.load(f)


def check_data_split_trap(data_dir):
    """The shipped test file is NOT drawn from the evaluation set.

    Kaggle's Data tab claims arc-agi_test_challenges.json is "a placeholder
    using tasks from arc-agi_evaluation_challenges.json". It is not: all 240
    are training tasks. Scoring against it locally is meaningless.
    """
    tr = load(data_dir, "arc-agi_training_challenges.json")
    ev = load(data_dir, "arc-agi_evaluation_challenges.json")
    te = load(data_dir, "arc-agi_test_challenges.json")
    check("placeholder shares 0 tasks with eval set", len(set(te) & set(ev)) == 0,
          f"{len(set(te) & set(ev))} overlap")
    check("placeholder is drawn from the TRAINING set", len(set(te) & set(tr)) == len(te),
          f"{len(set(te) & set(tr))}/{len(te)} -- validate on eval, never on this")
    check("eval and training are disjoint", len(set(ev) & set(tr)) == 0)


def check_submission_format(data_dir):
    ev_path = os.path.join(data_dir, "arc-agi_evaluation_challenges.json")
    ev = load(data_dir, "arc-agi_evaluation_challenges.json")

    data = ArcDataset.from_file(ev_path)
    skeleton = data.get_submission()
    problems = validate_format(skeleton, ev)
    check("fallback submission passes format validation", not problems,
          "; ".join(problems[:3]))

    n_out = sum(len(v["test"]) for v in ev.values())
    check("fallback covers every test output",
          sum(len(v) for v in skeleton.values()) == n_out,
          f"{sum(len(v) for v in skeleton.values())} vs {n_out}")

    # The validator must actually reject the common failure modes.
    broken = {k: [dict(a) for a in v] for k, v in skeleton.items()}
    victim = sorted(broken)[0]
    del broken[victim][0]["attempt_2"]
    check("validator catches a missing attempt_2",
          any("attempt_2 absent" in p for p in validate_format(broken, ev)))

    dropped = {k: v for k, v in skeleton.items() if k != victim}
    check("validator catches a missing task id",
          any("missing" in p for p in validate_format(dropped, ev)))

    ragged = {k: [dict(a) for a in v] for k, v in skeleton.items()}
    ragged[victim][0]["attempt_1"] = [[0, 0], [0]]
    check("validator catches a ragged grid",
          any("ragged" in p for p in validate_format(ragged, ev)))


def check_augmentation_roundtrip(data_dir):
    """invert_mod must undo forward_mod exactly, or every answer is scrambled."""
    ev_path = os.path.join(data_dir, "arc-agi_evaluation_challenges.json")
    ev = load(data_dir, "arc-agi_evaluation_challenges.json")

    keys = sorted(ev)[:12]
    ds = ArcDataset(queries=ev, keys=keys, is_orig=True).split_multi_replies()
    aug = ds.augment(n=2, seed=2)

    bad = 0
    for subkey in aug.keys:
        grid = np.array(ev[subkey.split(".")[0].split("_")[0]]["train"][0]["input"])
        there = ArcDataset.forward_mod(grid, subkey)
        back = ArcDataset.invert_mod(np.asarray(there), subkey)
        if not np.array_equal(np.asarray(back), grid):
            bad += 1
    check("forward_mod/invert_mod round-trip is exact", bad == 0,
          f"{bad}/{len(aug.keys)} mismatched")


def check_augmentation_order(data_dir):
    """arc_solver batches views by hardcoded offsets 0,2,4,...,14.

    It assumes augment(n=2) yields exactly 16 views per test input, ordered
    (identity, rot90, rot90^2, rot90^3) x (plain, transposed), two colour
    permutations each. If that ordering drifts, batching silently pairs the
    wrong perspectives and re-scoring degrades without any error.
    """
    ev = load(data_dir, "arc-agi_evaluation_challenges.json")
    key = sorted(ev)[0]
    ds = ArcDataset(queries=ev, keys=[key], is_orig=True).split_multi_replies()
    aug = ds.augment(n=2, seed=2)

    n_tests = len(ev[key]["test"])
    check("augment(n=2) yields 16 views per test input",
          len(aug.keys) == 16 * n_tests, f"{len(aug.keys)} for {n_tests} test(s)")

    expected = []
    for prefix in ("", "transpose"):
        for rots in range(4):
            ops = [o for o in (prefix,) if o] + ["rot90"] * rots
            expected += [".".join(ops) or "identity"] * 2

    got = []
    for subkey in sorted(aug.keys)[:16]:
        ops = [o for o in subkey.split(".")[1:]
               if not o.startswith("permute") and not o.startswith("ex")]
        got.append(".".join(ops) or "identity")
    check("view ordering matches the hardcoded DFS batching", got == expected,
          f"got {got[:4]}...")

    every = all(any(o.startswith("permute") for o in sk.split(".")) for sk in aug.keys)
    check("every view carries a colour permutation", every)


def check_sequence_lengths(data_dir, max_seq_length=8192):
    """Token budget under the 16-token vocab: 1 token/cell + 1 newline/row."""
    ev = load(data_dir, "arc-agi_evaluation_challenges.json")

    def grid_tokens(g):
        return sum(len(r) for r in g) + len(g)

    lens = []
    for task in ev.values():
        n = sum(grid_tokens(p["input"]) + grid_tokens(p["output"]) + 4 for p in task["train"])
        n += sum(grid_tokens(p["input"]) + 4 for p in task["test"])
        lens.append(n)
    lens = np.array(lens)
    over = int((lens > max_seq_length).sum())
    check(f"most eval tasks fit in {max_seq_length} tokens", over / len(lens) < 0.05,
          f"median={np.median(lens):.0f} p90={np.percentile(lens, 90):.0f} "
          f"max={lens.max()} over={over}/{len(lens)} (those drop demo pairs)")


def check_max_new_tokens():
    """A 30x30 reply must fit in the decode budget."""
    class FakeTok:
        def encode(self, text):
            return [0] * len(text.replace("<|im_end|>", "\x00"))
    n = QwenFormatter(tokenizer=FakeTok()).max_new_tokens()
    check("max_new_tokens covers a full 30x30 grid", n >= 30 * 30 + 30,
          f"{n} tokens")


def check_completion_masking():
    """The answer grids must be supervised; the question grids must not.

    Regression test for the NaN-loss bug: NVARC keyed off the `user`/`assistant`
    token ids, which this checkpoint's tokenizer drops during encoding, so every
    label was masked and the loss was NaN over zero targets.
    """
    from arc_mask import completion_labels

    IM_START, NL, EOS = 14, 10, 15
    USER, ASSIST = 11, 12

    def turn(role_id, content):
        ids = [IM_START]
        if role_id is not None:
            ids.append(role_id)
        return ids + [NL] + content + [EOS]

    q1, a1 = [1, 2, NL, 3, 4], [5, 6]
    q2, a2 = [7, 8], [9, 0, NL, 1]

    for label, role_u, role_a in [("with role words", USER, ASSIST),
                                  ("role words dropped", None, None)]:
        ids = (turn(role_u, q1) + turn(role_a, a1)
               + turn(role_u, q2) + turn(role_a, a2))
        lab = completion_labels(ids, IM_START, EOS, NL)

        supervised = [int(v) for v in lab if v != -100]
        expected = a1 + [EOS] + a2 + [EOS]
        check(f"masking supervises exactly the answers ({label})",
              supervised == expected, f"got {supervised}, want {expected}")

        # No prompt cell may leak into the targets.
        pos_q = set()
        i = 0
        for content, is_answer in [(q1, False), (a1, True), (q2, False), (a2, True)]:
            head = 2 if role_u is not None else 1
            i += head + 1
            if not is_answer:
                pos_q.update(range(i, i + len(content)))
            i += len(content) + 1
        check(f"no prompt token is supervised ({label})",
              all(lab[p] == -100 for p in sorted(pos_q) if p < len(lab)))

    # A malformed sequence must not raise, just supervise nothing.
    lab = completion_labels([IM_START, NL, 1, 2], IM_START, EOS, NL)
    check("unterminated turn masks everything rather than crashing",
          all(v == -100 for v in lab))


def check_end_to_end(data_dir):
    """Fabricate DFS candidates and run the real aggregation path.

    Exercises load_decoded_results -> run_selection_algo -> get_submission ->
    fill_submission -> validate_format -> validate_submission without a GPU.
    """
    import bz2
    import pickle
    import shutil
    import tempfile

    import arc_config
    import make_submission

    ev_path = os.path.join(data_dir, "arc-agi_evaluation_challenges.json")
    sol_path = os.path.join(data_dir, "arc-agi_evaluation_solutions.json")
    ev = load(data_dir, "arc-agi_evaluation_challenges.json")
    sol = load(data_dir, "arc-agi_evaluation_solutions.json")

    tmp = tempfile.mkdtemp()
    real_out, real_sub = arc_config.OUTPUT_DIR, arc_config.SUBMISSION_PATH
    try:
        arc_config.OUTPUT_DIR = os.path.join(tmp, "inference_outputs")
        arc_config.SUBMISSION_PATH = os.path.join(tmp, "submission.json")
        os.makedirs(arc_config.OUTPUT_DIR)

        # Plant the correct answer for the first 10 tasks, a decoy for the next 10,
        # and nothing at all for the rest -- so we also prove that tasks with no
        # candidates still emit a valid (wrong) entry rather than vanishing.
        winners = sorted(ev)[:10]
        losers = sorted(ev)[10:20]
        for key in winners + losers:
            for i in range(len(ev[key]["test"])):
                truth = np.array(sol[key][i])
                good = key in winners
                cand = truth if good else np.zeros_like(truth)
                # score_kgmon ranks by hits - mean(score_aug); lower NLL is better.
                payload = [{"beam_score": 0.01, "score_aug": [0.5] * 8,
                            "solution": cand},
                           {"beam_score": 2.0, "score_aug": [4.0] * 8,
                            "solution": np.ones_like(truth)}]
                name = f"{key}_{i}.transpose.permute0123456789.ex012"
                with bz2.BZ2File(os.path.join(arc_config.OUTPUT_DIR, name), "w") as f:
                    pickle.dump(payload, f)

        submission = make_submission.build(
            selector="kgmon", challenges_path=ev_path,
            solutions_path=sol_path, out_path=arc_config.SUBMISSION_PATH)

        with open(arc_config.SUBMISSION_PATH) as f:
            reloaded = json.load(f)
        check("submission round-trips through JSON", reloaded.keys() == submission.keys())
        check("every eval task present after aggregation", set(reloaded) == set(ev))

        data = ArcDataset.from_file(ev_path).load_replies(sol_path)
        score = data.validate_submission(reloaded)
        check("planted correct answers score exactly", abs(score - len(winners)) < 1e-6,
              f"scored {score:.2f}, expected {len(winners)}.00")

        untouched = sorted(ev)[20]
        check("task with no candidates still emits both attempts",
              reloaded[untouched][0]["attempt_1"] == [[0]]
              and reloaded[untouched][0]["attempt_2"] == [[0]])
    finally:
        arc_config.OUTPUT_DIR, arc_config.SUBMISSION_PATH = real_out, real_sub
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.path.join(HERE, os.pardir, "example"))
    a = p.parse_args()
    data_dir = os.path.abspath(a.data_dir)
    print(f"*** data: {data_dir}\n")

    check_data_split_trap(data_dir)
    print()
    check_submission_format(data_dir)
    print()
    check_augmentation_roundtrip(data_dir)
    print()
    check_augmentation_order(data_dir)
    print()
    check_sequence_lengths(data_dir)
    check_max_new_tokens()
    print()
    check_completion_masking()
    print()
    check_end_to_end(data_dir)

    print()
    if FAILURES:
        print(f"*** {len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("*** all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
