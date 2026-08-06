import os
import time
import json
import torch
import argparse
import torch.multiprocessing as mp

import arc_config


def local_worker(rank, queue, end_time):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    torch.set_default_device("cpu")

    # Fix Unsloth patching issue: ranks must import serially, not concurrently.
    flag = os.path.join(arc_config.WORKER_FLAG_DIR, f"worker{rank}")
    prev = os.path.join(arc_config.WORKER_FLAG_DIR, f"worker{rank-1}")
    if rank > 0:
        while not os.path.exists(prev):
            time.sleep(5)

    from arc_solver import worker

    with open(flag, "w") as f:
        f.write("Ok")

    print(f"[Rank {rank}] start!", flush=True)

    worker(rank, queue, end_time)

    print(f"[Rank {rank}] done!", flush=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    args = parser.parse_args()

    end_time = args.end_time or arc_config.deadline()

    rerun_mode = arc_config.is_rerun()

    with open(arc_config.TEST_CHALLENGES, "r") as f:
        data = json.load(f)

    os.makedirs(arc_config.WORKER_FLAG_DIR, exist_ok=True)
    os.makedirs(arc_config.OUTPUT_DIR, exist_ok=True)

    keys = sorted(data.keys())
    if not rerun_mode:
        # Commit run: the shipped test file is a placeholder of training tasks,
        # so solving all of it is both meaningless and slow. Solve a few to
        # prove the pipeline, and let make_submission.py fill in the rest.
        keys = [k for k in keys if k in arc_config.SMOKE_TEST_KEYS] or keys[:4]

    print(
        f"*** rerun={rerun_mode} tasks={len(keys)}/{len(data)} "
        f"workers={arc_config.NUM_WORKERS} "
        f"budget={(end_time - time.time()) / 3600:.2f}h",
        flush=True,
    )

    queue = mp.Manager().Queue()
    for key in keys:
        queue.put(key)
    for _ in range(arc_config.NUM_WORKERS):
        queue.put(None)

    mp.spawn(local_worker, args=(queue, end_time), nprocs=arc_config.NUM_WORKERS)
