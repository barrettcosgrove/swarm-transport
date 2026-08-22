"""
tools/train_seeds.py

Independent MAPPO runs across several seeds, written to one JSON.

Each seed is a fresh Env / actor / critic / optimizer. The file is a list of
{"seed": int, "iterations": [record, ...]} objects -- the same per-iteration
records a single run would log, grouped by seed. Rewritten every 10 iterations
of the current seed, so a crash still keeps completed seeds and the in-progress
seed so far.

This is the matched 250-iteration control on the current reward. Do not compare
it to a 400-iteration run: num_iterations is the LR-anneal denominator.

Usage:
    python -m tools.train_seeds
"""
import dataclasses
import json
import os

from train.config import Config
from train.mappo import MAPPOTrainer


SEEDS = (0, 1, 2)
LOG_PATH = "outputs/control_250.json"
CHECKPOINT_DIR = "train/checkpoints/control_250"


def main():
    combined = []

    def write_combined():
        os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
        with open(LOG_PATH, "w") as f:
            json.dump(combined, f, indent=2)
        return LOG_PATH

    for seed in SEEDS:
        config = dataclasses.replace(
            Config(),
            seed=seed,
            num_iterations=250,
            log_path=LOG_PATH,
            checkpoint_dir=os.path.join(CHECKPOINT_DIR, f"seed_{seed}"),
        )
        trainer = MAPPOTrainer(config)
        combined.append({"seed": seed, "iterations": trainer.logger.history})
        trainer.logger.write = write_combined

        print(f"=== seed {seed} / {config.num_iterations} iterations ===", flush=True)
        trainer.train()

    print(f"wrote {LOG_PATH}")


if __name__ == "__main__":
    main()
