"""
tools/train_seeds.py

Independent MAPPO runs across several seeds, written to one JSON.

C's recipe: obs_dim 32, closing-gated threat at radius 2.5, camp off,
flee off, approach always on. 7 seeds x 400 iterations. Do not point
it at outputs/variant_c_closing_threat.json -- that 3x250 log is the
matched baseline.

Usage:
    python -m tools.train_seeds
"""
import dataclasses
import json
import os

from train.config import Config
from train.mappo import MAPPOTrainer


SEEDS = (0, 1, 2, 3, 4, 5, 6)
LOG_PATH = "outputs/variant_c_400.json"
CHECKPOINT_DIR = "train/checkpoints/variant_c_400"


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
            num_iterations=400,
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
