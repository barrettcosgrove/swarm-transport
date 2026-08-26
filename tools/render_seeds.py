"""
tools/render_seeds.py

Renders the best policy of each variant C 400-iter seed as its own GIF.

Usage:
    python -m tools.render_seeds
"""
import torch

from env.env import Env
from env import scenario
from train.config import Config
from train.checkpoints import load_checkpoint
from train.mappo import Actor, Critic
from tools.render import record_episode, render_to_gif


SEEDS = (0, 1, 2, 3, 4, 5, 6)
CHECKPOINT_DIR = "train/checkpoints/variant_c_400"


def main():
    for seed in SEEDS:
        path = f"{CHECKPOINT_DIR}/seed_{seed}/checkpoint_best.pt"
        out = f"outputs/actor_variant_c_400_seed_{seed}.gif"

        config = Config(num_envs=4, seed=seed)
        actor = Actor(config.obs_dim, 2, config.hidden_dim)
        critic = Critic(config.obs_dim, config.n_agents, config.hidden_dim)
        load_checkpoint(path, actor, critic)
        actor.eval()

        def actor_policy(world_state, scenario_state, cfg, actor=actor):
            with torch.no_grad():
                return actor(scenario.observe(world_state, scenario_state, cfg)).clamp(-1.0, 1.0)

        env = Env(config)
        frames = record_episode(env, actor_policy, n_steps=config.max_steps)
        written = render_to_gif(frames, out, config, fps=8, every=2, hold_seconds=1.5)
        print(f"wrote {written}")


if __name__ == "__main__":
    main()
