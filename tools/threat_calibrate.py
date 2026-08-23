"""
tools/threat_calibrate.py

Sizes the closing-rate threat term against the time penalty.

The previous sweep sized threat against progress RMS. Progress is a
zero-mean telescoping term whose episode total is capped near 315;
threat is a one-signed accumulating cost -- a sibling of the time
penalty, not of progress. That mismatch is how variant B landed at
-138 per episode (230% of reward_time) while looking like "6.5% of
the progress signal."

This sweep replays a variant A checkpoint (obs_dim is unchanged, so
the weights still load) and recomputes the closing-rate form under
candidate coefficients. Reported per candidate:

  episode-summed threat, as a fraction of reward_time
  duty cycle: fraction of agent-steps with nonzero threat
  mean agents per step paying threat

The target band is episode threat -12 to -18, about 20-30% of the
time penalty. Variant A was -0.4; variant B was -138.

Usage:
    python -m tools.threat_calibrate
"""
import dataclasses

import torch

from env.env import Env
from env import scenario
from train.config import Config
from tools.freeze_probe import load_actor

COEFS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)


def escape_feasibility(config):
    """Closest approach when an agent starts fleeing at full thrust from rest.

    An agent accelerating against linear drag reaches the predator's 3.5 speed
    cap eventually -- thrust/drag is 20 -- so it cannot be run down forever.
    What decides the outcome is how much ground the predator makes up during
    the spin-up. That net closure, not the raw gap, is what the danger radius
    has to cover.
    """
    print("=== escape feasibility ===")
    a = config.agent_max_thrust / config.agent_mass
    k = config.agent_drag_coef / config.agent_mass
    v = 0.0
    closure = 0.0
    worst = 0.0
    for _ in range(200):
        v = min(v + (a - k * v) * config.dt, config.agent_max_speed)
        closure += (config.predator_max_speed - v) * config.dt
        worst = max(worst, closure)
    print(f"  predator {config.predator_max_speed:.2f}, agent thrust/drag "
          f"{a / k:.1f}; net closure during spin-up {worst:.3f}")
    print(f"  so escaping needs a reaction radius > capture "
          f"{config.predator_capture_radius:.2f} + {worst:.3f} = "
          f"{config.predator_capture_radius + worst:.3f}")
    print(f"{'radius':>8}{'warn steps':>12}{'closest':>10}{'verdict':>10}")
    for radius in (1.0, 1.5, 2.0, 2.5, 3.0):
        closest = radius - worst
        steps = (radius - config.predator_capture_radius) / (config.predator_max_speed * config.dt)
        print(f"{radius:>8.1f}{steps:>12.1f}{closest:>10.3f}"
              f"{'escapes' if closest > config.predator_capture_radius else 'caught':>10}")


def sweep(config, policy, seeds=(0, 1), n_steps=None):
    n_steps = n_steps or config.max_steps
    unit_ep = 0.0
    time_ep = 0.0
    duty_n = duty_hit = 0
    agents_paying = 0.0
    env_steps = 0
    episodes = 0
    running_unit = None
    running_time = None

    for seed in seeds:
        cfg = dataclasses.replace(config, seed=seed)
        env = Env(cfg)
        env.reset()
        E, N = cfg.num_envs, cfg.n_agents
        running_unit = torch.zeros(E)
        running_time = torch.zeros(E)
        for _ in range(n_steps):
            actions = policy(env.world_state, env.scenario_state, cfg)
            _, _, terminated, truncated, info = env.step(actions, training_progress=1.0)
            ws = info["world_state"]
            _, intrusion, closing = scenario.predator_closing(ws, cfg)
            unit = -(intrusion.pow(2) * closing)
            running_unit += unit.mean(-1)
            running_time += info["reward_terms"]["time"].mean(-1)
            live = unit != 0
            duty_hit += int(live.sum())
            duty_n += E * N
            agents_paying += float(live.float().sum(-1).mean())
            env_steps += 1
            ended = terminated | truncated
            if bool(ended.any()):
                unit_ep += float(running_unit[ended].sum())
                time_ep += float(running_time[ended].sum())
                episodes += int(ended.sum())
                running_unit = torch.where(ended, torch.zeros_like(running_unit), running_unit)
                running_time = torch.where(ended, torch.zeros_like(running_time), running_time)

    duty = duty_hit / max(duty_n, 1)
    mean_agents = agents_paying / max(env_steps, 1)
    unit_mean = unit_ep / max(episodes, 1)
    time_mean = time_ep / max(episodes, 1)
    print("\n=== closing-rate threat sweep (variant A geometry) ===")
    print(f"  episodes {episodes}  duty {duty:.1%}  agents paying/step {mean_agents:.2f}")
    print(f"  reward_time {time_mean:.1f}  unit-coef threat {unit_mean:.2f}")
    print(f"{'coef':>7}{'ep threat':>12}{'% of time':>11}{'band':>10}")
    for coef in COEFS:
        ep = coef * unit_mean
        frac = ep / time_mean if time_mean else float("nan")
        lo, hi = -18.0, -12.0
        band = "target" if lo <= ep <= hi else ("low" if ep > hi else "high")
        print(f"{coef:>7.2f}{ep:>12.1f}{frac:>11.0%}{band:>10}")
    print(f"  MEASURED_THREAT_DUTY = {duty:.4f}")


if __name__ == "__main__":
    config = Config(num_envs=16)
    escape_feasibility(config)
    path = "train/checkpoints/variant_a_progress_blame/seed_1/checkpoint_best.pt"
    sweep(config, load_actor(path, config))
