"""
tools/evaluate.py

Runs a policy across several seeds and reports how episodes actually end.

Exists because "did it get close to the goal" is not enough to tune against.
A win rate alone cannot distinguish "agents are dying" from "agents are slow"
from "agents wander" -- those three call for completely different fixes, and
splitting losses into captures and timeouts separates them in one number.

The damage histogram is the other half, and it is what retired the original
force-proportional damage model: scaling that coefficient moved the whole
distribution but left its shape alone, so the p95 stayed near 3x the median at
every scale. A flat drain reads as a tail ratio of 1.0, which is the quickest
confirmation that health has become a predictable budget rather than a lottery.

Usage:
    from tools.evaluate import evaluate, format_report
    print(format_report(evaluate(config, scripted_policy)))

    python -m tools.evaluate
"""
import dataclasses
from dataclasses import dataclass

import torch


@dataclass
class EvalResult:
    wins: int
    captures: int
    timeouts: int
    episodes: int
    win_rate: float
    mean_episode_steps: float
    damage_events: int
    damage_median: float
    damage_p95: float
    damage_max: float
    end_health: float

    @property
    def damage_tail_ratio(self) -> float:
        """p95 / median. Near 1.0 means every hit costs about the same; large
        means a handful of hits do the real damage."""
        if self.damage_median <= 0.0:
            return float("nan")
        return self.damage_p95 / self.damage_median


def evaluate(config, policy, seeds=(0, 1, 2), n_steps=None):
    """Run `policy` for one rollout per seed and tally outcomes.

    Environments auto-reset inside env.step, so a single rollout yields many
    episodes per seed -- the counts below are over episodes, not seeds.

    Defaults to twice max_steps because compute_done reads step_count before
    env.step increments it: a rollout of exactly max_steps can never truncate,
    which silently reports zero timeouts and drops every unfinished episode
    from the tally.
    """
    from env.env import Env

    n_steps = n_steps or 2 * config.max_steps
    wins = captures = timeouts = 0
    episode_lengths = []
    damage = []
    end_health = []

    for seed in seeds:
        cfg = dataclasses.replace(config, seed=seed)
        env = Env(cfg)
        env.reset()

        for _ in range(n_steps):
            world_state, scenario_state = env.world_state, env.scenario_state
            health_before = scenario_state.health.clone()

            actions = policy(world_state, scenario_state, cfg)
            _, _, terminated, truncated, info = env.step(actions, training_progress=1.0)

            at_goal = info["success"]
            wins += int((terminated & at_goal).sum())
            captures += int((terminated & ~at_goal).sum())
            timeouts += int((truncated & ~terminated).sum())

            ended = terminated | truncated
            if bool(ended.any()):
                episode_lengths += info["scenario_state"].step_count[ended].tolist()

            lost = health_before - info["scenario_state"].health
            damage += lost[lost > 0].tolist()

        end_health.append(float(env.scenario_state.health.mean()))

    episodes = wins + captures + timeouts
    d = torch.tensor(damage) if damage else torch.zeros(1)

    return EvalResult(
        wins=wins,
        captures=captures,
        timeouts=timeouts,
        episodes=episodes,
        win_rate=wins / episodes if episodes else 0.0,
        mean_episode_steps=sum(episode_lengths) / len(episode_lengths) if episode_lengths else float("nan"),
        damage_events=len(damage),
        damage_median=float(d.median()),
        damage_p95=float(d.quantile(0.95)),
        damage_max=float(d.max()),
        end_health=sum(end_health) / len(end_health),
    )


HEADER = (f"{'variant':<34}{'win%':>6}{'win':>5}{'cap':>5}{'time':>6}"
          f"{'len':>7}{'dmg med':>9}{'p95':>7}{'max':>7}{'tail':>6}")


def format_report(result, label=""):
    return (f"{label:<34}{result.win_rate:>5.0%}{result.wins:>5}{result.captures:>5}"
            f"{result.timeouts:>6}{result.mean_episode_steps:>7.0f}"
            f"{result.damage_median:>9.1f}{result.damage_p95:>7.1f}"
            f"{result.damage_max:>7.1f}{result.damage_tail_ratio:>6.1f}")


if __name__ == "__main__":
    from train.config import Config
    from tools.scripted_policy import scripted_policy

    config = Config(num_envs=16)
    print(HEADER)
    print(format_report(evaluate(config, scripted_policy), "current config"))
