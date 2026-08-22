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

Behavioural fields (position_ratio, push_efficiency, payload progress, predator
distance) use the same helpers as train/mappo.py so a periodic eval and an
iteration record cannot disagree on the metrics that gate the next change.

Usage:
    from tools.evaluate import evaluate, format_report
    print(format_report(evaluate(config, policy)))

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
    capture_rate: float
    timeout_rate: float
    mean_episode_steps: float
    mean_win_steps: float
    mean_capture_steps: float
    mean_timeout_steps: float
    damage_events: int
    damage_median: float
    damage_p95: float
    damage_max: float
    mean_end_health: float
    position_ratio: float
    push_efficiency: float
    mean_payload_progress: float
    mean_min_predator_dist: float
    mean_team_spread: float

    @property
    def damage_tail_ratio(self) -> float:
        """p95 / median. Near 1.0 means every hit costs about the same; large
        means a handful of hits do the real damage."""
        if self.damage_median <= 0.0:
            return float("nan")
        return self.damage_p95 / self.damage_median

    def as_log(self):
        """Flat dict for the training JSON, prefixed so it cannot collide with
        the noisier per-iteration success_rate from the rollout."""
        return {
            "eval_win_rate": self.win_rate,
            "eval_capture_rate": self.capture_rate,
            "eval_timeout_rate": self.timeout_rate,
            "eval_wins": self.wins,
            "eval_captures": self.captures,
            "eval_timeouts": self.timeouts,
            "eval_episodes": self.episodes,
            "eval_mean_episode_steps": self.mean_episode_steps,
            "eval_mean_win_steps": self.mean_win_steps,
            "eval_mean_capture_steps": self.mean_capture_steps,
            "eval_mean_timeout_steps": self.mean_timeout_steps,
            "eval_mean_end_health": self.mean_end_health,
            "eval_position_ratio": self.position_ratio,
            "eval_push_efficiency": self.push_efficiency,
            "eval_mean_payload_progress": self.mean_payload_progress,
            "eval_mean_min_predator_dist": self.mean_min_predator_dist,
            "eval_mean_team_spread": self.mean_team_spread,
        }


def _mean(values):
    return sum(values) / len(values) if values else float("nan")


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
    from env import scenario

    n_steps = n_steps or 2 * config.max_steps
    wins = captures = timeouts = 0
    win_lengths, capture_lengths, timeout_lengths = [], [], []
    damage = []
    end_health = []
    payload_progress = []
    behind = front = 0
    force_goal = force_mag = 0.0
    pred_sum = spread_sum = 0.0
    n_beh = 0

    for seed in seeds:
        cfg = dataclasses.replace(config, seed=seed)
        env = Env(cfg)
        env.reset()
        start_dist = torch.norm(
            env.world_state.payload_pos - env.scenario_state.goal_pos, dim=-1)

        for _ in range(n_steps):
            world_state, scenario_state = env.world_state, env.scenario_state
            health_before = scenario_state.health.clone()

            actions = policy(world_state, scenario_state, cfg)
            _, _, terminated, truncated, info = env.step(actions, training_progress=1.0)

            ws = info["world_state"]
            ss = info["scenario_state"]
            behind_mask, front_mask = scenario.payload_side_masks(
                ws, ss, cfg.push_zone_radius)
            behind += int(behind_mask.sum())
            front += int(front_mask.sum())
            goalward, magnitude = scenario.payload_goalward_force(ws, ss, cfg)
            force_goal += goalward.sum().item()
            force_mag += magnitude.sum().item()
            pred_sum += (ws.agent_pos - ws.predator_pos.unsqueeze(1)).norm(dim=-1).min(-1).values.sum().item()
            centroid = ws.agent_pos.mean(dim=1, keepdim=True)
            spread_sum += (ws.agent_pos - centroid).norm(dim=-1).mean(-1).sum().item()
            n_beh += ws.payload_pos.shape[0]

            at_goal = info["success"]
            win_mask = terminated & at_goal
            cap_mask = terminated & ~at_goal
            to_mask = truncated & ~terminated
            wins += int(win_mask.sum())
            captures += int(cap_mask.sum())
            timeouts += int(to_mask.sum())

            ended = terminated | truncated
            if bool(ended.any()):
                steps = info["scenario_state"].step_count
                win_lengths += steps[win_mask].tolist()
                capture_lengths += steps[cap_mask].tolist()
                timeout_lengths += steps[to_mask].tolist()
                end_health += ss.health[ended].tolist()
                payload_progress += (start_dist[ended] - info["payload_dist"][ended]).tolist()
                start_dist = torch.where(
                    ended,
                    torch.norm(env.world_state.payload_pos - env.scenario_state.goal_pos, dim=-1),
                    start_dist,
                )

            lost = health_before - ss.health
            damage += lost[lost > 0].tolist()

    episodes = wins + captures + timeouts
    d = torch.tensor(damage) if damage else torch.zeros(1)

    return EvalResult(
        wins=wins,
        captures=captures,
        timeouts=timeouts,
        episodes=episodes,
        win_rate=wins / episodes if episodes else 0.0,
        capture_rate=captures / episodes if episodes else 0.0,
        timeout_rate=timeouts / episodes if episodes else 0.0,
        mean_episode_steps=_mean(win_lengths + capture_lengths + timeout_lengths),
        mean_win_steps=_mean(win_lengths),
        mean_capture_steps=_mean(capture_lengths),
        mean_timeout_steps=_mean(timeout_lengths),
        damage_events=len(damage),
        damage_median=float(d.median()),
        damage_p95=float(d.quantile(0.95)),
        damage_max=float(d.max()),
        mean_end_health=_mean(end_health),
        position_ratio=behind / max(front, 1),
        push_efficiency=force_goal / force_mag if force_mag else 0.0,
        mean_payload_progress=_mean(payload_progress),
        mean_min_predator_dist=pred_sum / n_beh if n_beh else float("nan"),
        mean_team_spread=spread_sum / n_beh if n_beh else float("nan"),
    )


HEADER = (f"{'variant':<28}{'win%':>6}{'cap%':>6}{'to%':>6}"
          f"{'win':>5}{'cap':>5}{'to':>5}{'len':>6}"
          f"{'posR':>6}{'pshE':>6}{'prog':>7}{'pred':>6}{'hp':>6}")


def format_report(result, label=""):
    return (f"{label:<28}{result.win_rate:>5.0%}{result.capture_rate:>5.0%}{result.timeout_rate:>5.0%}"
            f"{result.wins:>5}{result.captures:>5}{result.timeouts:>5}"
            f"{result.mean_episode_steps:>6.0f}"
            f"{result.position_ratio:>6.2f}{result.push_efficiency:>6.1%}"
            f"{result.mean_payload_progress:>7.2f}"
            f"{result.mean_min_predator_dist:>6.2f}"
            f"{result.mean_end_health:>6.0f}")


if __name__ == "__main__":
    from env import scenario
    from train.config import Config
    from train.checkpoints import load_checkpoint
    from train.mappo import Actor, Critic

    CHECKPOINT = "train/checkpoints/full_250_push_reversion/checkpoint_250.pt"

    config = Config(num_envs=16)
    actor = Actor(config.obs_dim, 2, config.hidden_dim)
    critic = Critic(config.obs_dim, config.n_agents, config.hidden_dim)
    load_checkpoint(CHECKPOINT, actor, critic)
    actor.eval()

    def actor_policy(world_state, scenario_state, cfg):
        with torch.no_grad():
            return actor(scenario.observe(world_state, scenario_state, cfg)).clamp(-1.0, 1.0)

    print(HEADER)
    print(format_report(evaluate(config, actor_policy), "full_250_push_reversion"))
