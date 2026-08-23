"""
tools/threat_probe.py

Follow-up to tools/freeze_probe.py, which found that variant A spends 63% of
every episode with the predator inside 1.5 of the payload against the scripted
controller's 22%, and sits on predator cooldown 89% of the time in the nearest
band. That says the predator is camping the crate, so the questions here are
whether the agents respond to it at all and whether the reward gives them a
reason to.

  1. cos(action, away from predator), binned by agent->predator distance. The
     sign of this is whether evasion exists as a behaviour. DESIGN's earlier
     diagnosis measured +0.087 -- drifting toward the thing killing it.
  2. Per-term reward RMS, split private/shared. Sets threat_coef against the
     terms it competes with, which is what decides whether a private "step
     away" signal can move the policy at all.
  3. Who is actually pushing the payload: agent contact force vs predator
     contact force, and each one's goalward component.

Usage:
    python -m tools.threat_probe
"""
import dataclasses

import torch

from env.env import Env
from env import scenario
from env.physics import circle_box_dynamic_forces
from train.config import Config
from tools.freeze_probe import load_actor, scripted_policy_factory

DIST_BINS = [(0.0, 0.4), (0.4, 1.0), (1.0, 2.0), (2.0, 99.0)]


def probe(config, policy, seeds=(0, 1, 2), n_steps=None):
    n_steps = n_steps or 2 * config.max_steps
    cos_acc = {i: [0.0, 0] for i in range(len(DIST_BINS))}
    thrust_acc = {i: [0.0, 0] for i in range(len(DIST_BINS))}
    term_sq = {}
    term_spread = {}
    f_agent = f_pred = f_agent_goal = f_pred_goal = 0.0
    hits = 0
    episodes = 0
    steps = 0

    for seed in seeds:
        cfg = dataclasses.replace(config, seed=seed)
        env = Env(cfg)
        env.reset()

        for _ in range(n_steps):
            ws0, ss0 = env.world_state, env.scenario_state
            actions = policy(ws0, ss0, cfg)

            # measured on the pre-step state: this is the action the policy
            # chose given that geometry
            to_pred = ws0.predator_pos.unsqueeze(1) - ws0.agent_pos
            d = to_pred.norm(dim=-1)
            away = -to_pred / d.unsqueeze(-1).clamp(min=1e-6)
            a_norm = actions.norm(dim=-1)
            cos = (actions * away).sum(-1) / a_norm.clamp(min=1e-6)

            for i, (lo, hi) in enumerate(DIST_BINS):
                m = (d >= lo) & (d < hi)
                if bool(m.any()):
                    cos_acc[i][0] += float(cos[m].sum())
                    cos_acc[i][1] += int(m.sum())
                    thrust_acc[i][0] += float(a_norm[m].sum())
                    thrust_acc[i][1] += int(m.sum())

            h_before = ss0.health.clone()
            _, _, terminated, truncated, info = env.step(actions, training_progress=1.0)
            ws, ss = info["world_state"], info["scenario_state"]

            for name, t in info["reward_terms"].items():
                term_sq[name] = term_sq.get(name, 0.0) + float((t ** 2).sum())
                # spread across agents within an env: 0 means a purely shared
                # term, which the policy gradient largely ignores
                term_spread[name] = term_spread.get(name, 0.0) + float(t.std(dim=-1).sum())

            _, on_payload_from_agents = circle_box_dynamic_forces(
                ws.agent_pos, ws.agent_radius, ws.payload_pos,
                ws.payload_halfsize, cfg.payload_stiffness)
            _, on_payload_from_pred = circle_box_dynamic_forces(
                ws.predator_pos, ws.predator_radius, ws.payload_pos,
                ws.payload_halfsize, cfg.payload_stiffness)
            gd = ss.goal_pos - ws.payload_pos
            gd = gd / gd.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            f_agent += float(on_payload_from_agents.norm(dim=-1).sum())
            f_pred += float(on_payload_from_pred.norm(dim=-1).sum())
            f_agent_goal += float((on_payload_from_agents * gd).sum(-1).sum())
            f_pred_goal += float((on_payload_from_pred * gd).sum(-1).sum())

            hits += int(((h_before - ss.health) > 0).sum())
            episodes += int((terminated | truncated).sum())
            steps += cfg.num_envs

    return dict(cos=cos_acc, thrust=thrust_acc, term_sq=term_sq,
                term_spread=term_spread, steps=steps, hits=hits,
                episodes=max(episodes, 1), n_agents=config.n_agents,
                f_agent=f_agent, f_pred=f_pred,
                f_agent_goal=f_agent_goal, f_pred_goal=f_pred_goal)


def report(r, label):
    print(f"\n=== {label} ===")
    print(f"{'agent->predator':>16}{'n':>10}{'cos(away)':>11}{'|thrust|':>10}")
    for i, (lo, hi) in enumerate(DIST_BINS):
        s, n = r["cos"][i]
        if n == 0:
            continue
        print(f"{f'{lo:.1f}-{hi:.1f}':>16}{n:>10}{s / n:>11.3f}"
              f"{r['thrust'][i][0] / r['thrust'][i][1]:>10.3f}")

    n = r["steps"] * r["n_agents"]
    print(f"  {'term':<12}{'rms':>9}{'agent spread':>14}")
    for name, sq in sorted(r["term_sq"].items(), key=lambda kv: -kv[1]):
        rms = (sq / n) ** 0.5
        print(f"  {name:<12}{rms:>9.3f}{r['term_spread'][name] / r['steps']:>14.3f}")

    print(f"  hits/episode {r['hits'] / r['episodes']:.1f}   episodes {r['episodes']}")
    tot = r["f_agent"] + r["f_pred"]
    print(f"  payload contact force: agents {r['f_agent'] / max(tot, 1e-9):.1%}"
          f"  predator {r['f_pred'] / max(tot, 1e-9):.1%}")
    print(f"  goalward component:    agents {r['f_agent_goal'] / r['steps']:+.3f}"
          f"  predator {r['f_pred_goal'] / r['steps']:+.3f}  per step")


if __name__ == "__main__":
    config = Config(num_envs=16)
    for seed in (1, 2):
        path = f"train/checkpoints/variant_a_progress_blame/seed_{seed}/checkpoint_best.pt"
        report(probe(config, load_actor(path, config)), f"variant_a seed_{seed}")
    report(probe(config, scripted_policy_factory()), "scripted")
