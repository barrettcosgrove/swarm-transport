"""
tools/threat_probe.py

Separates the two remaining explanations for variant C's 19% capture rate.

  1. cos(action, away from predator), split by closing rate rather than
     distance. Distance-binned cosine is ambiguous: an agent pushing toward
     a goal that sits past the predator scores negative even when fleeing
     is the wrong answer. Closing > 0.2 is the population where fleeing is
     correct. Sign there is whether evasion exists as a behaviour.
  2. For each capture, the minimum predator distance in the 20 steps before
     it. Fraction below the 1.609 escape floor is whether those captures
     were already committed (standoff problem) or still evadable
     (representation problem).
  3. Per-term reward RMS and payload contact force, carried over so a
     rerun still has the old context.

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
LOOKBACK = 20
CLOSING_ON = scenario.PREDATOR_CLOSING_ON


def _spin_up_closure(config):
    """Net ground a predator at its speed cap makes up while an agent
    accelerates from rest. Same integration as tests/test_reward_invariants.
    """
    a = config.agent_max_thrust / config.agent_mass
    k = config.agent_drag_coef / config.agent_mass
    v = 0.0
    closure = 0.0
    worst = 0.0
    for _ in range(200):
        v = min(v + (a - k * v) * config.dt, config.agent_max_speed)
        closure += (config.predator_max_speed - v) * config.dt
        worst = max(worst, closure)
    return worst


def probe(config, policy, seeds=(0, 1, 2), n_steps=None):
    n_steps = n_steps or 2 * config.max_steps
    escape_floor = config.predator_capture_radius + _spin_up_closure(config)
    cos_acc = {i: [0.0, 0] for i in range(len(DIST_BINS))}
    thrust_acc = {i: [0.0, 0] for i in range(len(DIST_BINS))}
    closing_cos = {"on": [0.0, 0], "off": [0.0, 0]}
    closing_thrust = {"on": [0.0, 0], "off": [0.0, 0]}
    term_sq = {}
    term_spread = {}
    f_agent = f_pred = f_agent_goal = f_pred_goal = 0.0
    hits = 0
    episodes = 0
    steps = 0
    capture_mins = []

    for seed in seeds:
        cfg = dataclasses.replace(config, seed=seed)
        env = Env(cfg)
        env.reset()
        hist = []

        for _ in range(n_steps):
            ws0, ss0 = env.world_state, env.scenario_state
            actions = policy(ws0, ss0, cfg)

            # measured on the pre-step state: this is the action the policy
            # chose given that geometry
            cosine, _ = scenario.predator_evade_masks(
                ws0, actions, cfg.predator_danger_radius)
            _, _, closing = scenario.predator_closing(ws0, cfg)
            d = (ws0.predator_pos.unsqueeze(1) - ws0.agent_pos).norm(dim=-1)
            a_norm = actions.norm(dim=-1)

            for i, (lo, hi) in enumerate(DIST_BINS):
                m = (d >= lo) & (d < hi)
                if bool(m.any()):
                    cos_acc[i][0] += float(cosine[m].sum())
                    cos_acc[i][1] += int(m.sum())
                    thrust_acc[i][0] += float(a_norm[m].sum())
                    thrust_acc[i][1] += int(m.sum())

            on = closing > CLOSING_ON
            off = closing <= 0.0
            for key, mask in (("on", on), ("off", off)):
                if bool(mask.any()):
                    closing_cos[key][0] += float(cosine[mask].sum())
                    closing_cos[key][1] += int(mask.sum())
                    closing_thrust[key][0] += float(a_norm[mask].sum())
                    closing_thrust[key][1] += int(mask.sum())

            min_dist = d.min(-1).values
            hist.append(min_dist.clone())
            if len(hist) > LOOKBACK:
                hist.pop(0)

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
            ended = terminated | truncated
            captured = terminated & info["captured"]
            if bool(captured.any()):
                pre_min = torch.stack(hist).min(0).values
                capture_mins += pre_min[captured].tolist()
            if bool(ended.any()):
                wipe = torch.full_like(min_dist, 1e9)
                hist[:] = [torch.where(ended, wipe, t) for t in hist]
            episodes += int(ended.sum())
            steps += cfg.num_envs

    return dict(cos=cos_acc, thrust=thrust_acc, term_sq=term_sq,
                term_spread=term_spread, steps=steps, hits=hits,
                episodes=max(episodes, 1), n_agents=config.n_agents,
                f_agent=f_agent, f_pred=f_pred,
                f_agent_goal=f_agent_goal, f_pred_goal=f_pred_goal,
                closing_cos=closing_cos, closing_thrust=closing_thrust,
                capture_mins=capture_mins, escape_floor=escape_floor)


def _mean_pair(pair):
    total, n = pair
    return total / n if n else float("nan")


def report(r, label):
    print(f"\n=== {label} ===")
    print(f"{'agent->predator':>16}{'n':>10}{'cos(away)':>11}{'|thrust|':>10}")
    for i, (lo, hi) in enumerate(DIST_BINS):
        s, n = r["cos"][i]
        if n == 0:
            continue
        print(f"{f'{lo:.1f}-{hi:.1f}':>16}{n:>10}{s / n:>11.3f}"
              f"{r['thrust'][i][0] / r['thrust'][i][1]:>10.3f}")

    print(f"  closing-conditioned cosine (fleeing is correct only when closing > {CLOSING_ON})")
    print(f"    {'pop':<10}{'n':>10}{'cos(away)':>11}{'|thrust|':>10}")
    for key, name in (("on", "closing"), ("off", "not closing")):
        n = r["closing_cos"][key][1]
        print(f"    {name:<10}{n:>10}{_mean_pair(r['closing_cos'][key]):>11.3f}"
              f"{_mean_pair(r['closing_thrust'][key]):>10.3f}")

    mins = r["capture_mins"]
    floor = r["escape_floor"]
    if mins:
        t = torch.tensor(mins)
        below = float((t < floor).float().mean())
        print(f"  captures {len(mins)}  pre-capture min pred-dist  "
              f"p50 {float(t.median()):.2f}  p10 {float(t.quantile(0.1)):.2f}  "
              f"below escape floor {floor:.3f}: {below:.0%}")
    else:
        print("  captures 0")

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
    for seed in (0, 1, 2):
        path = f"train/checkpoints/variant_d_danger_radius/seed_{seed}/checkpoint_best.pt"
        report(probe(config, load_actor(path, config)), f"variant_d seed_{seed}")
    report(probe(config, scripted_policy_factory()), "scripted")
