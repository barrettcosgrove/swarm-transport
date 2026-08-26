"""
tools/freeze_probe.py

Throwaway diagnostic for two questions the eval table cannot answer:

  1. Why do agents stop moving when the predator comes near the payload?
     Binned by predator->payload distance, reports what the agents near the
     payload are actually doing: thrust magnitude, whether they stay inside
     push_zone_radius, and what the progress term pays them there. The
     zone-attributed progress reward is the suspect -- it bills whoever is in
     the zone for payload motion regardless of who caused it, and the predator
     pushes the payload too (physics.step, force_payload_from_predator).

  2. Would a larger push_coef make the payload overshoot the goal? push_coef
     is a reward coefficient, so the honest bound is a physics question: drive
     every agent at full thrust into the payload and see how fast it can
     possibly go, then compare a step of that against the success region.

Usage:
    python -m tools.freeze_probe
"""
import dataclasses

import torch

from env.env import Env
from env import scenario
from train.config import Config
from train.checkpoints import load_checkpoint
from train.mappo import Actor, Critic

BINS = [(0.0, 0.75), (0.75, 1.5), (1.5, 3.0), (3.0, 99.0)]
NEAR_PAYLOAD = 1.0        # "at the crate", for the thrust measurement
CONTACT = 0.42            # payload circumradius 0.283 + agent radius 0.1, rounded


def load_actor(path, config):
    actor = Actor(config.obs_dim, 2, config.hidden_dim)
    critic = Critic(config.obs_dim, config.n_agents, config.hidden_dim)
    load_checkpoint(path, actor, critic)
    actor.eval()

    def policy(world_state, scenario_state, cfg):
        with torch.no_grad():
            return actor(scenario.observe(world_state, scenario_state, cfg)).clamp(-1.0, 1.0)
    return policy


def scripted_policy_factory():
    from tools.scripted_policy import scripted_policy
    return lambda ws, ss, cfg: scripted_policy(scenario.observe(ws, ss, cfg), cfg)


def probe(config, policy, seeds=(0, 1, 2), n_steps=None):
    n_steps = n_steps or 2 * config.max_steps
    acc = {i: {k: 0.0 for k in
               ("steps", "thrust", "thrust_n", "in_zone", "agents", "contact",
                "pay_speed", "goalward", "prog_zone", "prog_zone_n", "prog_neg",
                "pred_touch", "cooldown", "ring")}
           for i in range(len(BINS))}

    payload_speeds = []
    overshoot = {"win": 0, "reached_close_but_lost": 0, "episodes": 0}

    for seed in seeds:
        cfg = dataclasses.replace(config, seed=seed)
        env = Env(cfg)
        env.reset()
        min_dist = torch.full((cfg.num_envs,), 1e9)

        for _ in range(n_steps):
            actions = policy(env.world_state, env.scenario_state, cfg)
            _, _, terminated, truncated, info = env.step(actions, training_progress=1.0)
            ws, ss = info["world_state"], info["scenario_state"]

            pred_pay = (ws.predator_pos - ws.payload_pos).norm(dim=-1)          # (E,)
            a_pay = (ws.agent_pos - ws.payload_pos.unsqueeze(1)).norm(dim=-1)   # (E, N)
            thrust = actions.norm(dim=-1)                                       # (E, N)
            in_zone = a_pay <= cfg.push_zone_radius
            ring = (a_pay > cfg.push_zone_radius) & (a_pay <= cfg.push_zone_radius + 0.3)
            near = a_pay <= NEAR_PAYLOAD
            prog = info["reward_terms"]["progress"]                             # (E, N)

            goal_dir = ss.goal_pos - ws.payload_pos
            goal_dir = goal_dir / goal_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            pay_speed = ws.payload_vel.norm(dim=-1)
            goalward = (ws.payload_vel * goal_dir).sum(-1)
            payload_speeds.append(pay_speed)

            # did the predator's own body touch the payload this step
            pred_touch = pred_pay <= (cfg.predator_radius + 0.29)

            for i, (lo, hi) in enumerate(BINS):
                m = (pred_pay >= lo) & (pred_pay < hi)                          # (E,)
                if not bool(m.any()):
                    continue
                a = acc[i]
                a["steps"] += int(m.sum())
                nm = near & m.unsqueeze(1)
                a["thrust"] += float(thrust[nm].sum())
                a["thrust_n"] += int(nm.sum())
                a["in_zone"] += float(in_zone[m.unsqueeze(1).expand_as(in_zone)].sum())
                a["ring"] += float(ring[m.unsqueeze(1).expand_as(ring)].sum())
                a["contact"] += float((a_pay <= CONTACT)[m.unsqueeze(1).expand_as(in_zone)].sum())
                a["agents"] += int(m.sum()) * cfg.n_agents
                a["pay_speed"] += float(pay_speed[m].sum())
                a["goalward"] += float(goalward[m].sum())
                zm = in_zone & m.unsqueeze(1)
                a["prog_zone"] += float(prog[zm].sum())
                a["prog_zone_n"] += int(zm.sum())
                a["prog_neg"] += float(prog[zm].clamp(max=0.0).sum())
                a["pred_touch"] += int((pred_touch & m).sum())
                a["cooldown"] += int(((ss.predator_cooldown > 0) & m).sum())

            dist = info["payload_dist"]
            min_dist = torch.minimum(min_dist, dist)
            ended = terminated | truncated
            if bool(ended.any()):
                won = info["success"] & ended
                overshoot["episodes"] += int(ended.sum())
                overshoot["win"] += int(won.sum())
                overshoot["reached_close_but_lost"] += int(
                    (ended & ~info["success"] & (min_dist < 1.5)).sum())
                min_dist = torch.where(ended, torch.full_like(min_dist, 1e9), min_dist)

    speeds = torch.cat(payload_speeds)
    return acc, speeds, overshoot


def report(acc, speeds, overshoot, label):
    print(f"\n=== {label} ===")
    print(f"{'pred->payload':>14}{'steps':>8}{'|thrust|':>9}{'zone%':>7}{'ring%':>7}"
          f"{'touch%':>8}{'payV':>7}{'goalV':>7}{'progZ':>8}{'neg%':>7}"
          f"{'predHit%':>9}{'cool%':>7}")
    for i, (lo, hi) in enumerate(BINS):
        a = acc[i]
        if a["steps"] == 0:
            continue
        s, ag = a["steps"], max(a["agents"], 1)
        pz = a["prog_zone"] / max(a["prog_zone_n"], 1)
        neg = a["prog_neg"] / a["prog_zone"] if a["prog_zone"] else float("nan")
        print(f"{f'{lo:.2f}-{hi:.2f}':>14}{s:>8}"
              f"{a['thrust'] / max(a['thrust_n'], 1):>9.3f}"
              f"{a['in_zone'] / ag:>7.1%}{a['ring'] / ag:>7.1%}"
              f"{a['contact'] / ag:>8.1%}"
              f"{a['pay_speed'] / s:>7.3f}{a['goalward'] / s:>7.3f}"
              f"{pz:>8.2f}{neg:>7.1%}"
              f"{a['pred_touch'] / s:>9.1%}{a['cooldown'] / s:>7.1%}")
    q = torch.tensor([0.5, 0.9, 0.99, 1.0])
    print(f"  payload speed  p50 {speeds.quantile(q)[0]:.3f}  p90 {speeds.quantile(q)[1]:.3f}"
          f"  p99 {speeds.quantile(q)[2]:.3f}  max {speeds.max():.3f}"
          f"   (per step: max {speeds.max() * 0.05:.3f})")
    ep = max(overshoot["episodes"], 1)
    print(f"  episodes {overshoot['episodes']}  won {overshoot['win'] / ep:.1%}"
          f"  got within 1.5 then lost {overshoot['reached_close_but_lost'] / ep:.1%}")


def physics_ceiling(config):
    """Top payload speed when every agent shoves at full thrust from behind.

    push_coef changes the reward, not the physics, so this is the hard upper
    bound on any overshoot a larger coefficient could buy.
    """
    cfg = dataclasses.replace(config, num_envs=4)
    env = Env(cfg)
    env.reset()
    best = 0.0
    for _ in range(600):
        ws, ss = env.world_state, env.scenario_state
        goal_dir = ss.goal_pos - ws.payload_pos
        goal_dir = goal_dir / goal_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        # every agent drives straight at the goal side of the crate
        actions = goal_dir.unsqueeze(1).expand(-1, cfg.n_agents, -1).clone()
        _, _, _, _, info = env.step(actions, training_progress=1.0)
        best = max(best, float(info["world_state"].payload_vel.norm(dim=-1).max()))
    print(f"\n=== physics ceiling (all agents full thrust, goalward) ===")
    print(f"  max payload speed {best:.3f} units/s -> {best * config.dt:.3f} per step")
    print(f"  success region diameter {2 * config.success_threshold:.2f}; "
          f"tunneling needs a step > {2 * config.success_threshold:.2f}")


if __name__ == "__main__":
    CHECKPOINT_DIR = "train/checkpoints/variant_d_danger_radius"
    SEEDS = (0, 1, 2)

    config = Config(num_envs=16)
    physics_ceiling(config)
    for seed in SEEDS:
        path = f"{CHECKPOINT_DIR}/seed_{seed}/checkpoint_best.pt"
        acc, speeds, over = probe(config, load_actor(path, config))
        report(acc, speeds, over, f"variant_d seed_{seed}")
    acc, speeds, over = probe(config, scripted_policy_factory())
    report(acc, speeds, over, "scripted")
