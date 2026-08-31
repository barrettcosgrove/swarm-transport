"""
tools/render.py

Records episodes and writes them out as GIF or MP4.

Deliberately two-phase:
  1. record_episode() / record_episodes() run the simulation and store
     snapshots. No drawing happens here -- matplotlib is far slower than
     physics, so drawing inline would throttle the simulation to the render rate.
  2. render_video() walks those snapshots and draws them.

Because the phases are separate, you can re-render the same recorded episode
with different visual settings without re-running any physics.

Usage:
    from tools.render import record_episode, render_video, render_to_gif

    frames = record_episode(env, policy, n_steps=250)
    render_video(frames, "outputs/rollout.mp4", config, fps=8, every=1)

    python -m tools.render
    # 100-episode rollouts of the trained policy and the scripted controller,
    # ranked, top 6 wins as MP4
"""
import math
import os
from collections import deque
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")           # headless backend -- no display needed
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from matplotlib.patheffects import withStroke
import numpy as np
import imageio.v2 as imageio
import torch


# ---------------------------------------------------------------- colors

COLOR_FLOOR = "#1a1d23"
COLOR_WALL = "#2c313a"
COLOR_OBSTACLE = "#3a404c"
COLOR_SHADOW = "#0d0f12"
COLOR_AGENT = "#3d8bfd"
COLOR_AGENT_PUSH = "#7ec8ff"
COLOR_AGENT_HURT = "#ff3b3b"
COLOR_PREDATOR = "#e03131"
COLOR_PREDATOR_COOL = "#8a4a4a"
COLOR_PAYLOAD = "#f59e0b"
COLOR_PAYLOAD_SUCCESS = "#22c55e"
COLOR_GOAL_EDGE = "#14532d"
COLOR_GOAL_FILL = "#22c55e"
COLOR_LOCKON = "#f87171"
COLOR_HUNTED = "#fde68a"
COLOR_TEXT = "#e8eaed"
COLOR_HEALTH_OK = "#22c55e"
COLOR_HEALTH_MID = "#eab308"
COLOR_HEALTH_LOW = "#ef4444"
COLOR_HEALTH_TRACK = "#12151a"
COLOR_HEALTH_BORDER = "#6b7280"

# world-unit offset for the 2.5D drop shadow
SHADOW_DX = 0.08
SHADOW_DY = -0.08
# velocity ticks: world units per (sim unit / sec). At speed 8 the tick is 0.4.
VEL_TICK_SCALE = 0.05

# backtrack only counts as "near the goal" once closest approach is inside this
# many success radii. Early wobble is already in max_backtrack; this term is
# specifically the last-second bounce-out that makes a win look accidental.
NEAR_GOAL_RADII = 2.0
# payload halfsize is 0.2; a sliver of spring contact is expected, sliding
# through a box is not. Wins above this max AABB penetration are dropped.
MAX_OBSTACLE_PEN = 0.12


# ---------------------------------------------------------------- recording

@dataclass
class Frame:
    """One timestep's worth of everything the renderer needs to draw.

    Every tensor is cloned at capture time -- the simulation keeps mutating
    its own state, and without the clone every Frame would end up pointing
    at the same final values.
    """
    agent_pos: torch.Tensor        # (E, n_agents, 2)
    agent_vel: torch.Tensor        # (E, n_agents, 2)
    predator_pos: torch.Tensor     # (E, 2)
    predator_vel: torch.Tensor     # (E, 2)
    payload_pos: torch.Tensor      # (E, 2)
    goal_pos: torch.Tensor         # (E, 2)
    obstacle_center: torch.Tensor  # (E, n_obstacles, 2)
    obstacle_active: torch.Tensor  # (E, n_obstacles) bool
    health: torch.Tensor           # (E,)
    step_count: torch.Tensor       # (E,)
    took_damage: torch.Tensor      # (E,) bool -- health dropped this step
    at_goal: torch.Tensor          # (E,) bool -- payload within success radius
    terminated: torch.Tensor       # (E,) bool -- success or capture, this step
    truncated: torch.Tensor       # (E,) bool -- hit max_steps, this step
    win_count: torch.Tensor        # (E,) running total, carried forward every frame
    loss_count: torch.Tensor       # (E,) running total, carried forward every frame
    predator_target: torch.Tensor  # (E,) long -- lock-on agent index
    predator_cooldown: torch.Tensor  # (E,) steps remaining
    behind_mask: torch.Tensor      # (E, n_agents) bool -- push-zone, same as trainer


@dataclass
class EpisodeRecord:
    """One finished episode: the frames to draw plus the numbers that rank it."""
    index: int
    frames: list
    won: bool
    captured: bool
    timed_out: bool
    n_steps: int
    start_dist: float
    end_dist: float
    end_health: float
    damage: float
    max_backtrack: float
    near_goal_backtrack: float
    push_efficiency: float
    position_ratio: float
    behind_frac: float
    path_straightness: float
    payload_progress: float
    capture_occupancy: float
    camping_time: float
    mean_predator_dist: float
    min_predator_dist: float
    team_spread: float
    evade_cosine: float
    hunted_evade_cosine: float
    hunted_thrust: float
    closing_thrust: float
    max_obstacle_pen: float
    score: float = 0.0


def _capture_frame(ws, ss, health_before, at_goal, terminated, truncated,
                   win_count, loss_count, config):
    from env import scenario
    behind_mask, _ = scenario.payload_side_masks(ws, ss, config.push_zone_radius)
    return Frame(
        agent_pos=ws.agent_pos.clone(),
        agent_vel=ws.agent_vel.clone(),
        predator_pos=ws.predator_pos.clone(),
        predator_vel=ws.predator_vel.clone(),
        payload_pos=ws.payload_pos.clone(),
        goal_pos=ss.goal_pos.clone(),
        obstacle_center=ws.obstacle_center.clone(),
        obstacle_active=ws.obstacle_active.clone(),
        health=ss.health.clone(),
        step_count=ss.step_count.clone(),
        took_damage=(ss.health < health_before),
        at_goal=at_goal,
        terminated=terminated.clone(),
        truncated=truncated.clone(),
        win_count=win_count.clone(),
        loss_count=loss_count.clone(),
        predator_target=ss.predator_target.clone(),
        predator_cooldown=ss.predator_cooldown.clone(),
        behind_mask=behind_mask.clone(),
    )


def record_episode(env, policy, n_steps, training_progress=1.0):
    """Run the simulation, capturing a Frame per step. No drawing.

    `policy` is any callable (world_state, scenario_state, config) -> actions,
    so this works identically for the scripted controller and a trained actor.

    Win/loss are classified from terminated/truncated + at_goal:
      terminated & at_goal   -> win (success)
      terminated & ~at_goal  -> loss (captured)
      truncated & ~terminated -> loss (timeout)
    """
    frames = []
    env.reset()
    E = env.config.num_envs
    win_count = torch.zeros(E)
    loss_count = torch.zeros(E)
    min_payload_dist = torch.full((E,), float("inf"))

    for _ in range(n_steps):
        ws, ss = env.world_state, env.scenario_state
        actions = policy(ws, ss, env.config)

        health_before = ss.health.clone()
        obs, reward, terminated, truncated, info = env.step(actions, training_progress)
        ws, ss = info["world_state"], info["scenario_state"]

        payload_dist = info["payload_dist"]
        min_payload_dist = torch.minimum(min_payload_dist, payload_dist)
        at_goal = info["success"]

        win_count = win_count + (terminated & at_goal).float()
        loss_count = loss_count + (terminated & ~at_goal).float() + (truncated & ~terminated).float()

        frames.append(_capture_frame(
            ws, ss, health_before, at_goal, terminated, truncated,
            win_count, loss_count, env.config))

    print(f"closest approach per env (threshold={env.config.success_threshold}):")
    for e in range(E):
        print(f"  env {e}: {min_payload_dist[e].item():.3f}")

    return frames


class _EpisodeAccum:
    """Running sums for one in-progress episode. Reset after every done."""

    def __init__(self, start_dist, start_health):
        self.start_dist = start_dist
        self.start_health = start_health
        self.frames = []
        self.dists = []
        self.payload_path = []
        self.behind = 0
        self.front = 0
        self.behind_steps = 0
        self.force_goal = 0.0
        self.force_mag = 0.0
        self.pred_sum = 0.0
        self.pred_min = float("inf")
        self.spread_sum = 0.0
        self.n_beh = 0
        self.max_obstacle_pen = 0.0
        self.evade = {k: 0.0 for k in (
            "evade_cosine_sum", "evade_count", "hunted_cosine_sum", "hunted_thrust_sum",
            "hunted_count", "closing_thrust_sum", "closing_count",
            "capture_hits", "agent_steps", "camping_hits", "env_steps")}

    def add_evasion(self, stats):
        for key, value in stats.items():
            if key in self.evade:
                self.evade[key] += value

    def observe_step(self, ws, ss, info, config, e=0):
        from env import scenario

        behind_mask, front_mask = scenario.payload_side_masks(
            ws, ss, config.push_zone_radius)
        self.behind += int(behind_mask[e].sum())
        self.front += int(front_mask[e].sum())
        if bool(behind_mask[e].any()):
            self.behind_steps += 1

        goalward, magnitude = scenario.payload_goalward_force(ws, ss, config)
        self.force_goal += float(goalward[e])
        self.force_mag += float(magnitude[e])

        pred = float((ws.agent_pos[e] - ws.predator_pos[e]).norm(dim=-1).min())
        self.pred_sum += pred
        self.pred_min = min(self.pred_min, pred)
        centroid = ws.agent_pos[e].mean(dim=0)
        self.spread_sum += float((ws.agent_pos[e] - centroid).norm(dim=-1).mean())
        self.n_beh += 1

        dist = float(info["payload_dist"][e])
        self.dists.append(dist)
        self.payload_path.append((
            float(ws.payload_pos[e, 0]), float(ws.payload_pos[e, 1])))
        self.max_obstacle_pen = max(
            self.max_obstacle_pen, _payload_obstacle_penetration(ws, e))


def _payload_obstacle_penetration(ws, e=0):
    """Max AABB penetration of the payload into any active obstacle.

    Same geometry as physics.box_box_forces: overlap on both axes, then the
    axis of least penetration. Inactive obstacles are ignored.
    """
    delta = ws.payload_pos[e] - ws.obstacle_center[e]
    overlap = (ws.payload_halfsize + ws.obstacle_halfsize) - torch.abs(delta)
    overlap = torch.clamp(overlap, min=0.0)
    penetration = overlap.min(dim=-1).values
    penetration = torch.where(ws.obstacle_active[e], penetration, torch.zeros_like(penetration))
    return float(penetration.max()) if penetration.numel() else 0.0


def _backtrack_stats(dists, near_radius):
    """Max distance the payload ever gave back after a closer approach.

    `near_goal_backtrack` is the same quantity gated to after the payload has
    already been inside `near_radius` of the goal -- a last-second bounce-out
    that max_backtrack would otherwise treat the same as an early wobble.
    """
    min_so_far = float("inf")
    max_backtrack = 0.0
    near_goal_backtrack = 0.0
    for dist in dists:
        if dist < min_so_far:
            min_so_far = dist
        backtrack = dist - min_so_far
        if backtrack > max_backtrack:
            max_backtrack = backtrack
        if min_so_far <= near_radius and backtrack > near_goal_backtrack:
            near_goal_backtrack = backtrack
    return max_backtrack, near_goal_backtrack


def _path_straightness(payload_path, start_dist):
    if len(payload_path) < 2:
        return 0.0
    diffs = np.diff(np.asarray(payload_path, dtype=np.float64), axis=0)
    path_len = float(np.linalg.norm(diffs, axis=1).sum())
    if path_len <= 1e-6:
        return 0.0
    return start_dist / path_len


def _ratio(total, n, default=0.0):
    return total / n if n else default


def _clip01(value):
    return max(0.0, min(1.0, value))


def _finalize_episode(index, accum, won, captured, timed_out, n_steps,
                      end_health, end_dist, config):
    max_backtrack, near_goal_backtrack = _backtrack_stats(
        accum.dists, NEAR_GOAL_RADII * config.success_threshold)
    evade = accum.evade
    return EpisodeRecord(
        index=index,
        frames=accum.frames,
        won=won,
        captured=captured,
        timed_out=timed_out,
        n_steps=n_steps,
        start_dist=accum.start_dist,
        end_dist=end_dist,
        end_health=end_health,
        damage=max(accum.start_health - end_health, 0.0),
        max_backtrack=max_backtrack,
        near_goal_backtrack=near_goal_backtrack,
        push_efficiency=_ratio(accum.force_goal, accum.force_mag),
        position_ratio=accum.behind / max(accum.front, 1),
        behind_frac=_ratio(accum.behind_steps, max(n_steps, 1)),
        path_straightness=_path_straightness(accum.payload_path, accum.start_dist),
        payload_progress=accum.start_dist - end_dist,
        capture_occupancy=_ratio(evade["capture_hits"], evade["agent_steps"]),
        camping_time=_ratio(evade["camping_hits"], evade["env_steps"]),
        mean_predator_dist=_ratio(accum.pred_sum, accum.n_beh, default=float("nan")),
        min_predator_dist=accum.pred_min if accum.pred_min < float("inf") else float("nan"),
        team_spread=_ratio(accum.spread_sum, accum.n_beh, default=float("nan")),
        evade_cosine=_ratio(evade["evade_cosine_sum"], evade["evade_count"], default=0.0),
        hunted_evade_cosine=_ratio(
            evade["hunted_cosine_sum"], evade["hunted_count"], default=0.0),
        hunted_thrust=_ratio(evade["hunted_thrust_sum"], evade["hunted_count"], default=0.0),
        closing_thrust=_ratio(evade["closing_thrust_sum"], evade["closing_count"], default=0.0),
        max_obstacle_pen=accum.max_obstacle_pen,
    )


def score_episode(ep, config):
    """Higher is a better demo clip. All terms in [0, 1].

    Primary weight sits on the things that read on camera: a short win, no
    reverse at the goal, and agents actually pushing from behind. Secondary
    terms dump lucky-but-ugly wins (low health, camping predator, time spent
    inside the capture ring, a scribbled payload path).
    """
    decisiveness = 1.0 - ep.n_steps / max(config.max_steps, 1)
    smoothness = 1.0 / (1.0 + ep.max_backtrack)
    near_goal = 1.0 / (1.0 + 4.0 * ep.near_goal_backtrack)
    push = _clip01(ep.push_efficiency)
    position = ep.position_ratio / (ep.position_ratio + 1.0)
    behind = _clip01(ep.behind_frac)
    health = _clip01(ep.end_health / config.max_health)
    damage_ok = 1.0 - _clip01(ep.damage / config.max_health)
    straight = _clip01(ep.path_straightness)
    progress = _clip01(ep.payload_progress / max(ep.start_dist, 1e-6))
    occupancy = 1.0 - _clip01(ep.capture_occupancy)
    camping = 1.0 - _clip01(ep.camping_time)
    evade = 0.5 * (ep.evade_cosine + 1.0)
    hunted = 0.5 * (ep.hunted_evade_cosine + 1.0)
    pred = 0.5 if math.isnan(ep.mean_predator_dist) else _clip01(ep.mean_predator_dist / 2.5)
    if math.isnan(ep.team_spread):
        spread = 0.5
    else:
        spread = math.exp(-((ep.team_spread - 1.1) ** 2) / (2 * 0.6 ** 2))

    return (
        0.16 * decisiveness
        + 0.14 * smoothness
        + 0.12 * near_goal
        + 0.12 * push
        + 0.07 * position
        + 0.05 * behind
        + 0.08 * health
        + 0.04 * damage_ok
        + 0.05 * straight
        + 0.03 * progress
        + 0.04 * occupancy
        + 0.03 * camping
        + 0.02 * evade
        + 0.02 * hunted
        + 0.02 * pred
        + 0.01 * spread
    )


def rank_episodes(episodes, config, max_obstacle_pen=MAX_OBSTACLE_PEN):
    """Score every episode, return wins sorted best-first.

    Wins whose payload slid through an obstacle (max AABB penetration above
    `max_obstacle_pen`) are dropped so a demo cannot open on a tunneling clip.
    """
    for ep in episodes:
        ep.score = score_episode(ep, config)
    wins = [ep for ep in episodes if ep.won]
    clean = [ep for ep in wins if ep.max_obstacle_pen <= max_obstacle_pen]
    dropped = len(wins) - len(clean)
    if dropped:
        print(f"dropped {dropped}/{len(wins)} wins with obstacle penetration > {max_obstacle_pen}")
    return sorted(clean, key=lambda ep: ep.score, reverse=True)


def format_ranking(ranked, n=15):
    header = (f"{'#':>3} {'idx':>4} {'score':>6} {'steps':>5} {'back':>6} {'ngbk':>6}"
              f"{'pshE':>6} {'posR':>6} {'bhnd':>5} {'hp':>6} {'capO':>5} {'strt':>5}"
              f"{'camp':>5} {'obsP':>6}")
    lines = [header]
    for i, ep in enumerate(ranked[:n], start=1):
        lines.append(
            f"{i:>3} {ep.index:>4} {ep.score:>6.3f} {ep.n_steps:>5}"
            f" {ep.max_backtrack:>6.3f} {ep.near_goal_backtrack:>6.3f}"
            f" {ep.push_efficiency:>5.1%} {ep.position_ratio:>6.2f}"
            f" {ep.behind_frac:>5.0%} {ep.end_health:>6.1f}"
            f" {ep.capture_occupancy:>4.0%} {ep.path_straightness:>5.2f}"
            f" {ep.camping_time:>4.0%} {ep.max_obstacle_pen:>6.3f}"
        )
    return "\n".join(lines)


def record_episodes(env, policy, n_episodes, training_progress=1.0):
    """Run one environment until `n_episodes` have finished. No drawing.

    Metrics are accumulated live (forces, side masks, evasion) so ranking does
    not have to reconstruct them from positions after the fact. Frames are
    kept per episode so a later select can stitch a demo without re-simulating.
    """
    from env import scenario

    env.reset()
    config = env.config
    e = 0
    episodes = []
    win_count = torch.zeros(config.num_envs)
    loss_count = torch.zeros(config.num_envs)

    def new_accum():
        start_dist = float(torch.norm(
            env.world_state.payload_pos[e] - env.scenario_state.goal_pos[e]))
        start_health = float(env.scenario_state.health[e])
        return _EpisodeAccum(start_dist, start_health)

    accum = new_accum()
    max_total_steps = n_episodes * config.max_steps + config.max_steps

    for _ in range(max_total_steps):
        if len(episodes) >= n_episodes:
            break

        ws, ss = env.world_state, env.scenario_state
        actions = policy(ws, ss, config)
        accum.add_evasion(scenario.evasion_step_stats(ws, ss, actions, config))
        health_before = ss.health.clone()
        _, _, terminated, truncated, info = env.step(actions, training_progress)
        ws, ss = info["world_state"], info["scenario_state"]
        at_goal = info["success"]

        # push / path stats on the pre-reset board, same as tools.evaluate
        accum.observe_step(ws, ss, info, config, e=e)

        won_now = bool(terminated[e] and at_goal[e])
        captured_now = bool(terminated[e] and not at_goal[e])
        timed_out_now = bool(truncated[e] and not terminated[e])
        win_count = win_count + (terminated & at_goal).float()
        loss_count = loss_count + (terminated & ~at_goal).float() + (truncated & ~terminated).float()

        accum.frames.append(_capture_frame(
            ws, ss, health_before, at_goal, terminated, truncated,
            win_count, loss_count, config))

        if won_now or captured_now or timed_out_now:
            n_steps = int(ss.step_count[e])
            episodes.append(_finalize_episode(
                index=len(episodes),
                accum=accum,
                won=won_now,
                captured=captured_now,
                timed_out=timed_out_now,
                n_steps=n_steps,
                end_health=float(ss.health[e]),
                end_dist=float(info["payload_dist"][e]),
                config=config,
            ))
            accum = new_accum()
    else:
        raise RuntimeError(
            f"finished {len(episodes)}/{n_episodes} episodes before the step cap")

    return episodes


# ---------------------------------------------------------------- drawing

def compute_arena_limit(config, margin=1.15):
    """
    View bounds derived from the walls themselves, rather than a separate
    hardcoded constant that could drift out of sync with the actual arena
    size as the walls are retuned.

    Measured to each wall's INNER face, not its outer extent, so the framing
    tracks the playable area rather than how thick the walls happen to be.
    Wall thickness is a containment constant chosen in train/config.py to keep
    a fast body from crossing a wall's midline, and it has no business
    deciding how far the camera pulls back -- at 3.0 thick, outer extents
    would spend a fifth of the frame on grey border. The walls simply clip at
    the frame edge, which reads the same as a solid boundary.
    """
    max_extent = 0.0
    for center, halfsize in zip(config.wall_center, config.wall_halfsize):
        # the thin axis is the one the wall's face is on; the long axis just
        # runs the wall far enough to seal the corners
        thin_axis = int(torch.argmin(halfsize))
        inner_face = abs(float(center[thin_axis])) - float(halfsize[thin_axis])
        max_extent = max(max_extent, inner_face)
    return max_extent * margin


def _health_color(frac):
    if frac < 0.3:
        return COLOR_HEALTH_LOW
    if frac < 0.55:
        return COLOR_HEALTH_MID
    return COLOR_HEALTH_OK


class _PanelArtists:
    """Holds the matplotlib patch objects for one panel.

    These are created ONCE and then repositioned each frame, rather than
    calling ax.clear() and rebuilding every patch. Measured on a 2x2 grid:
    ~3 fps rebuilding, ~55 fps repositioning -- a 16x difference, which is
    what makes rendering a full episode take seconds instead of a minute.
    """

    def __init__(self, ax, config, n_agents, n_obstacles, arena_margin=1.15,
                 show_diagnostics=True, show_tally=True, show_health_value=False,
                 payload_trail=False, body_trail_len=0, filled_goal=False,
                 prominent_banner=False):
        lim = compute_arena_limit(config, margin=arena_margin)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(COLOR_FLOOR)
        for spine in ax.spines.values():
            spine.set_visible(False)

        self.show_diagnostics = show_diagnostics
        self.show_tally = show_tally
        self.show_health_value = show_health_value
        self.use_payload_trail = payload_trail
        self.body_trail_len = body_trail_len
        self.n_agents = n_agents
        self.agent_radius = config.agent_radius
        self._last_step = None
        self._label_stroke = [withStroke(linewidth=2.4, foreground="#111111")]

        for wc, wh in zip(config.wall_center, config.wall_halfsize):
            ax.add_patch(Rectangle(
                (float(wc[0] - wh[0]), float(wc[1] - wh[1])),
                float(2 * wh[0]), float(2 * wh[1]),
                facecolor=COLOR_WALL, edgecolor="none", zorder=1,
            ))

        oh = config.obstacle_halfsize
        self.obstacle_shadows = []
        self.obstacles = []
        for i in range(n_obstacles):
            hw = float(oh[i][0]) if oh.dim() > 1 else float(oh[0])
            hh = float(oh[i][1]) if oh.dim() > 1 else float(oh[1])
            self.obstacle_halfsize = (hw, hh)
            shadow = Rectangle((0, 0), 2 * hw, 2 * hh,
                               facecolor=COLOR_SHADOW, edgecolor="none",
                               alpha=0.55, zorder=5.3)
            patch = Rectangle((0, 0), 2 * hw, 2 * hh,
                               facecolor=COLOR_OBSTACLE, edgecolor="none", zorder=5.5)
            ax.add_patch(shadow)
            ax.add_patch(patch)
            self.obstacle_shadows.append(shadow)
            self.obstacles.append(patch)

        goal_fill = to_rgba(COLOR_GOAL_FILL, 0.16) if filled_goal else "none"
        self.goal = Circle(
            (0, 0), config.success_threshold,
            facecolor=goal_fill, edgecolor=COLOR_GOAL_EDGE,
            linestyle="solid", linewidth=2.6, zorder=3)
        ax.add_patch(self.goal)

        self.payload_xs = []
        self.payload_ys = []
        self.payload_trail, = ax.plot(
            [], [], color=COLOR_PAYLOAD, lw=2.8, alpha=0.75, zorder=3.5,
            solid_capstyle="round")
        self.payload_trail.set_visible(payload_trail)

        self.agent_hist = [deque(maxlen=max(body_trail_len, 1)) for _ in range(n_agents)]
        self.agent_trails = []
        for _ in range(n_agents):
            line, = ax.plot([], [], color=COLOR_AGENT, lw=1.6, alpha=0.55,
                             zorder=5.7, solid_capstyle="round")
            line.set_visible(body_trail_len > 0)
            self.agent_trails.append(line)
        self.predator_hist = deque(maxlen=max(body_trail_len, 1))
        self.predator_trail, = ax.plot(
            [], [], color=COLOR_PREDATOR, lw=1.8, alpha=0.6, zorder=5.7,
            solid_capstyle="round")
        self.predator_trail.set_visible(body_trail_len > 0)

        self.lockon, = ax.plot(
            [], [], color=COLOR_LOCKON, lw=1.2, alpha=0.85, zorder=5.8,
            solid_capstyle="round")

        ph = config.payload_halfsize
        self.payload_halfsize = (float(ph[0]), float(ph[1]))
        self.payload_shadow = Rectangle(
            (0, 0), 2 * self.payload_halfsize[0], 2 * self.payload_halfsize[1],
            facecolor=COLOR_SHADOW, edgecolor="none", alpha=0.55, zorder=4.8)
        self.payload = Rectangle((0, 0), 2 * self.payload_halfsize[0],
                                   2 * self.payload_halfsize[1],
                                   facecolor=COLOR_PAYLOAD, edgecolor="none", zorder=5)
        ax.add_patch(self.payload_shadow)
        ax.add_patch(self.payload)

        self.capture = Circle(
            (0, 0), config.predator_capture_radius,
            facecolor=to_rgba(COLOR_PREDATOR, 0.14),
            edgecolor=to_rgba(COLOR_PREDATOR, 0.45),
            linestyle="solid", linewidth=1.0, zorder=4)
        ax.add_patch(self.capture)

        self.agent_shadows = []
        self.agents = []
        self.agent_labels = []
        self.agent_ticks = []
        for i in range(n_agents):
            shadow = Circle((0, 0), config.agent_radius,
                            facecolor=COLOR_SHADOW, edgecolor="none",
                            alpha=0.55, zorder=5.6)
            patch = Circle((0, 0), config.agent_radius, facecolor=COLOR_AGENT,
                            edgecolor="#dbeafe", linewidth=0.6, zorder=6)
            ax.add_patch(shadow)
            ax.add_patch(patch)
            self.agent_shadows.append(shadow)
            self.agents.append(patch)
            label = ax.text(
                0, 0, str(i + 1), ha="center", va="center",
                fontsize=6, color="white", weight="bold", zorder=7,
                path_effects=self._label_stroke)
            self.agent_labels.append(label)
            tick, = ax.plot([], [], color="white", lw=1.3, alpha=0.85,
                             zorder=6.5, solid_capstyle="round")
            self.agent_ticks.append(tick)

        self.hunted_ring = Circle(
            (0, 0), config.agent_radius * 1.55,
            facecolor="none", edgecolor=COLOR_HUNTED, linewidth=1.6, zorder=6.2)
        ax.add_patch(self.hunted_ring)

        self.predator_shadow = Circle(
            (0, 0), config.predator_radius,
            facecolor=COLOR_SHADOW, edgecolor="none", alpha=0.55, zorder=5.6)
        self.predator = Circle((0, 0), config.predator_radius,
                                facecolor=COLOR_PREDATOR, edgecolor="#fecaca",
                                linewidth=0.7, zorder=6)
        ax.add_patch(self.predator_shadow)
        ax.add_patch(self.predator)
        self.predator_tick, = ax.plot(
            [], [], color="white", lw=1.4, alpha=0.85, zorder=6.5,
            solid_capstyle="round")

        self.text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                             va="top", ha="left", fontsize=7, family="monospace",
                             color=COLOR_TEXT)
        self.text.set_visible(show_diagnostics)

        self.tally = ax.text(0.02, 0.02, "", transform=ax.transAxes,
                              va="bottom", ha="left", fontsize=8, family="monospace",
                              weight="bold", color=COLOR_TEXT)
        self.tally.set_visible(show_tally)

        self.outcome_veil = Rectangle(
            (0.0, 0.0), 1.0, 1.0, transform=ax.transAxes,
            facecolor=COLOR_FLOOR, edgecolor="none", alpha=0.0, zorder=19, visible=False)
        ax.add_patch(self.outcome_veil)

        if prominent_banner:
            banner_xy, banner_w, banner_h, banner_fs = (0.10, 0.34), 0.80, 0.32, 28
        else:
            banner_xy, banner_w, banner_h, banner_fs = (0.10, 0.40), 0.80, 0.20, 16
        self.banner_bg = FancyBboxPatch(
            banner_xy, banner_w, banner_h, transform=ax.transAxes,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor="#1f2937", edgecolor="#9ca3af",
            linewidth=2.0 if prominent_banner else 1.5,
            alpha=0.94, zorder=20, visible=False)
        ax.add_patch(self.banner_bg)
        self.banner_text = ax.text(0.5, banner_xy[1] + banner_h / 2, "",
                                     transform=ax.transAxes,
                                     va="center", ha="center", fontsize=banner_fs,
                                     weight="bold", zorder=21, visible=False)
        self.prominent_banner = prominent_banner

        bar_x, bar_y, bar_w, bar_h = 0.56, 0.93, 0.40, 0.042
        pad = 0.006
        ax.add_patch(FancyBboxPatch(
            (bar_x, bar_y), bar_w, bar_h, transform=ax.transAxes,
            boxstyle="round,pad=0.003,rounding_size=0.012",
            facecolor=COLOR_HEALTH_TRACK, edgecolor=COLOR_HEALTH_BORDER,
            linewidth=1.2, zorder=10))
        inner_x = bar_x + pad
        inner_y = bar_y + pad
        inner_w = bar_w - 2 * pad
        inner_h = bar_h - 2 * pad
        self.health_bar = Rectangle(
            (inner_x, inner_y), inner_w, inner_h, transform=ax.transAxes,
            facecolor=COLOR_HEALTH_OK, edgecolor="none", zorder=11)
        self.health_bar_x = inner_x
        self.health_bar_width = inner_w
        ax.add_patch(self.health_bar)
        self.health_label = ax.text(
            bar_x - 0.018, bar_y + bar_h / 2, "", transform=ax.transAxes,
            va="center", ha="right", fontsize=8, family="sans-serif",
            weight="bold", color=COLOR_TEXT, zorder=12)
        self.health_label.set_visible(show_health_value)

    def _reset_trails(self):
        self.payload_xs = []
        self.payload_ys = []
        self.payload_trail.set_data([], [])
        for hist, line in zip(self.agent_hist, self.agent_trails):
            hist.clear()
            line.set_data([], [])
        self.predator_hist.clear()
        self.predator_trail.set_data([], [])

    def update(self, frame, e, config):
        for i, patch in enumerate(self.obstacles):
            if bool(frame.obstacle_active[e, i]):
                cx, cy = float(frame.obstacle_center[e, i][0]), float(frame.obstacle_center[e, i][1])
                xy = (cx - self.obstacle_halfsize[0], cy - self.obstacle_halfsize[1])
                patch.set_xy(xy)
                patch.set_visible(True)
                self.obstacle_shadows[i].set_xy((xy[0] + SHADOW_DX, xy[1] + SHADOW_DY))
                self.obstacle_shadows[i].set_visible(True)
            else:
                patch.set_visible(False)
                self.obstacle_shadows[i].set_visible(False)

        gx, gy = frame.goal_pos[e]
        self.goal.center = (float(gx), float(gy))

        px, py = float(frame.payload_pos[e][0]), float(frame.payload_pos[e][1])
        self.payload.set_xy((px - self.payload_halfsize[0],
                              py - self.payload_halfsize[1]))
        self.payload_shadow.set_xy((
            px - self.payload_halfsize[0] + SHADOW_DX,
            py - self.payload_halfsize[1] + SHADOW_DY))
        self.payload.set_facecolor(
            COLOR_PAYLOAD_SUCCESS if bool(frame.at_goal[e]) else COLOR_PAYLOAD
        )

        hurt = bool(frame.took_damage[e])
        target = int(frame.predator_target[e])
        behind = frame.behind_mask[e]
        agent_xy = []
        for i, patch in enumerate(self.agents):
            ax_, ay_ = float(frame.agent_pos[e, i][0]), float(frame.agent_pos[e, i][1])
            patch.center = (ax_, ay_)
            self.agent_shadows[i].center = (ax_ + SHADOW_DX, ay_ + SHADOW_DY)
            if hurt:
                color = COLOR_AGENT_HURT
            elif bool(behind[i]):
                color = COLOR_AGENT_PUSH
            else:
                color = COLOR_AGENT
            patch.set_facecolor(color)
            self.agent_labels[i].set_position((ax_, ay_ + self.agent_radius * 2.4))
            vx = float(frame.agent_vel[e, i, 0])
            vy = float(frame.agent_vel[e, i, 1])
            self.agent_ticks[i].set_data(
                [ax_, ax_ + vx * VEL_TICK_SCALE],
                [ay_, ay_ + vy * VEL_TICK_SCALE])
            agent_xy.append((ax_, ay_))

        tx, ty = agent_xy[target]
        self.hunted_ring.center = (tx, ty)

        rx, ry = float(frame.predator_pos[e][0]), float(frame.predator_pos[e][1])
        self.predator.center = (rx, ry)
        self.predator_shadow.center = (rx + SHADOW_DX, ry + SHADOW_DY)
        self.capture.center = (rx, ry)
        self.lockon.set_data([rx, tx], [ry, ty])
        if float(frame.predator_cooldown[e]) > 0.0:
            self.predator.set_facecolor(COLOR_PREDATOR_COOL)
            self.predator.set_alpha(0.7)
        else:
            self.predator.set_facecolor(COLOR_PREDATOR)
            self.predator.set_alpha(1.0)
        pvx = float(frame.predator_vel[e, 0])
        pvy = float(frame.predator_vel[e, 1])
        self.predator_tick.set_data(
            [rx, rx + pvx * VEL_TICK_SCALE],
            [ry, ry + pvy * VEL_TICK_SCALE])

        step = int(frame.step_count[e])
        if self._last_step is not None and step < self._last_step:
            self._reset_trails()
        new_sample = self._last_step != step or not self.payload_xs
        if new_sample:
            if self.use_payload_trail:
                self.payload_xs.append(px)
                self.payload_ys.append(py)
                self.payload_trail.set_data(self.payload_xs, self.payload_ys)
            if self.body_trail_len > 0:
                for hist, line, (ax_, ay_) in zip(
                        self.agent_hist, self.agent_trails, agent_xy):
                    hist.append((ax_, ay_))
                    xs = [p[0] for p in hist]
                    ys = [p[1] for p in hist]
                    line.set_data(xs, ys)
                self.predator_hist.append((rx, ry))
                self.predator_trail.set_data(
                    [p[0] for p in self.predator_hist],
                    [p[1] for p in self.predator_hist])
        self._last_step = step

        health = float(frame.health[e])
        max_health = config.max_health
        frac = max(health, 0.0) / max_health
        hp_color = _health_color(frac)

        if self.show_diagnostics:
            dist = float(torch.norm(frame.payload_pos[e] - frame.goal_pos[e]))
            self.text.set_text(
                f"env {e}\n"
                f"health {health:5.1f}/{max_health:.0f}\n"
                f"step   {step:3d}/{config.max_steps}\n"
                f"dist   {dist:5.2f}"
            )

        self.health_bar.set_width(self.health_bar_width * frac)
        self.health_bar.set_facecolor(hp_color)
        self.health_bar.set_visible(frac > 0.0)
        if self.show_health_value:
            self.health_label.set_text(f"HP  {health:.0f}")
            self.health_label.set_color(hp_color)

        if self.show_tally:
            wins = int(frame.win_count[e])
            losses = int(frame.loss_count[e])
            self.tally.set_text(f"W {wins}  L {losses}")

        # outcome banner: only visible on the exact frame(s) an episode ended.
        # Since render_video redraws this SAME frame repeatedly during a
        # hold, the banner naturally persists for the hold's duration with
        # no separate timer needed here.
        won = bool(frame.terminated[e] and frame.at_goal[e])
        captured = bool(frame.terminated[e] and not frame.at_goal[e])
        timed_out = bool(frame.truncated[e] and not frame.terminated[e])

        if won or captured or timed_out:
            self.outcome_veil.set_visible(True)
            self.banner_bg.set_visible(True)
            self.banner_text.set_visible(True)
            if won:
                veil = to_rgba(COLOR_PAYLOAD_SUCCESS, 0.22 if self.prominent_banner else 0.10)
                self.outcome_veil.set_facecolor(veil)
                self.banner_bg.set_facecolor("#14532d")
                self.banner_bg.set_edgecolor(COLOR_GOAL_EDGE)
                self.banner_text.set_text("WIN")
                self.banner_text.set_color("#86efac")
            elif captured:
                veil = to_rgba(COLOR_PREDATOR, 0.22 if self.prominent_banner else 0.10)
                self.outcome_veil.set_facecolor(veil)
                self.banner_bg.set_facecolor("#7f1d1d")
                self.banner_bg.set_edgecolor(COLOR_PREDATOR)
                self.banner_text.set_text("CAPTURED")
                self.banner_text.set_color("#fca5a5")
            else:
                veil = to_rgba("#6b7280", 0.22 if self.prominent_banner else 0.10)
                self.outcome_veil.set_facecolor(veil)
                self.banner_bg.set_facecolor("#374151")
                self.banner_bg.set_edgecolor("#9ca3af")
                self.banner_text.set_text("TIMEOUT")
                self.banner_text.set_color("#d1d5db")
        else:
            self.outcome_veil.set_visible(False)
            self.banner_bg.set_visible(False)
            self.banner_text.set_visible(False)


def _build_panel_schedule(frames, panel_idx, every, hold_frames):
    """
    Build the sequence of recorded-frame indices to draw for ONE panel,
    inserting `hold_frames` repeats every time that panel's episode ends.

    This runs independently per panel -- panel 0 finishing on step 3 while
    panel 1 is still mid-episode on step 50 means panel 0 holds while panel
    1 keeps advancing normally. This is deliberate: freezing the whole grid
    whenever any single panel finishes would defeat the point of watching
    four independent episodes side by side.
    """
    schedule = []
    i = 0
    n = len(frames)
    while i < n:
        schedule.append(i)
        ended = bool(frames[i].terminated[panel_idx] or frames[i].truncated[panel_idx])
        if ended:
            schedule.extend([i] * hold_frames)
            i += every
            continue
        # `every` would skip a terminal sitting on an odd index; freeze on it
        # anyway so a stitched demo cannot drop a WIN / CAPTURED / TIMEOUT hold.
        terminal = None
        for j in range(i + 1, min(i + every, n)):
            if bool(frames[j].terminated[panel_idx] or frames[j].truncated[panel_idx]):
                terminal = j
                break
        if terminal is not None:
            schedule.append(terminal)
            schedule.extend([terminal] * hold_frames)
            i = terminal + 1
        else:
            i += every
    return schedule


def _write_video(output_path, images, fps):
    """Encode RGB frames. MP4 uses H.264/yuv420p for GitHub/browser playback."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower()
    if ext in (".mp4", ".m4v", ".mov"):
        h, w = images[0].shape[:2]
        if w % 2 or h % 2:
            images = [img[:h - (h % 2), :w - (w % 2)] for img in images]
        imageio.mimsave(
            output_path, images, fps=fps, codec="libx264",
            pixelformat="yuv420p", macro_block_size=1, quality=8)
        # GitHub and most markdown previews strip <video> unless the src is
        # a drag-and-drop user-attachments URL. A GIF sibling is what the
        # README can actually display.
        gif_path = os.path.splitext(output_path)[0] + ".gif"
        imageio.mimsave(gif_path, images, fps=min(int(fps), 12), loop=0)
    else:
        # loop=0 writes the Netscape 2.0 extension (infinite loop). Without it
        # GitHub and most markdown previews show the GIF as a still first frame.
        imageio.mimsave(output_path, images, fps=fps, loop=0)
    return output_path


def render_video(frames, output_path, config, fps=20, every=1, n_panels=4,
                 hold_seconds=1.5, arena_margin=1.15, show_diagnostics=True,
                 show_tally=True, show_health_value=False, payload_trail=False,
                 body_trail_len=0, filled_goal=False, prominent_banner=False):
    """Draw recorded frames and write GIF or MP4 (codec from the file suffix).

    `every` skips frames at DRAW time, not record time -- so you can render
    a quick low-frame-count version and a detailed one from the same
    recording without re-running the simulation.

    `hold_seconds` controls how long each panel pauses on its outcome banner
    once an episode ends -- converted to a frame count using `fps`, since a
    video file has no independent notion of "pause," only "show this frame longer."
    """
    if not frames:
        raise ValueError("no frames to render")

    E = frames[0].agent_pos.shape[0]
    n_panels = min(n_panels, E)
    n_agents = frames[0].agent_pos.shape[1]
    n_obstacles = frames[0].obstacle_center.shape[1]
    hold_frames = int(round(hold_seconds * fps))

    if n_panels == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4), dpi=100)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        axes = [ax]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=80)
        axes = axes.flatten()
        for ax in axes[n_panels:]:
            ax.set_visible(False)
        fig.tight_layout()
    fig.patch.set_facecolor(COLOR_FLOOR)

    artists = [_PanelArtists(
        axes[e], config, n_agents, n_obstacles,
        arena_margin=arena_margin,
        show_diagnostics=show_diagnostics,
        show_tally=show_tally,
        show_health_value=show_health_value,
        payload_trail=payload_trail,
        body_trail_len=body_trail_len,
        filled_goal=filled_goal,
        prominent_banner=prominent_banner,
    ) for e in range(n_panels)]

    # one schedule per panel, since panels can hold at different points and
    # for different total durations depending on when each one's episodes end
    schedules = [_build_panel_schedule(frames, e, every, hold_frames) for e in range(n_panels)]
    max_len = max(len(s) for s in schedules)
    for s in schedules:
        if len(s) < max_len:
            s.extend([s[-1]] * (max_len - len(s)))   # pad shorter panels by holding their last frame

    images = []
    for out_i in range(max_len):
        for e in range(n_panels):
            frame_idx = schedules[e][out_i]
            artists[e].update(frames[frame_idx], e, config)
        fig.canvas.draw()
        # buffer_rgba() is the current API -- tostring_rgb() was removed in
        # recent matplotlib. The .copy() matters: the buffer is reused each
        # draw, so without it every appended frame aliases the same memory.
        buf = np.asarray(fig.canvas.buffer_rgba())
        images.append(buf[:, :, :3].copy())

    plt.close(fig)
    return _write_video(output_path, images, fps)


def render_to_gif(*args, **kwargs):
    """Backward-compatible alias used by tools.render_seeds."""
    return render_video(*args, **kwargs)


DEMO_RENDER_KWARGS = dict(
    fps=16, every=2, n_panels=1, hold_seconds=1.5,
    arena_margin=1.02, show_diagnostics=False, show_tally=False,
    show_health_value=True, payload_trail=True, body_trail_len=12,
    filled_goal=True, prominent_banner=True,
)


def render_policy_demo(env, policy, output_path, label, n_episodes=100, n_pick=6):
    """Collect, rank, and write a single-panel demo video for any policy."""
    print(f"recording {n_episodes} episodes ({label}) ...", flush=True)
    episodes = record_episodes(env, policy, n_episodes=n_episodes)
    n_wins = sum(1 for ep in episodes if ep.won)
    n_cap = sum(1 for ep in episodes if ep.captured)
    n_to = sum(1 for ep in episodes if ep.timed_out)
    print(f"collected {len(episodes)} episodes  ({n_wins} wins, {n_cap} captures, {n_to} timeouts)")

    ranked = rank_episodes(episodes, env.config)
    print(format_ranking(ranked))
    selected = ranked[:n_pick]
    if not selected:
        raise RuntimeError(f"no winning episodes to render for {label}")

    print(f"selected {len(selected)} of {len(ranked)} clean wins:")
    for i, ep in enumerate(selected, start=1):
        print(f"  {i}. episode {ep.index}  score={ep.score:.3f}  steps={ep.n_steps}"
              f"  back={ep.max_backtrack:.3f}  push={ep.push_efficiency:.1%}"
              f"  hp={ep.end_health:.0f}  obsP={ep.max_obstacle_pen:.3f}")

    frames = [frame for ep in selected for frame in ep.frames]
    path = render_video(frames, output_path, env.config, **DEMO_RENDER_KWARGS)
    print(f"wrote {path}")
    return path


# ---------------------------------------------------------------- entry point

if __name__ == "__main__":
    from env.env import Env
    from env import scenario
    from train.config import Config
    from train.checkpoints import load_checkpoint
    from train.mappo import Actor, Critic
    from tools.scripted_policy import scripted_policy

    CHECKPOINT = "train/checkpoints/variant_c_400/seed_4/checkpoint_best.pt"
    N_EPISODES = 100
    N_PICK = 6

    config = Config(num_envs=1, seed=0)
    actor = Actor(config.obs_dim, 2, config.hidden_dim)
    critic = Critic(config.obs_dim, config.n_agents, config.hidden_dim)
    load_checkpoint(CHECKPOINT, actor, critic)
    actor.eval()

    def actor_policy(world_state, scenario_state, cfg, actor=actor):
        with torch.no_grad():
            return actor(scenario.observe(world_state, scenario_state, cfg)).clamp(-1.0, 1.0)

    def scripted_wrapper(world_state, scenario_state, cfg):
        return scripted_policy(scenario.observe(world_state, scenario_state, cfg), cfg)

    render_policy_demo(
        Env(config), actor_policy, "outputs/actor_variant_c_400_seed_4_demo.mp4",
        label=CHECKPOINT, n_episodes=N_EPISODES, n_pick=N_PICK)
    render_policy_demo(
        Env(config), scripted_wrapper, "outputs/scripted_demo.mp4",
        label="scripted", n_episodes=N_EPISODES, n_pick=N_PICK)
