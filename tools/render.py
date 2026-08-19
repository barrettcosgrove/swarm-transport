"""
tools/render.py

Records episodes and writes them out as an animated GIF showing a 2x2 grid
of environments running simultaneously.

Deliberately two-phase:
  1. record_episode() runs the simulation and stores a snapshot per step.
     No drawing happens here -- matplotlib is far slower than physics, so
     drawing inline would throttle the simulation to the render rate.
  2. render_to_gif() walks those snapshots and draws them.

Because the phases are separate, you can re-render the same recorded episode
with different visual settings without re-running any physics.

Usage:
    from tools.render import record_episode, render_to_gif

    frames = record_episode(env, policy, n_steps=250)
    render_to_gif(frames, "outputs/rollout.gif", config, fps=20, every=2)

    python -m tools.render
"""
import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")           # headless backend -- no display needed
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import imageio.v2 as imageio
import torch


# ---------------------------------------------------------------- colors

COLOR_AGENT = "#2a78d6"          # blue
COLOR_AGENT_HURT = "#ff2020"     # bright red flash on damage
COLOR_PREDATOR = "#d62728"       # red
COLOR_PAYLOAD = "#ff8c1a"        # orange
COLOR_PAYLOAD_SUCCESS = "#2ca02c"  # green flash on reaching goal
COLOR_GOAL = "#2ca02c"           # green, drawn hollow so the payload stays visible
COLOR_WALL = "#888888"           # grey
COLOR_OBSTACLE = "#666666"
COLOR_HEALTH_OK = "#4caf50"
COLOR_HEALTH_LOW = "#d62728"


# ---------------------------------------------------------------- recording

@dataclass
class Frame:
    """One timestep's worth of everything the renderer needs to draw.

    Every tensor is cloned at capture time -- the simulation keeps mutating
    its own state, and without the clone every Frame would end up pointing
    at the same final values.
    """
    agent_pos: torch.Tensor        # (E, n_agents, 2)
    predator_pos: torch.Tensor     # (E, 2)
    payload_pos: torch.Tensor      # (E, 2)
    goal_pos: torch.Tensor         # (E, 2)
    obstacle_center: torch.Tensor  # (E, n_obstacles, 2)
    obstacle_active: torch.Tensor  # (E, n_obstacles) bool
    health: torch.Tensor           # (E,)
    step_count: torch.Tensor       # (E,)
    took_damage: torch.Tensor      # (E,) bool -- health dropped this step
    at_goal: torch.Tensor          # (E,) bool -- payload within success radius
    terminated: torch.Tensor       # (E,) bool -- success or capture, this step
    truncated: torch.Tensor        # (E,) bool -- hit max_steps, this step
    win_count: torch.Tensor        # (E,) running total, carried forward every frame
    loss_count: torch.Tensor       # (E,) running total, carried forward every frame


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

        frames.append(Frame(
            agent_pos=ws.agent_pos.clone(),
            predator_pos=ws.predator_pos.clone(),
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
        ))
    
    print(f"closest approach per env (threshold={env.config.success_threshold}):")
    for e in range(E):
        print(f"  env {e}: {min_payload_dist[e].item():.3f}")

    return frames


# ---------------------------------------------------------------- drawing

def compute_arena_limit(config, margin=1.15):
    """
    View bounds derived from the walls themselves, rather than a separate
    hardcoded constant that could drift out of sync with the actual arena
    size as it gets tuned during Phase 2 calibration.

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


class _PanelArtists:
    """Holds the matplotlib patch objects for one panel.

    These are created ONCE and then repositioned each frame, rather than
    calling ax.clear() and rebuilding every patch. Measured on a 2x2 grid:
    ~3 fps rebuilding, ~55 fps repositioning -- a 16x difference, which is
    what makes rendering a full episode take seconds instead of a minute.
    """

    def __init__(self, ax, config, n_agents, n_obstacles):
        lim = compute_arena_limit(config)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        # walls never move -- draw once, never touch again
        for wc, wh in zip(config.wall_center, config.wall_halfsize):
            ax.add_patch(Rectangle(
                (float(wc[0] - wh[0]), float(wc[1] - wh[1])),
                float(2 * wh[0]), float(2 * wh[1]),
                facecolor=COLOR_WALL, edgecolor="none", zorder=1,
            ))

        oh = config.obstacle_halfsize
        self.obstacles = []
        for i in range(n_obstacles):
            hw = float(oh[i][0]) if oh.dim() > 1 else float(oh[0])
            hh = float(oh[i][1]) if oh.dim() > 1 else float(oh[1])
            patch = Rectangle((0, 0), 2 * hw, 2 * hh,
                               facecolor=COLOR_OBSTACLE, edgecolor="none", zorder=2)
            self.obstacle_halfsize = (hw, hh)
            ax.add_patch(patch)
            self.obstacles.append(patch)

        # hollow + dashed so the payload stays visible once it moves inside
        self.goal = Circle((0, 0), config.success_threshold, facecolor="none",
                            edgecolor=COLOR_GOAL, linestyle="--", linewidth=2, zorder=3)
        ax.add_patch(self.goal)

        ph = config.payload_halfsize
        self.payload_halfsize = (float(ph[0]), float(ph[1]))
        self.payload = Rectangle((0, 0), 2 * self.payload_halfsize[0],
                                   2 * self.payload_halfsize[1],
                                   facecolor=COLOR_PAYLOAD, edgecolor="none", zorder=4)
        ax.add_patch(self.payload)

        self.agents = []
        for _ in range(n_agents):
            patch = Circle((0, 0), config.agent_radius, facecolor=COLOR_AGENT,
                            edgecolor="none", zorder=5)
            ax.add_patch(patch)
            self.agents.append(patch)

        self.predator = Circle((0, 0), config.predator_radius,
                                facecolor=COLOR_PREDATOR, edgecolor="none", zorder=5)
        ax.add_patch(self.predator)

        # capture radius is larger than the body and is what actually drains
        # health, so without drawing it the damage looks like it fires at range
        self.capture = Circle((0, 0), config.predator_capture_radius, facecolor="none",
                               edgecolor=COLOR_PREDATOR, linestyle=":", linewidth=1,
                               alpha=0.6, zorder=4)
        ax.add_patch(self.capture)

        self.text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                             va="top", ha="left", fontsize=7, family="monospace")

        # persistent tally, always visible, separate from the transient outcome banner
        self.tally = ax.text(0.02, 0.02, "", transform=ax.transAxes,
                              va="bottom", ha="left", fontsize=8, family="monospace",
                              weight="bold")

        # outcome banner -- hidden by default, shown only on the frame(s) an
        # episode ends. Covers most of the panel so it's impossible to miss
        # even while skimming a fast-moving grid.
        self.banner_bg = Rectangle((0.1, 0.4), 0.8, 0.2, transform=ax.transAxes,
                                     facecolor="white", edgecolor="black", linewidth=1.5,
                                     alpha=0.9, zorder=20, visible=False)
        ax.add_patch(self.banner_bg)
        self.banner_text = ax.text(0.5, 0.5, "", transform=ax.transAxes,
                                     va="center", ha="center", fontsize=16, weight="bold",
                                     zorder=21, visible=False)

        # health bar: a grey background track with a colored fill on top.
        # transform=ax.transAxes puts these in panel-relative coordinates
        # (0-1) rather than world coordinates, so they stay pinned to the
        # corner regardless of what the simulation is doing.
        bar_x, bar_y, bar_w, bar_h = 0.62, 0.94, 0.35, 0.03
        ax.add_patch(Rectangle((bar_x, bar_y), bar_w, bar_h, transform=ax.transAxes,
                                facecolor="#dddddd", edgecolor="none", zorder=10))
        self.health_bar = Rectangle((bar_x, bar_y), bar_w, bar_h, transform=ax.transAxes,
                                      facecolor=COLOR_HEALTH_OK, edgecolor="none", zorder=11)
        self.health_bar_width = bar_w
        ax.add_patch(self.health_bar)

    def update(self, frame, e, config):
        for i, patch in enumerate(self.obstacles):
            if bool(frame.obstacle_active[e, i]):
                cx, cy = frame.obstacle_center[e, i]
                patch.set_xy((float(cx) - self.obstacle_halfsize[0],
                               float(cy) - self.obstacle_halfsize[1]))
                patch.set_visible(True)
            else:
                patch.set_visible(False)

        gx, gy = frame.goal_pos[e]
        self.goal.center = (float(gx), float(gy))

        px, py = frame.payload_pos[e]
        self.payload.set_xy((float(px) - self.payload_halfsize[0],
                              float(py) - self.payload_halfsize[1]))
        self.payload.set_facecolor(
            COLOR_PAYLOAD_SUCCESS if bool(frame.at_goal[e]) else COLOR_PAYLOAD
        )

        hurt = bool(frame.took_damage[e])
        for i, patch in enumerate(self.agents):
            ax_, ay_ = frame.agent_pos[e, i]
            patch.center = (float(ax_), float(ay_))
            patch.set_facecolor(COLOR_AGENT_HURT if hurt else COLOR_AGENT)

        rx, ry = frame.predator_pos[e]
        self.predator.center = (float(rx), float(ry))
        self.capture.center = (float(rx), float(ry))

        health = float(frame.health[e])
        max_health = config.max_health
        step = int(frame.step_count[e])
        dist = float(torch.norm(frame.payload_pos[e] - frame.goal_pos[e]))

        self.text.set_text(
            f"env {e}\n"
            f"health {health:5.1f}/{max_health:.0f}\n"
            f"step   {step:3d}/{config.max_steps}\n"
            f"dist   {dist:5.2f}"
        )

        frac = max(health, 0.0) / max_health
        self.health_bar.set_width(self.health_bar_width * frac)
        self.health_bar.set_facecolor(
            COLOR_HEALTH_LOW if frac < 0.3 else COLOR_HEALTH_OK
        )

        wins = int(frame.win_count[e])
        losses = int(frame.loss_count[e])
        self.tally.set_text(f"W {wins}  L {losses}")

        # outcome banner: only visible on the exact frame(s) an episode ended.
        # Since render_to_gif redraws this SAME frame repeatedly during a
        # hold, the banner naturally persists for the hold's duration with
        # no separate timer needed here.
        won = bool(frame.terminated[e] and frame.at_goal[e])
        captured = bool(frame.terminated[e] and not frame.at_goal[e])
        timed_out = bool(frame.truncated[e] and not frame.terminated[e])

        if won or captured or timed_out:
            self.banner_bg.set_visible(True)
            self.banner_text.set_visible(True)
            if won:
                self.banner_bg.set_facecolor("#d4f7d4")
                self.banner_text.set_text("WIN")
                self.banner_text.set_color(COLOR_PAYLOAD_SUCCESS)
            elif captured:
                self.banner_bg.set_facecolor("#fadada")
                self.banner_text.set_text("CAPTURED")
                self.banner_text.set_color(COLOR_PREDATOR)
            else:
                self.banner_bg.set_facecolor("#f0f0f0")
                self.banner_text.set_text("TIMEOUT")
                self.banner_text.set_color("#555555")
        else:
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
    for i in range(0, len(frames), every):
        schedule.append(i)
        ended = bool(frames[i].terminated[panel_idx] or frames[i].truncated[panel_idx])
        if ended:
            schedule.extend([i] * hold_frames)
    return schedule


def render_to_gif(frames, output_path, config, fps=20, every=1, n_panels=4, hold_seconds=1.5):
    """Draw recorded frames as a 2x2 grid and write an animated GIF.

    `every` skips frames at DRAW time, not record time -- so you can render
    a quick low-frame-count version and a detailed one from the same
    recording without re-running the simulation.

    `hold_seconds` controls how long each panel pauses on its outcome banner
    once an episode ends -- converted to a frame count using `fps`, since a
    GIF has no independent notion of "pause," only "show this frame longer."
    """
    if not frames:
        raise ValueError("no frames to render")

    E = frames[0].agent_pos.shape[0]
    n_panels = min(n_panels, E)
    n_agents = frames[0].agent_pos.shape[1]
    n_obstacles = frames[0].obstacle_center.shape[1]
    hold_frames = int(round(hold_seconds * fps))

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=80)
    axes = axes.flatten()
    for ax in axes[n_panels:]:
        ax.set_visible(False)

    artists = [_PanelArtists(axes[e], config, n_agents, n_obstacles)
               for e in range(n_panels)]
    fig.tight_layout()

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

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    imageio.mimsave(output_path, images, fps=fps)
    return output_path


# ---------------------------------------------------------------- entry point

if __name__ == "__main__":
    from env.env import Env
    from env import scenario
    from train.config import Config
    from train.checkpoints import load_checkpoint
    from train.mappo import Actor, Critic

    CHECKPOINT = "train/checkpoints/preview_100_push_reward/checkpoint_100.pt"

    config = Config(num_envs=4)
    actor = Actor(config.obs_dim, 2, config.hidden_dim)
    critic = Critic(config.obs_dim, config.n_agents, config.hidden_dim)
    load_checkpoint(CHECKPOINT, actor, critic)
    actor.eval()

    def actor_policy(world_state, scenario_state, cfg):
        with torch.no_grad():
            return actor(scenario.observe(world_state, scenario_state, cfg)).clamp(-1.0, 1.0)

    env = Env(config)
    frames = record_episode(env, actor_policy, n_steps=config.max_steps)
    path = render_to_gif(frames, "outputs/actor_rollout.gif", config, fps=8, every=2, hold_seconds=1.5)
    print(f"wrote {path}")