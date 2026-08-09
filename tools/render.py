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
    from tools.scripted_policy import scripted_policy

    frames = record_episode(env, scripted_policy, n_steps=250)
    render_to_gif(frames, "outputs/rollout.gif", config, fps=20, every=2)
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


def record_episode(env, policy, n_steps, training_progress=1.0):
    """Run the simulation, capturing a Frame per step. No drawing.

    `policy` is any callable (world_state, scenario_state, config) -> actions,
    so this works identically for the scripted controller and a trained actor.
    """
    frames = []
    env.reset()

    for _ in range(n_steps):
        ws, ss = env.world_state, env.scenario_state
        actions = policy(ws, ss, env.config)

        health_before = ss.health.clone()
        env.step(actions, training_progress)
        ws, ss = env.world_state, env.scenario_state

        payload_dist = torch.norm(ws.payload_pos - ss.goal_pos, dim=-1)

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
            at_goal=(payload_dist < env.config.success_threshold),
        ))

    return frames


# ---------------------------------------------------------------- drawing

def compute_arena_limit(config, margin=1.15):
    max_extent = 0.0
    for center, halfsize in zip(config.wall_center, config.wall_halfsize):
        max_extent = max(max_extent, abs(center[0]) + halfsize[0], abs(center[1]) + halfsize[1])
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

        self.text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                             va="top", ha="left", fontsize=7, family="monospace")

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


def render_to_gif(frames, output_path, config, fps=20, every=1, n_panels=4):
    """Draw recorded frames as a 2x2 grid and write an animated GIF.

    `every` skips frames at DRAW time, not record time -- so you can render
    a quick low-frame-count version and a detailed one from the same
    recording without re-running the simulation.
    """
    if not frames:
        raise ValueError("no frames to render")

    E = frames[0].agent_pos.shape[0]
    n_panels = min(n_panels, E)
    n_agents = frames[0].agent_pos.shape[1]
    n_obstacles = frames[0].obstacle_center.shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=80)
    axes = axes.flatten()
    for ax in axes[n_panels:]:
        ax.set_visible(False)

    artists = [_PanelArtists(axes[e], config, n_agents, n_obstacles)
               for e in range(n_panels)]
    fig.tight_layout()

    images = []
    for i in range(0, len(frames), every):
        for e in range(n_panels):
            artists[e].update(frames[i], e, config)
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
    from train.config import Config
    from tools.scripted_policy import scripted_policy

    config = Config(num_envs=4)
    env = Env(config)
    frames = record_episode(env, scripted_policy, n_steps=config.max_steps)
    path = render_to_gif(frames, "outputs/scripted_rollout.gif", config, fps=20, every=2)
    print(f"wrote {path}")