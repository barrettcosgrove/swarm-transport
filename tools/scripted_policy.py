import torch
from env.physics import circle_box_static_forces

def scripted_policy(world_state, scenario_state, config):
    """
    Returns: (E, n_agents, 2) actions in [-1, 1]

    Not learned, not clever -- exists only to verify the task is solvable
    with the current physics and reward before training anything.
    """
    agent_pos = world_state.agent_pos                      # (E, N, 2)
    payload_pos = world_state.payload_pos.unsqueeze(1)     # (E, 1, 2)
    goal_pos = scenario_state.goal_pos.unsqueeze(1)        # (E, 1, 2)

    # ---- direction the payload needs to travel ----
    to_goal = goal_pos - payload_pos                       # (E, 1, 2)
    goal_dir = to_goal / torch.clamp(torch.norm(to_goal, dim=-1, keepdim=True), min=1e-6)

    # ---- the staging point behind the payload ----
    standoff = config.payload_halfsize.max() + config.agent_radius + 0.15
    push_point = payload_pos - goal_dir * standoff          # (E, 1, 2)

    # ---- which mode is each agent in? ----
    to_push_point = push_point - agent_pos                  # (E, N, 2)
    dist_to_push_point = torch.norm(to_push_point, dim=-1, keepdim=True)   # (E, N, 1)
    in_push_mode = dist_to_push_point < config.scripted_push_threshold

    approach_dir = to_push_point / torch.clamp(dist_to_push_point, min=1e-6)
    steer = torch.where(in_push_mode, goal_dir.expand_as(approach_dir), approach_dir)

    # ---- obstacle and wall repulsion ----
    repulsion = _avoidance(agent_pos, world_state, config)  # (E, N, 2)
    steer = steer + repulsion

    # ---- normalize to a valid action ----
    steer_norm = torch.clamp(torch.norm(steer, dim=-1, keepdim=True), min=1e-6)
    return steer / steer_norm

def _avoidance(agent_pos, world_state, config):
    """
    Steering repulsion away from walls and obstacles.

    Reuses circle_box_static_forces -- the collision force is already exactly
    "how hard is this thing pushing me away, and from what direction," which
    is precisely the steering signal we want. Same trick as the collision
    penalty in compute_reward: don't re-derive proximity detection, read it
    off the physics you already trust.
    """
    avoid_radius = config.agent_radius + config.scripted_avoid_margin

    wall_push = circle_box_static_forces(
        agent_pos, avoid_radius,
        world_state.wall_center, world_state.wall_halfsize, 1.0
    )
    obstacle_push = circle_box_static_forces(
        agent_pos, avoid_radius,
        world_state.obstacle_center, world_state.obstacle_halfsize, 1.0
    )
    push = wall_push + obstacle_push

    # perpendicular nudge -- prevents deadlock when repulsion points exactly
    # opposite the steering direction and the two cancel to zero
    perp = torch.stack([-push[..., 1], push[..., 0]], dim=-1)
    return (push + 0.3 * perp) * config.scripted_avoid_gain