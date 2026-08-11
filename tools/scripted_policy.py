import torch
import torch.nn.functional as F
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

    # ---- break off and dodge when the predator closes in ----
    steer = steer + _evasion(agent_pos, scenario_state, world_state, config)

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


def _evasion(agent_pos, scenario_state, world_state, config):
    """
    Steering away from the predator, ramping up as it closes.

    Only the agent the predator is committed to breaks off -- the rest keep
    pushing. This is the part that matters: the predator guards the payload,
    so it is nearly always close to whoever is in position to push, and an
    every-agent rule empties the payload of pushers exactly when the team is
    best placed to move it. Damage lands on whoever is in range regardless, so
    one evader buys the same survival for a third of the cost.

    Reads the committed target straight off scenario_state rather than
    re-deriving it. Recomputing the argmin here would disagree with the
    predator the moment its commitment timer holds a target that is no longer
    the nearest -- which is most of the time, and precisely when it matters.

    Agents out-run the predator (20 against a 6.0 speed cap), so fleeing works;
    the aim is only to survive contact, not win a chase.
    """
    predator_pos = world_state.predator_pos.unsqueeze(1)                  # (E, 1, 2)

    to_agent = agent_pos - predator_pos                                   # (E, N, 2)
    dist = torch.norm(to_agent, dim=-1, keepdim=True)
    away = to_agent / torch.clamp(dist, min=1e-6)
    urgency = torch.clamp(1.0 - dist / config.scripted_evade_radius, min=0.0)

    hunted = F.one_hot(scenario_state.predator_target, num_classes=agent_pos.shape[1])
    hunted = hunted.unsqueeze(-1).to(agent_pos.dtype)                     # (E, N, 1)

    # sidestep instead of retreating straight back. A purely radial flee walks
    # the agent directly away from the payload and the predator just follows,
    # so the agent gives up ground every time without shaking anything off.
    tangent = torch.stack([-away[..., 1], away[..., 0]], dim=-1)
    evade = away + config.scripted_evade_tangent * tangent
    return evade * urgency * hunted * config.scripted_evade_gain