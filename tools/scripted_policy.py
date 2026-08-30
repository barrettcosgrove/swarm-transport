import torch
from env.physics import circle_box_static_forces


def scripted_policy(obs, config):
    """
    Returns: (E, n_agents, 2) actions in [-1, 1]

    Same input as Actor.forward -- the per-agent observation from
    scenario.observe -- so a comparison against a trained policy is not
    the scripted controller reading privileged state the actor never gets.

    Not learned, not clever -- exists only to verify the task is solvable
    with the current physics, reward, and observation before training anything.
    """
    n_others = config.n_agents - 1
    scale = config.obs_pos_scale

    own_pos = obs[..., 0:2]
    goal_offset = obs[..., 4:6]
    payload_offset = obs[..., 6:8]
    rel_pos_other = obs[..., 10:10 + 2 * n_others].reshape(*obs.shape[:-1], n_others, 2)
    predator_start = 10 + 4 * n_others + 1
    predator_offset = obs[..., predator_start:predator_start + 2]

    agent_pos = own_pos * scale
    payload_off = payload_offset * scale
    goal_off = goal_offset * scale
    pred_off = predator_offset * scale
    rel_pos = rel_pos_other * scale

    # ---- direction the payload needs to travel ----
    to_goal = goal_off - payload_off                       # (E, N, 2)
    goal_dir = to_goal / torch.clamp(torch.norm(to_goal, dim=-1, keepdim=True), min=1e-6)

    # ---- the staging point behind the payload, relative to the agent ----
    to_push_point = payload_off - goal_dir * config.push_standoff
    dist_to_push_point = torch.norm(to_push_point, dim=-1, keepdim=True)
    in_push_mode = dist_to_push_point < config.scripted_push_threshold

    approach_dir = to_push_point / torch.clamp(dist_to_push_point, min=1e-6)
    steer = torch.where(in_push_mode, goal_dir, approach_dir)

    # ---- wall repulsion (own_pos is in the observation specifically for this) ----
    repulsion = _avoidance(agent_pos, config)
    steer = steer + repulsion

    # ---- break off and dodge when this agent is the closest to the predator ----
    steer = steer + _evasion(pred_off, rel_pos, config)

    # ---- normalize to a valid action ----
    steer_norm = torch.clamp(torch.norm(steer, dim=-1, keepdim=True), min=1e-6)
    return steer / steer_norm


def _avoidance(agent_pos, config):
    """
    Steering repulsion away from walls.

    Obstacles are not in the observation -- they spawn per episode and
    observe() never includes them -- so they cannot be avoided here.
    Walls are fixed and own_pos is absolute, which is enough.

    Reuses circle_box_static_forces -- the collision force is already exactly
    "how hard is this thing pushing me away, and from what direction," which
    is precisely the steering signal we want.
    """
    avoid_radius = config.agent_radius + config.scripted_avoid_margin

    push = circle_box_static_forces(
        agent_pos, avoid_radius,
        config.wall_center, config.wall_halfsize, 1.0
    )

    # perpendicular nudge -- prevents deadlock when repulsion points exactly
    # opposite the steering direction and the two cancel to zero
    perp = torch.stack([-push[..., 1], push[..., 0]], dim=-1)
    return (push + 0.3 * perp) * config.scripted_avoid_gain


def _evasion(pred_off, rel_pos, config):
    """
    Steering away from the predator, ramping up as it closes.

    Only the locally nearest agent breaks off -- the rest keep pushing.
    The trained policy does not observe predator_target, so the committed
    hunt cannot be read off scenario_state. Each agent sees predator_offset
    and teammate relative positions, which is enough to tell whether it is
    the closest: |predator - other| = |predator_offset - rel_pos_other|.

    This disagrees with the predator the moment its commitment timer holds
    a target that is no longer the nearest. That is the information the
    observation does not contain.

    Agents out-run the predator (20 against a 6.0 speed cap), so fleeing works;
    the aim is only to survive contact, not win a chase.
    """
    dist = torch.norm(pred_off, dim=-1, keepdim=True)
    # predator_offset is predator - agent, so away is the opposite
    away = -pred_off / torch.clamp(dist, min=1e-6)
    urgency = torch.clamp(1.0 - dist / config.scripted_evade_radius, min=0.0)

    teammate_pred_dist = (pred_off.unsqueeze(-2) - rel_pos).norm(dim=-1)
    closest = dist.squeeze(-1) <= teammate_pred_dist.min(dim=-1).values
    closest = closest.unsqueeze(-1).to(pred_off.dtype)

    # sidestep instead of retreating straight back. A purely radial flee walks
    # the agent directly away from the payload and the predator just follows,
    # so the agent gives up ground every time without shaking anything off.
    tangent = torch.stack([-away[..., 1], away[..., 0]], dim=-1)
    evade = away + config.scripted_evade_tangent * tangent
    return evade * urgency * closest * config.scripted_evade_gain
