from dataclasses import dataclass
import torch
from .world import WorldState
from .physics import circle_box_static_forces, circle_box_dynamic_forces
import math 
import dataclasses

@dataclass
class ScenarioState:
    goal_pos: torch.Tensor            # (E, 2)
    health: torch.Tensor              # (E,)
    prev_health: torch.Tensor         # (E,)
    step_count: torch.Tensor          # (E,)
    predator_cooldown: torch.Tensor   # (E,)
    prev_payload_dist: torch.Tensor   # (E,)
    # per-agent, unlike every other prev_ field here: the shaping term it
    # feeds is private to each agent
    prev_agent_pushpoint_dist: torch.Tensor # (E, n_agents)
    predator_noise: torch.Tensor      # (E, 2)
    predator_target: torch.Tensor     # (E,) long -- agent the predator is committed to
    predator_retarget_timer: torch.Tensor  # (E,) steps left before it may reconsider
    
    
def _sample_obstacle_centers(payload_pos, theta, config, generator) -> torch.Tensor:   # (E, n_obstacles, 2)
    """Obstacle centers for one batch of resets, valid by construction.

    Polar, like every other spawn: an angular sector drawn from the arc left
    over once the predator/goal corridor is excluded, and a radius uniform in
    the band. train/config.py carries the derivation of each bound; the short
    version is that band_min clears the agent annulus, band_max clears the
    goal, the corridor wedge clears the predator, and the minimum angular gap
    clears the other obstacles.

    Sectors are drawn without replacement rather than one obstacle per fixed
    quadrant, so layouts differ between episodes instead of being the same
    cross at a different rotation. Leaving a sector empty only ever widens a
    gap, so it cannot break the clearance guarantee.

    No rejection loop: every sample is legal the first time. A retry would
    have to be per-environment, which is exactly the thing a batched reset
    cannot do cheaply.
    """
    E = payload_pos.shape[0]
    dev = config.device
    n_obstacles = config.n_obstacles

    arc = 2 * math.pi - 2 * config.obstacle_corridor_half_angle
    sector_width = arc / config.obstacle_n_sectors
    # room to jitter inside a sector while still keeping the gap to the next
    # one. Negative means the sector count and the gap disagree, and every
    # clearance downstream is void.
    jitter_range = sector_width - config.obstacle_min_angular_gap
    assert jitter_range >= 0.0, "obstacle_n_sectors too high to hold obstacle_min_angular_gap"

    # argsort of uniform keys is a random permutation per env, so the first
    # n_obstacles entries are distinct sectors
    sector = torch.rand(E, config.obstacle_n_sectors, generator=generator, device=dev).argsort(dim=-1)[:, :n_obstacles]
    within_sector = torch.rand(E, n_obstacles, generator=generator, device=dev) * jitter_range

    angle = theta.unsqueeze(1) + config.obstacle_corridor_half_angle + sector * sector_width + within_sector
    radius = config.obstacle_band_min + torch.rand(E, n_obstacles, generator=generator, device=dev) * (config.obstacle_band_max - config.obstacle_band_min)

    offset = torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=-1)
    return payload_pos.unsqueeze(1) + offset


def reset(num_envs, config, generator) -> (WorldState, ScenarioState):
    E = num_envs
    dev = config.device

    # 1. payload — center + small jitter
    payload_pos = (torch.rand(E, 2, generator=generator, device=dev) - 0.5) \
                  * 2 * config.payload_jitter_radius

    # 2. agents — random angle each, radius uniform within the annulus
    agent_angles = torch.rand(E, config.n_agents, generator=generator, device=dev) * 2 * math.pi
    agent_radii = config.r_agent_min + torch.rand(E, config.n_agents, generator=generator, device=dev) \
                  * (config.r_agent_max - config.r_agent_min)
    agent_offset = torch.stack([agent_radii * torch.cos(agent_angles),
                                 agent_radii * torch.sin(agent_angles)], dim=-1)
    agent_pos = payload_pos.unsqueeze(1) + agent_offset          # (E, n_agents, 2)

    # 3. predator — free angle theta, fixed radius beyond the agent annulus
    theta = torch.rand(E, generator=generator, device=dev) * 2 * math.pi
    predator_pos = payload_pos + config.predator_spawn_radius \
                   * torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    # 4. goal — same angle, further out, small jitter
    goal_theta = theta + (torch.rand(E, generator=generator, device=dev) - 0.5) \
                 * 2 * config.goal_angle_jitter
    goal_pos = payload_pos + config.goal_radius \
               * torch.stack([torch.cos(goal_theta), torch.sin(goal_theta)], dim=-1)

    # 5. obstacles — a ring between the agent annulus and the goal, placed
    # last so they can be kept clear of everything already on the board
    obstacle_center = _sample_obstacle_centers(payload_pos, theta, config, generator)

    world_state = WorldState(
        agent_pos=agent_pos, agent_vel=torch.zeros_like(agent_pos),
        predator_pos=predator_pos, predator_vel=torch.zeros_like(predator_pos),
        payload_pos=payload_pos, payload_vel=torch.zeros_like(payload_pos),
        agent_radius=config.agent_radius, agent_mass=config.agent_mass,
        predator_radius=config.predator_radius, predator_mass=config.predator_mass,
        payload_halfsize=config.payload_halfsize, payload_mass=config.payload_mass,
        wall_center=config.wall_center, wall_halfsize=config.wall_halfsize,
        # obstacles carry a per-env axis so reset_at can swap them per environment
        obstacle_center=obstacle_center,
        obstacle_halfsize=config.obstacle_halfsize,
        obstacle_active=config.obstacle_active.expand(E, -1),
    )

    agent_pushpoint_dist = agent_pushpoint_geometry(
        agent_pos, payload_pos, goal_pos, config.approach_target_standoff)

    scenario_state = ScenarioState(
        goal_pos=goal_pos,
        health=torch.full((E,), config.max_health, device=dev),
        prev_health=torch.full((E,), config.max_health, device=dev),
        step_count=torch.zeros(E, dtype=torch.long, device=dev),
        predator_cooldown=torch.zeros(E, device=dev),
        prev_payload_dist=torch.norm(payload_pos - goal_pos, dim=-1),
        # seeded from the spawn state, so the first step of an episode scores a
        # real one-step delta rather than a jump from zero
        prev_agent_pushpoint_dist=agent_pushpoint_dist,
        predator_noise=torch.zeros(E, 2, device=dev),
        # timer at zero so the first step picks a target normally
        predator_target=torch.zeros(E, dtype=torch.long, device=dev),
        predator_retarget_timer=torch.zeros(E, device=dev),
    )
    return world_state, scenario_state

def reset_at(world_state, scenario_state, needs_reset, config, generator) -> tuple[WorldState, ScenarioState]:
    E = world_state.agent_pos.shape[0]
    fresh_world, fresh_scenario = reset(E, config, generator)

    mask3 = needs_reset.view(E, 1, 1)
    mask2 = needs_reset.unsqueeze(-1)
    mask1 = needs_reset

    new_world = dataclasses.replace(world_state,
        agent_pos=torch.where(mask3, fresh_world.agent_pos, world_state.agent_pos),
        agent_vel=torch.where(mask3, fresh_world.agent_vel, world_state.agent_vel),
        obstacle_center=torch.where(mask3, fresh_world.obstacle_center, world_state.obstacle_center),
        predator_pos=torch.where(mask2, fresh_world.predator_pos, world_state.predator_pos),
        predator_vel=torch.where(mask2, fresh_world.predator_vel, world_state.predator_vel),
        payload_pos=torch.where(mask2, fresh_world.payload_pos, world_state.payload_pos),
        payload_vel=torch.where(mask2, fresh_world.payload_vel, world_state.payload_vel),
    )

    new_scenario = dataclasses.replace(scenario_state,
        goal_pos=torch.where(mask2, fresh_scenario.goal_pos, scenario_state.goal_pos),
        health=torch.where(mask1, fresh_scenario.health, scenario_state.health),
        step_count=torch.where(mask1, fresh_scenario.step_count, scenario_state.step_count),
        predator_cooldown=torch.where(mask1, fresh_scenario.predator_cooldown, scenario_state.predator_cooldown),
        prev_payload_dist=torch.where(mask1, fresh_scenario.prev_payload_dist, scenario_state.prev_payload_dist),
        # mask2, not mask1: this is (E, n_agents), so a bare (E,) mask would
        # broadcast across the agent axis instead of the env axis
        prev_agent_pushpoint_dist=torch.where(mask2, fresh_scenario.prev_agent_pushpoint_dist, scenario_state.prev_agent_pushpoint_dist),
        prev_health=torch.where(mask1, fresh_scenario.prev_health, scenario_state.prev_health),
        predator_noise=torch.where(mask2, fresh_scenario.predator_noise, scenario_state.predator_noise),
        predator_target=torch.where(mask1, fresh_scenario.predator_target, scenario_state.predator_target),
        predator_retarget_timer=torch.where(mask1, fresh_scenario.predator_retarget_timer, scenario_state.predator_retarget_timer),
    )

    return new_world, new_scenario
    
_NEIGHBOR_INDEX_CACHE = {}


def _neighbor_index(n_agents, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(row, col) gather indices selecting every agent's teammates.

    Cached rather than rebuilt per call: observe() runs every step, the indices
    depend only on n_agents, and building them fresh both allocated needlessly
    and left them on the CPU regardless of where the tensors being gathered
    live -- which fails outright on CUDA.
    """
    key = (n_agents, str(device))
    if key not in _NEIGHBOR_INDEX_CACHE:
        col = torch.tensor([[j for j in range(n_agents) if j != i] for i in range(n_agents)],
                           device=device)
        row = torch.arange(n_agents, device=device).unsqueeze(1).expand(n_agents, n_agents - 1)
        _NEIGHBOR_INDEX_CACHE[key] = (row, col)
    return _NEIGHBOR_INDEX_CACHE[key]


def observe(world_state, scenario_state, config) -> torch.Tensor: 
    E = world_state.agent_pos.shape[0]
    
    own_pos = world_state.agent_pos / config.obs_pos_scale
    own_vel = world_state.agent_vel / config.obs_vel_scale
    
    goal_offset = (scenario_state.goal_pos.unsqueeze(1) - world_state.agent_pos) / config.obs_pos_scale
    payload_offset = (world_state.payload_pos.unsqueeze(1) - world_state.agent_pos) / config.obs_pos_scale
    payload_rel_vel = (world_state.payload_vel.unsqueeze(1) - world_state.agent_vel) / config.obs_vel_scale
    
    rel_pos_other_agents = world_state.agent_pos.unsqueeze(1) - world_state.agent_pos.unsqueeze(2)
    rel_vel_other_agents = world_state.agent_vel.unsqueeze(1) - world_state.agent_vel.unsqueeze(2)
    
    row_index, col_index = _neighbor_index(config.n_agents, world_state.agent_pos.device)
    
    rel_pos_other_agents = rel_pos_other_agents[:, row_index, col_index].reshape(E, config.n_agents, -1) / config.obs_pos_scale
    rel_vel_other_agents = rel_vel_other_agents[:, row_index, col_index].reshape(E, config.n_agents, -1) / config.obs_vel_scale
    
    time_remaining = 1 - scenario_state.step_count / config.max_steps
    time_remaining = time_remaining.unsqueeze(1).expand(E, config.n_agents).unsqueeze(2)
    
    predator_offset = (world_state.predator_pos.unsqueeze(1) - world_state.agent_pos) / config.obs_pos_scale
    predator_rel_vel = (world_state.predator_vel.unsqueeze(1) - world_state.agent_vel) / config.obs_vel_scale
    
    shared_health = scenario_state.health.unsqueeze(1).expand(E, config.n_agents).unsqueeze(2) / config.max_health
    
    obs = torch.cat([own_pos, own_vel, goal_offset, payload_offset, payload_rel_vel, rel_pos_other_agents, rel_vel_other_agents, time_remaining, predator_offset, predator_rel_vel, shared_health], dim=-1)
    return obs

def agent_pushpoint_geometry(agent_pos, payload_pos, goal_pos, standoff):
    """Distance from each agent to the standoff point behind the payload,
    on the payload->goal line. Shared by reset, compute_reward and env.step
    rather than written out at each site: the approach term is a difference
    between this step's distance and last step's, so a discrepancy between
    how the two ends are measured would show up as a reward the agent cannot
    influence. The target is where a push actually helps, not the payload's
    center -- so closing this distance and being usefully positioned are the
    same thing.
    """
    to_goal = goal_pos - payload_pos
    goal_dir = to_goal / to_goal.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    push_point = payload_pos - goal_dir * standoff          # (E, 2)
    dist = (push_point.unsqueeze(1) - agent_pos).norm(dim=-1)  # (E, n_agents)
    return dist


def payload_side_masks(world_state, scenario_state, zone_radius):
    """Agents inside ``zone_radius`` of the payload, split by side of the goal line.

    Behind is cosine < -0.4 (the push side). Front is cosine > 0.4 (blocking).
    Shared by the trainer log and tools.evaluate so the two reports cannot
    drift apart on the metric that gates the reward change.
    """
    rel = world_state.agent_pos - world_state.payload_pos.unsqueeze(1)
    dist = rel.norm(dim=-1)
    goal_dir = scenario_state.goal_pos - world_state.payload_pos
    goal_dir = goal_dir / goal_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    side = (rel * goal_dir.unsqueeze(1)).sum(-1) / dist.clamp(min=1e-6)
    in_zone = dist <= zone_radius
    return in_zone & (side < -0.4), in_zone & (side > 0.4)


# Closing above this is the population where fleeing is the right answer.
# Shared by evasion_step_stats and tools/threat_probe so the split cannot drift.
PREDATOR_CLOSING_ON = 0.2
# Predator sitting on the crate. Shared by evaluate and the trainer log.
PREDATOR_CAMPING_PAYLOAD_DIST = 0.75


def predator_evade_masks(world_state, actions, danger_radius):
    """cos(action, away from predator) and who is inside ``danger_radius``.

    Sign is the whole question: negative means agents steer into the thing
    damaging them, which is what variant A measured at -0.19. Shared by the
    trainer log and tools.evaluate so the two reports cannot drift apart on
    the metric that gates the evasion change.
    """
    to_pred = world_state.predator_pos.unsqueeze(1) - world_state.agent_pos
    dist = to_pred.norm(dim=-1)
    away = -to_pred / dist.unsqueeze(-1).clamp(min=1e-6)
    cosine = (actions * away).sum(-1) / actions.norm(dim=-1).clamp(min=1e-6)
    return cosine, dist <= danger_radius


def predator_hunted_mask(scenario_state, n_agents):
    """Which agent the predator is committed to, as an (E, n_agents) bool."""
    return torch.nn.functional.one_hot(
        scenario_state.predator_target, num_classes=n_agents).bool()


def predator_camping(world_state, payload_dist=PREDATOR_CAMPING_PAYLOAD_DIST):
    """True on environments where the predator is sitting on the crate."""
    return (world_state.predator_pos - world_state.payload_pos).norm(dim=-1) < payload_dist


def _masked_sum_count(mask, values):
    n = int(mask.sum())
    return (float(values[mask].sum()) if n else 0.0), n


def evasion_step_stats(world_state, scenario_state, actions, config):
    """Per-step evasion sums and counts.

    Shared by the trainer log and tools.evaluate so the two reports cannot
    drift apart on the metrics that gate the camping term: unsigned
    evade_cosine was blind to 19% vs 0% capture.
    """
    cosine, in_danger = predator_evade_masks(
        world_state, actions, config.predator_danger_radius)
    dist, _, closing = predator_closing(world_state, config)
    hunted = predator_hunted_mask(scenario_state, config.n_agents)
    a_norm = actions.norm(dim=-1)
    hunted_d = hunted & in_danger
    other_d = (~hunted) & in_danger
    closing_on = closing > PREDATOR_CLOSING_ON
    hunted_cos_sum, hunted_n = _masked_sum_count(hunted_d, cosine)
    hunted_thr_sum, _ = _masked_sum_count(hunted_d, a_norm)
    other_cos_sum, other_n = _masked_sum_count(other_d, cosine)
    other_thr_sum, _ = _masked_sum_count(other_d, a_norm)
    closing_thr_sum, closing_n = _masked_sum_count(closing_on, a_norm)
    evade_sum, evade_n = _masked_sum_count(in_danger, cosine)
    return {
        "evade_cosine_sum": evade_sum,
        "evade_count": evade_n,
        "hunted_cosine_sum": hunted_cos_sum,
        "hunted_thrust_sum": hunted_thr_sum,
        "hunted_count": hunted_n,
        "other_cosine_sum": other_cos_sum,
        "other_thrust_sum": other_thr_sum,
        "other_count": other_n,
        "closing_thrust_sum": closing_thr_sum,
        "closing_count": closing_n,
        "capture_hits": int((dist < config.predator_capture_radius).sum()),
        "agent_steps": int(dist.numel()),
        "camping_hits": int(predator_camping(world_state).sum()),
        "env_steps": int(world_state.payload_pos.shape[0]),
    }


def hunted_flee_term(world_state, scenario_state, actions, config):
    """Private bonus for the scripted evade action.

    ``flee_coef * hunted * in_danger * closing * cos(away) * |action|``.
    Zero for everyone except the committed target, zero when the hunter
    is parked (closing=0), zero outside danger_radius. Signed: thrusting
    at the predator costs, thrusting away pays. ``|action|`` is clamped
    at 1 so a diagonal cannot out-earn a unit-norm flee.
    """
    E, N = world_state.agent_pos.shape[:2]
    zeros = torch.zeros(E, N, device=world_state.agent_pos.device,
                        dtype=world_state.agent_pos.dtype)
    if actions is None or config.flee_coef == 0.0:
        return zeros
    cosine, in_danger = predator_evade_masks(
        world_state, actions, config.predator_danger_radius)
    _, _, closing = predator_closing(world_state, config)
    hunted = predator_hunted_mask(scenario_state, config.n_agents)
    thrust = actions.norm(dim=-1).clamp(max=1.0)
    gate = hunted.float() * in_danger.float() * closing
    return config.flee_coef * gate * cosine * thrust


def predator_closing(world_state, config):
    """Distance, intrusion in [0, 1], and closing rate in [0, 1].

    Closing is the component of relative velocity along the line from agent
    to predator, normalized by predator_max_speed. Zero when the gap is not
    shrinking -- a hunter parked on the crate costs nothing. Shared by
    reward_terms and tools/threat_calibrate so the sweep cannot drift from
    the term it is sizing.
    """
    to_pred = world_state.predator_pos.unsqueeze(1) - world_state.agent_pos
    dist = to_pred.norm(dim=-1)
    unit = to_pred / dist.clamp(min=1e-6).unsqueeze(-1)
    intrusion = ((config.predator_danger_radius - dist)
                 .clamp(min=0.0) / config.predator_danger_radius)
    rel_vel = world_state.predator_vel.unsqueeze(1) - world_state.agent_vel
    closing = (-(rel_vel * unit).sum(-1)).clamp(min=0.0) / config.predator_max_speed
    return dist, intrusion, closing.clamp(max=1.0)


def payload_goalward_force(world_state, scenario_state, config):
    """Net agent force on the payload, projected onto the goal direction.

    Returns ``(goalward, magnitude)``, both (E,). Efficiency is their ratio.
    """
    _, force_on_payload = circle_box_dynamic_forces(
        world_state.agent_pos, world_state.agent_radius,
        world_state.payload_pos, world_state.payload_halfsize, config.payload_stiffness)
    goal_dir = scenario_state.goal_pos - world_state.payload_pos
    goal_dir = goal_dir / goal_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return (force_on_payload * goal_dir).sum(-1), force_on_payload.norm(dim=-1)


def goalward_push_penetration(world_state, scenario_state, config) -> torch.Tensor:
    """Signed contact penetration toward the goal, before ``push_coef``."""
    force_on_agent, _ = circle_box_dynamic_forces(
        world_state.agent_pos, world_state.agent_radius,
        world_state.payload_pos, world_state.payload_halfsize, config.payload_stiffness)
    goal_dir = scenario_state.goal_pos - world_state.payload_pos
    goal_dir = goal_dir / goal_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return ((-force_on_agent) * goal_dir.unsqueeze(1)).sum(-1) / config.payload_stiffness


def reward_terms(world_state, scenario_state, training_progress, config,
                 actions=None, action_world_state=None, action_scenario_state=None) -> dict[str, torch.Tensor]:
    """Every reward component separately, each (E, n_agents), keyed by name.

    compute_reward is the sum of these. Split apart because the totals hide the
    thing you need when a policy misbehaves: a run whose return sits at -400
    could be paying for the clock, for damage, or for shoving the payload the
    wrong way, and those call for opposite fixes. Diagnosing the camping/suicide
    failure took a throwaway script to recover numbers this function returns for
    free, and train/mappo.py now logs them per iteration.

    Which terms are private to an agent and which are shared across the team is
    the load-bearing distinction here, so it is called out per term below. A
    shared term is identical for all n_agents, so an individual's action barely
    moves it, and the policy gradient largely ignores it.
    """
    E = world_state.agent_pos.shape[0]
    dev = world_state.agent_pos.device

    curr_payload_dist = torch.norm(world_state.payload_pos - scenario_state.goal_pos, dim=-1)
    progress = scenario_state.prev_payload_dist - curr_payload_dist
    # SHARED + PRIVATE: the raw progress is a team outcome. Attribute most of
    # it to agents inside push_zone_radius so a pusher and a blocker no longer
    # receive the same number. Split among whoever is in the zone (unlike
    # health, which is flat) so the team total matches the old shared term.
    in_zone = ((world_state.agent_pos - world_state.payload_pos.unsqueeze(1))
               .norm(dim=-1) <= config.push_zone_radius)
    n_zone = in_zone.sum(-1, keepdim=True).clamp(min=1.0)
    blame = config.progress_blame_fraction
    shared = (1.0 - blame) * progress.unsqueeze(1).expand(E, config.n_agents)
    credited = blame * progress.unsqueeze(1) * in_zone * config.n_agents / n_zone
    progress_reward = config.progress_coef * (shared + credited)
    
    time_penalty = torch.full((E, config.n_agents), -config.time_penalty_coef,
                              device=world_state.agent_pos.device)
    
    success = curr_payload_dist < config.success_threshold
    success_reward = (success.float() * config.success_reward).unsqueeze(1).expand(E, config.n_agents)

    # PRIVATE: distance closed toward the standoff point. Job is purely to get
    # an agent from spawn into pushing position. Goes silent once an agent is
    # holding the point (target moves with the payload), so it cannot reward
    # sustained pushing -- that is push_reward's job below. Anneals when
    # approach_anneal_fraction > 0; 0.0 keeps the coefficient at start for the
    # whole run, which is the schedule that taught return-to-the-box.
    curr_pushpoint_dist = agent_pushpoint_geometry(
        world_state.agent_pos, world_state.payload_pos,
        scenario_state.goal_pos, config.approach_target_standoff)
    if config.approach_anneal_fraction <= 0.0:
        approach_coef = config.approach_coef_start
    else:
        approach_anneal = min(training_progress / config.approach_anneal_fraction, 1.0)
        approach_coef = config.approach_coef_start * (1 - approach_anneal)
    
    approach_reward = approach_coef * (
        scenario_state.prev_agent_pushpoint_dist - curr_pushpoint_dist)

    # PRIVATE, NOT ANNEALED: this agent's own contact force on the payload,
    # projected onto the goal direction. Zero with no overlap -- cannot be
    # farmed by hovering at the push point. Positive only while this agent's
    # contact is actually moving the payload toward the goal. This is the
    # term that pays during sustained pushing, which approach_reward cannot.
    push = goalward_push_penetration(world_state, scenario_state, config)
    push_reward = config.push_coef * push

    # PRIVATE: each agent's own distance to the predator. Zero beyond
    # predator_danger_radius, so this is "you are about to be hit", not a
    # standing reason to avoid the payload the predator guards. Squared so
    # both the value and its derivative vanish at the boundary.
    #
    # Gated on closing rate, not on cooldown or raw proximity. Variant B's
    # distance field taxed anyone near a hunter that camps the crate, and
    # the policy correctly left: episode threat -138 against progress +0.4,
    # agent-payload distance 0.80 -> 2.44, health unchanged. Closing is
    # zero while the predator is parked or moving away, so the push zone
    # stays free; it fires only while the gap is shrinking. A cooling
    # predator is already speed-capped at half, so the physics does the
    # gating that threat_cooldown_factor used to.
    dist_to_predator, intrusion, closing = predator_closing(world_state, config)
    threat_reward = -config.threat_coef * intrusion.pow(2) * closing

    # PRIVATE, NOT GATED ON CLOSING: standing in the capture bubble while the
    # hunter is parked. Closing-gated threat is identically zero there, which
    # is the freeze variant D measured -- 32-47% of agents still in the push
    # zone with the predator on the crate. Squared so value and derivative
    # vanish at camp_radius. Tight by construction: beyond ~0.8 this becomes
    # a standing reason to abandon the crate, which is variant B.
    camp_intrusion = ((config.predator_camp_radius - dist_to_predator)
                      .clamp(min=0.0) / config.predator_camp_radius)
    camp_reward = -config.camp_coef * camp_intrusion.pow(2)

    # PRIVATE: the scripted action, not a position tax. Scored on the
    # board the action was chosen against (action_world_state), matching
    # evasion_step_stats. Post-step geometry would credit a flee to a
    # gap that has already moved.
    flee_ws = action_world_state if action_world_state is not None else world_state
    flee_ss = action_scenario_state if action_scenario_state is not None else scenario_state
    flee_reward = hunted_flee_term(flee_ws, flee_ss, actions, config)

    # SHARED + PRIVATE: update_health drains a flat health_loss_per_step on
    # any() over agents, so the pool itself does not know who was hit. This
    # split is the only place per-agent responsibility can enter. The private
    # share is flat -- blame * health_loss for every agent in range -- not
    # divided among them, because dividing would make an agent's own bill
    # shrink as teammates joined it in the danger zone.
    in_capture = dist_to_predator < config.predator_capture_radius
    health_loss = scenario_state.prev_health - scenario_state.health
    blame = config.health_loss_blame_fraction
    shared = (1.0 - blame) * health_loss.unsqueeze(1).expand(E, config.n_agents)
    private = torch.where(
        in_capture,
        blame * health_loss.unsqueeze(1).expand(E, config.n_agents),
        torch.zeros(E, config.n_agents, device=dev),
    )
    health_loss_reward = -config.health_loss_coef * (shared + private)

    # SHARED: health hitting zero is a team outcome. Whoever happened to be
    # touching at the last tick is not the cause.
    captured = (scenario_state.health).clone().detach() <= 0.0
    captured_reward = (captured.float() * config.captured_reward).unsqueeze(1).expand(E, config.n_agents)

    agent_wall_force = circle_box_static_forces(world_state.agent_pos, world_state.agent_radius, world_state.wall_center, world_state.wall_halfsize, config.wall_stiffness)
    agent_obstacle_force = circle_box_static_forces(world_state.agent_pos, world_state.agent_radius, world_state.obstacle_center, world_state.obstacle_halfsize, config.obstacle_stiffness)
    collision_magnitude = torch.norm(agent_wall_force + agent_obstacle_force, dim=-1)
    collision_reward =  -config.collision_coef * collision_magnitude

    return {
        "progress": progress_reward,
        "time": time_penalty,
        "success": success_reward,
        "approach": approach_reward,
        "push": push_reward,
        "health": health_loss_reward,
        "captured": captured_reward,
        "collision": collision_reward,
        "threat": threat_reward,
        "camp": camp_reward,
        "flee": flee_reward,
    }


def compute_reward(world_state, scenario_state, training_progress, config, actions=None) -> torch.Tensor:   # (E, n_agents)
    return torch.stack(tuple(reward_terms(world_state, scenario_state, training_progress, config, actions=actions).values())).sum(0)

def compute_done(world_state, scenario_state, config) -> tuple[torch.Tensor, torch.Tensor]:
    curr_payload_dist = torch.norm(world_state.payload_pos - scenario_state.goal_pos, dim=-1)
    success = curr_payload_dist < config.success_threshold
    captured = (scenario_state.health).clone().detach() <= 0.0
    
    terminated = success | captured
    truncated = (scenario_state.step_count >= config.max_steps) & ~terminated
    
    return terminated, truncated

def predator_policy(world_state, scenario_state, config, generator=None) -> (torch.Tensor, ScenarioState):  # (E, 2)
    E = world_state.agent_pos.shape[0]
    
    dist_agent_to_predator = torch.norm(world_state.agent_pos - world_state.predator_pos.unsqueeze(1), dim=-1)
    dist_agent_to_payload = torch.norm(world_state.agent_pos - world_state.payload_pos.unsqueeze(1), dim=-1)
    score = dist_agent_to_predator + config.predator_weight * dist_agent_to_payload
    
    # Commit to a target for a while instead of re-deciding every step. Pure
    # per-step argmin switched targets ~73 times an episode, because the moment
    # an agent flees it stops being the nearest and the predator turns on
    # whoever stayed to push. That makes evading impossible by construction,
    # and it makes the predator's behaviour high-frequency noise for a policy
    # trying to learn against it.
    timer = scenario_state.predator_retarget_timer
    retarget = timer <= 0.0
    target = torch.where(retarget, torch.argmin(score, dim=-1), scenario_state.predator_target)
    new_timer = torch.where(retarget,
                            torch.full_like(timer, config.predator_target_commit_steps),
                            torch.clamp(timer - 1.0, min=0.0))
    
    target_pos = world_state.agent_pos[torch.arange(E), target]
    pred_pos = world_state.predator_pos
    
    to_target = target_pos - pred_pos
    direction = to_target / torch.norm(to_target, dim=-1, keepdim=True).clamp(min=1e-6)
    
    # randn with the env's generator, not randn_like: randn_like draws from the
    # global RNG, which left rollouts irreproducible even under a fixed seed
    noise_sample = torch.randn(scenario_state.predator_noise.shape, generator=generator,
                               device=scenario_state.predator_noise.device,
                               dtype=scenario_state.predator_noise.dtype)
    new_noise = (config.ou_decay * scenario_state.predator_noise) + (noise_sample * config.ou_sigma)
    noisy_direction = direction + new_noise
    noisy_direction = noisy_direction / torch.norm(noisy_direction, dim=-1, keepdim=True).clamp(min=1e-6)
    
    # cooldown slows the predator by capping its speed, not its thrust -- see
    # effective_predator_max_speed, applied in env.step
    action = noisy_direction
    
    new_scenario_state = dataclasses.replace(scenario_state, predator_noise=new_noise,
                                             predator_target=target,
                                             predator_retarget_timer=new_timer)
    
    return action, new_scenario_state


def effective_predator_max_speed(scenario_state, config) -> torch.Tensor:   # (E, 1)
    """The predator's speed cap this step, halved while it is cooling down.

    DESIGN.md expresses the cooldown as a speed cap rather than a thrust cut so
    that a cooling predator still turns as sharply, it just cannot close as
    fast. Shaped (E, 1) to broadcast against predator_vel in physics.integrate.
    """
    cooldown = scenario_state.predator_cooldown
    capped = torch.where(
        cooldown > 0.0,
        torch.full_like(cooldown, config.predator_max_speed * config.predator_cooldown_speed_factor),
        torch.full_like(cooldown, config.predator_max_speed),
    )
    return capped.unsqueeze(-1)

def update_health(world_state, scenario_state, config) -> ScenarioState: 
    """Drain a flat health_loss_per_step whenever any agent is inside the
    predator's capture radius, per DESIGN.md section 2.

    Deliberately not proportional to contact force. Penalty forces scale with
    penetration depth, and penetration depth is set by how far a body travels
    in one dt -- agents cover 1.0 units per step against a 0.25 contact
    distance, so a force-scaled drain made a single touch cost anywhere from 1
    to 100 health depending on the approach angle. Scaling that down moves the
    whole distribution but leaves the spread alone; the p95 stayed at roughly
    3x the median at every scale tested. A flat rate makes health a legible
    budget: max_health / health_loss_per_step contacts, whatever the geometry.

    Capture radius is its own constant rather than the sum of body radii, so
    "close enough to be hurt" can be tuned without resizing the bodies.
    """
    dist_to_predator = torch.norm(world_state.agent_pos - world_state.predator_pos.unsqueeze(1), dim=-1)
    
    # any(), not sum(): the health pool is shared, so a second agent in range
    # is already caught by the first one's drain
    in_capture = (dist_to_predator < config.predator_capture_radius).any(dim=-1)
    
    no_cooldown = scenario_state.predator_cooldown <= 0.0
    deal_damage = no_cooldown & in_capture
    
    damage = torch.where(deal_damage,
                         torch.full_like(scenario_state.health, config.health_loss_per_step),
                         torch.zeros_like(scenario_state.health))
    new_health = torch.clamp(scenario_state.health - damage, min=0.0)
    
    new_predator_cooldown = torch.where(deal_damage, torch.full_like(scenario_state.predator_cooldown, config.predator_cooldown_duration), torch.clamp(scenario_state.predator_cooldown - 1, min=0.0))
    
    updated_scenario_state = dataclasses.replace(scenario_state, health=new_health, prev_health=scenario_state.health, predator_cooldown=new_predator_cooldown)
    return updated_scenario_state
 


