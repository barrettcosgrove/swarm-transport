from dataclasses import dataclass
import torch
from .world import WorldState
from .physics import circle_box_static_forces, circle_circle_forces
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
    predator_noise: torch.Tensor      # (E, 2)
    
    
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

    world_state = WorldState(
        agent_pos=agent_pos, agent_vel=torch.zeros_like(agent_pos),
        predator_pos=predator_pos, predator_vel=torch.zeros_like(predator_pos),
        payload_pos=payload_pos, payload_vel=torch.zeros_like(payload_pos),
        agent_radius=config.agent_radius, agent_mass=config.agent_mass,
        predator_radius=config.predator_radius, predator_mass=config.predator_mass,
        payload_halfsize=config.payload_halfsize, payload_mass=config.payload_mass,
        wall_center=config.wall_center, wall_halfsize=config.wall_halfsize,
        # obstacles carry a per-env axis so reset_at can swap them per environment
        obstacle_center=config.obstacle_center.expand(E, -1, -1),
        obstacle_halfsize=config.obstacle_halfsize,
        obstacle_active=config.obstacle_active.expand(E, -1),
    )

    scenario_state = ScenarioState(
        goal_pos=goal_pos,
        health=torch.full((E,), config.max_health, device=dev),
        prev_health=torch.full((E,), config.max_health, device=dev),
        step_count=torch.zeros(E, dtype=torch.long, device=dev),
        predator_cooldown=torch.zeros(E, device=dev),
        prev_payload_dist=torch.norm(payload_pos - goal_pos, dim=-1),
        predator_noise=torch.zeros(E, 2, device=dev),
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
        prev_health=torch.where(mask1, fresh_scenario.prev_health, scenario_state.prev_health),
        predator_noise=torch.where(mask2, fresh_scenario.predator_noise, scenario_state.predator_noise),
    )

    return new_world, new_scenario
    
def observe(world_state, scenario_state, config) -> torch.Tensor: 
    E = world_state.agent_pos.shape[0]
    
    own_pos = world_state.agent_pos
    own_vel = world_state.agent_vel
    
    goal_offset = scenario_state.goal_pos.unsqueeze(1) - own_pos
    payload_offset = world_state.payload_pos.unsqueeze(1) - own_pos
    payload_rel_vel = world_state.payload_vel.unsqueeze(1) - own_vel
    
    rel_pos_other_agents = own_pos.unsqueeze(1) - own_pos.unsqueeze(2)
    rel_vel_other_agents = own_vel.unsqueeze(1) - own_vel.unsqueeze(2)
    
    col_index = [[j for j in range(config.n_agents) if j != i] for i in range(config.n_agents)]
    col_index = torch.tensor(col_index)
    row_index = torch.arange(config.n_agents).unsqueeze(1).expand(config.n_agents, config.n_agents-1)
    
    rel_pos_other_agents = rel_pos_other_agents[:, row_index, col_index].reshape(E, config.n_agents, -1)
    rel_vel_other_agents = rel_vel_other_agents[:, row_index, col_index].reshape(E, config.n_agents, -1)
    
    time_remaining = 1 - scenario_state.step_count / config.max_steps
    time_remaining = time_remaining.unsqueeze(1).expand(E, config.n_agents).unsqueeze(2)
    
    predator_offset = world_state.predator_pos.unsqueeze(1) - own_pos
    predator_rel_vel = world_state.predator_vel.unsqueeze(1) - own_vel
    
    shared_health = scenario_state.health.unsqueeze(1).expand(E, config.n_agents).unsqueeze(2)
    
    obs = torch.cat([own_pos, own_vel, goal_offset, payload_offset, payload_rel_vel, rel_pos_other_agents, rel_vel_other_agents, time_remaining, predator_offset, predator_rel_vel, shared_health], dim=-1)
    return obs

def compute_reward(world_state, scenario_state, training_progress, config) -> torch.Tensor:   # (E, n_agents)
    E = world_state.agent_pos.shape[0]
    
    curr_payload_dist = torch.norm(world_state.payload_pos - scenario_state.goal_pos, dim=-1)
    progress = scenario_state.prev_payload_dist - curr_payload_dist
    progress_reward = (config.progress_coef * progress).unsqueeze(1).expand(E, config.n_agents)
    
    time_penalty = torch.full((E, config.n_agents), -config.time_penalty_coef)
    
    success = curr_payload_dist < config.success_threshold
    success_reward = (success.float() * config.success_reward).unsqueeze(1).expand(E, config.n_agents)
    
    dist_agent_to_payload = torch.norm(world_state.payload_pos.unsqueeze(1) - world_state.agent_pos, dim=-1)
    anneal_fraction = min(training_progress / config.proximity_anneal_fraction, 1.0)
    proximity_coef = config.proximity_coef_start * (1 - anneal_fraction)
    proximity_reward = -proximity_coef * dist_agent_to_payload
    
    health_loss = scenario_state.prev_health - scenario_state.health
    health_loss_reward =  (-config.health_loss_coef * health_loss).unsqueeze(1).expand(E, config.n_agents)
    
    captured = (scenario_state.health).clone().detach() <= 0.0
    captured_reward = (captured.float() * config.captured_reward).unsqueeze(1).expand(E, config.n_agents)
    
    agent_wall_force = circle_box_static_forces(world_state.agent_pos, world_state.agent_radius, world_state.wall_center, world_state.wall_halfsize, config.wall_stiffness)
    agent_obstacle_force = circle_box_static_forces(world_state.agent_pos, world_state.agent_radius, world_state.obstacle_center, world_state.obstacle_halfsize, config.obstacle_stiffness)
    collision_magnitude = torch.norm(agent_wall_force + agent_obstacle_force, dim=-1)
    collision_reward =  -config.collision_coef * collision_magnitude
    
    reward = progress_reward + time_penalty + success_reward + proximity_reward + health_loss_reward + captured_reward + collision_reward
    return reward
    
def compute_done(world_state, scenario_state, config) -> tuple[torch.Tensor, torch.Tensor]:
    E = world_state.agent_pos.shape[0]
    
    curr_payload_dist = torch.norm(world_state.payload_pos - scenario_state.goal_pos, dim=-1)
    success = curr_payload_dist < config.success_threshold
    captured = (scenario_state.health).clone().detach() <= 0.0
    
    terminated = success | captured
    truncated = (scenario_state.step_count >= config.max_steps) & ~terminated
    
    return terminated, truncated

def predator_policy(world_state, scenario_state, config) -> (torch.Tensor, ScenarioState):  # (E, 2)   [Version 2]
    E = world_state.agent_pos.shape[0]
    
    dist_agent_to_predator = torch.norm(world_state.agent_pos - world_state.predator_pos.unsqueeze(1), dim=-1)
    dist_agent_to_payload = torch.norm(world_state.agent_pos - world_state.payload_pos.unsqueeze(1), dim=-1)
    score = dist_agent_to_predator + config.predator_weight * dist_agent_to_payload
    
    target = torch.argmin(score, dim=-1)
    target_pos = world_state.agent_pos[torch.arange(E), target]
    pred_pos = world_state.predator_pos
    
    to_target = target_pos - pred_pos
    direction = to_target / torch.norm(to_target, dim=-1, keepdim=True).clamp(min=1e-6)
    
    new_noise = (config.ou_decay * scenario_state.predator_noise) + (torch.randn_like(scenario_state.predator_noise) * config.ou_sigma)
    noisy_direction = direction + new_noise
    noisy_direction = noisy_direction / torch.norm(noisy_direction, dim=-1, keepdim=True).clamp(min=1e-6)
    
    throttle = torch.where(scenario_state.predator_cooldown > 0.0,
                           torch.full_like(scenario_state.predator_cooldown, config.predator_cooldown_speed_factor),
                           torch.ones_like(scenario_state.predator_cooldown))
    action = throttle.unsqueeze(1) * noisy_direction
    
    new_scenario_state = dataclasses.replace(scenario_state, predator_noise=new_noise)
    
    return action, new_scenario_state

def update_health(world_state, scenario_state, config) -> ScenarioState: 
    E = world_state.agent_pos.shape[0]
    
    agent_pos = world_state.agent_pos
    agent_radius = world_state.agent_radius
    predator_pos = world_state.predator_pos
    predator_radius = world_state.predator_radius
    
    force_on_agents, _ = circle_circle_forces(agent_pos, agent_radius, predator_pos, predator_radius, config.predator_stiffness)
    contact_magnitude = torch.norm(force_on_agents, dim=-1)
    total_contact_magnitude = contact_magnitude.sum(dim=-1)
    
    no_cooldown = scenario_state.predator_cooldown <= 0.0
    in_contact = total_contact_magnitude > 0.0
    deal_damage = no_cooldown & in_contact
    
    damage = torch.where(deal_damage, config.health_loss_per_step * total_contact_magnitude, torch.zeros_like(scenario_state.health))
    new_health = torch.clamp(scenario_state.health - damage, min=0.0)
    
    new_predator_cooldown = torch.where(deal_damage, torch.full_like(scenario_state.predator_cooldown, config.predator_cooldown_duration), torch.clamp(scenario_state.predator_cooldown - 1, min=0.0))
    
    updated_scenario_state = dataclasses.replace(scenario_state, health=new_health, prev_health=scenario_state.health, predator_cooldown=new_predator_cooldown)
    return updated_scenario_state
 


