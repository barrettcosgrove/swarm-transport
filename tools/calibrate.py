"""
tools/calibrate.py

Measures the payload's actual cruising speed under the current physics
constants, and recommends max_steps so solo single-agent traversal consumes
50-70% of the episode -- the ratio from DESIGN.md's arena-sizing procedure.

Run this any time dt, stiffness, mass, or thrust change -- the "right"
max_steps depends on the whole chain of constants, not dt alone.
"""
import dataclasses

import torch
import env.scenario as scenario
import env.physics as physics


def measure_cruising_speed(config, warmup_steps=50, measure_steps=100, seed=0):
    # One agent, placed by hand rather than sampled. The spawn annulus is sized
    # against the arena, so a randomly placed agent can easily start far enough
    # from the payload that it never makes contact and this reads zero.
    config = dataclasses.replace(config, n_agents=1)
    generator = torch.Generator().manual_seed(seed)
    world_state, scenario_state = scenario.reset(1, config, generator)

    goal_dir = scenario_state.goal_pos - world_state.payload_pos
    goal_dir = goal_dir / torch.norm(goal_dir, dim=-1, keepdim=True)

    # agent against the payload's trailing face, already in contact
    standoff = float(config.payload_halfsize.max()) + config.agent_radius
    agent_pos = (world_state.payload_pos - goal_dir * standoff).unsqueeze(1)

    # predator moved off to the side. It spawns on the payload->goal ray, so
    # left alone this measures the speed of shoving a payload and a predator.
    perp = torch.stack([-goal_dir[:, 1], goal_dir[:, 0]], dim=-1)
    predator_pos = world_state.payload_pos + perp * config.goal_radius

    # obstacles parked out of reach, same reasoning. Obstacle centers are
    # sampled per episode and are allowed to sit on the payload->goal line;
    # this shove has no avoidance in it, so one in the way would be measured
    # as the payload being slow rather than blocked.
    obstacle_center = torch.full_like(world_state.obstacle_center, 1e6)

    world_state = dataclasses.replace(
        world_state,
        agent_pos=agent_pos, agent_vel=torch.zeros_like(agent_pos),
        predator_pos=predator_pos, predator_vel=torch.zeros_like(predator_pos),
        obstacle_center=obstacle_center,
    )

    agent_action = goal_dir.unsqueeze(1)  # sustained max effort, straight at the goal
    predator_action = torch.zeros(1, 2)   # predator held still -- keep the measurement clean

    def do_step(ws):
        return physics.step(
            ws, agent_action, predator_action, config.dt,
            config.agent_max_thrust, config.predator_max_thrust,
            config.agent_drag_coef, config.predator_drag_coef, config.payload_drag_coef,
            config.body_stiffness, config.wall_stiffness, config.obstacle_stiffness, config.payload_stiffness,
            config.predator_max_speed,
        )

    for _ in range(warmup_steps):        # let thrust/drag reach equilibrium before measuring
        world_state = do_step(world_state)

    start_pos = world_state.payload_pos.clone()
    for _ in range(measure_steps):
        world_state = do_step(world_state)
    end_pos = world_state.payload_pos.clone()

    distance = torch.norm(end_pos - start_pos, dim=-1).item()
    elapsed = measure_steps * config.dt
    return distance / elapsed


def recommend_max_steps(config, target_fraction=0.6):
    speed = measure_cruising_speed(config)
    traversal_time = config.goal_radius / speed
    total_budget = traversal_time / target_fraction
    return int(total_budget / config.dt), speed


if __name__ == "__main__":
    from train.config import Config
    config = Config()
    recommended, speed = recommend_max_steps(config)
    print(f"measured cruising speed: {speed:.3f} units/sec")
    print(f"current max_steps: {config.max_steps}")
    print(f"recommended max_steps: {recommended}")