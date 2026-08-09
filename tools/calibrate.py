"""
tools/calibrate.py

Measures the payload's actual cruising speed under the current physics
constants, and recommends max_steps so solo single-agent traversal consumes
50-70% of the episode -- the ratio from DESIGN.md's arena-sizing procedure.

Run this any time dt, stiffness, mass, or thrust change -- the "right"
max_steps depends on the whole chain of constants, not dt alone.
"""
import torch
import env.scenario as scenario
import env.physics as physics


def measure_cruising_speed(config, warmup_steps=50, measure_steps=100, seed=0):
    generator = torch.Generator().manual_seed(seed)
    world_state, scenario_state = scenario.reset(1, config, generator)

    goal_dir = scenario_state.goal_pos - world_state.payload_pos
    goal_dir = goal_dir / torch.norm(goal_dir, dim=-1, keepdim=True)

    agent_action = torch.zeros(1, config.n_agents, 2)
    agent_action[:, 0] = goal_dir       # only agent 0 pushes, straight toward the goal
    predator_action = torch.zeros(1, 2)  # predator held still -- keep the measurement clean

    def do_step(ws):
        return physics.step(
            ws, agent_action, predator_action, config.dt,
            config.agent_max_thrust, config.predator_max_thrust,
            config.agent_drag_coef, config.predator_drag_coef, config.payload_drag_coef,
            config.body_stiffness, config.wall_stiffness, config.obstacle_stiffness, config.payload_stiffness,
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