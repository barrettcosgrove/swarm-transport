"""Print reward terms for an agent moving inward along the goal line."""
import dataclasses

import torch

from env import scenario
from train.config import Config


distances = torch.linspace(0.20, 0.60, 9)
config = dataclasses.replace(Config(), num_envs=len(distances), n_agents=1)
world, state = scenario.reset(len(distances), config, torch.Generator().manual_seed(0))
payload = torch.zeros(len(distances), 2)
goal = torch.tensor([1.0, 0.0]).expand_as(payload)
agent = torch.stack((-distances, torch.zeros_like(distances)), dim=-1).unsqueeze(1)
previous = agent - torch.tensor([0.01, 0.0])
world = dataclasses.replace(
    world, agent_pos=agent, payload_pos=payload,
    predator_pos=torch.full_like(payload, 100.0),
    obstacle_center=torch.full_like(world.obstacle_center, 100.0))
state = dataclasses.replace(
    state, goal_pos=goal, prev_payload_dist=torch.ones_like(distances),
    prev_agent_pushpoint_dist=scenario.agent_pushpoint_geometry(
        previous, payload, goal, config.approach_target_standoff),
    predator_cooldown=torch.ones_like(distances))

terms = scenario.reward_terms(world, state, training_progress=0.0, config=config)
for index, distance in enumerate(distances):
    values = {name: round(float(value[index, 0]), 4) for name, value in terms.items()}
    print(f"distance={float(distance):.2f} {values}")
