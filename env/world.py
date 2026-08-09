from dataclasses import dataclass
import torch

@dataclass
class WorldState:
    agent_pos: torch.Tensor          # (E, n_agents, 2)
    agent_vel: torch.Tensor          # (E, n_agents, 2)
    agent_radius: torch.Tensor       # scalar or (n_agents,) -- homogeneous agents
    agent_mass: torch.Tensor         # (E, n_agents, 1) -- corrected shape

    predator_pos: torch.Tensor       # (E, 2)
    predator_vel: torch.Tensor       # (E, 2)
    predator_radius: torch.Tensor    # scalar
    predator_mass: torch.Tensor      # (E, 1)

    payload_pos: torch.Tensor        # (E, 2)
    payload_vel: torch.Tensor        # (E, 2)
    payload_halfsize: torch.Tensor   # (2,)
    payload_mass: torch.Tensor       # (E, 1)

    wall_center: torch.Tensor        # (n_walls, 2)     -- no E axis, same for every env
    wall_halfsize: torch.Tensor      # (n_walls, 2)

    obstacle_center: torch.Tensor    # (E, n_obstacles, 2)
    obstacle_halfsize: torch.Tensor  # (n_obstacles, 2)
    obstacle_active: torch.Tensor    # (E, n_obstacles), bool