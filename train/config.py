from dataclasses import dataclass
import torch

@dataclass
class Config:
    # environment
    device: str = "cpu"
    num_envs: int = 64
    n_agents: int = 3
    max_steps: int = 500
    dt: float = 0.05

    # sizes and masses
    agent_radius: float = 0.1
    agent_mass: float = 1.0
    predator_radius: float = 0.15
    predator_mass: float = 1.0
    payload_halfsize: torch.Tensor = torch.tensor([0.2, 0.2])
    payload_mass: float = 5.0

    # static geometry. Walls are four boxes 0.5 thick whose inner faces sit at
    # +/- 2.5, clearing the 1.8 goal ring plus 0.2 payload jitter.
    wall_center: torch.Tensor = torch.tensor([[3.0, 0.0], [-3.0, 0.0], [0.0, 3.0], [0.0, -3.0]])
    wall_halfsize: torch.Tensor = torch.tensor([[0.5, 3.5], [0.5, 3.5], [3.5, 0.5], [3.5, 0.5]])

    # physics.step has no notion of obstacle_active, so a disabled obstacle has
    # to be relocated out of reach rather than masked off.
    obstacle_center: torch.Tensor = torch.full((1, 2), 1e6)
    obstacle_halfsize: torch.Tensor = torch.full((1, 2), 0.2)
    obstacle_active: torch.Tensor = torch.zeros(1, dtype=torch.bool)

    # physics constants
    body_stiffness: float = 200.0
    wall_stiffness: float = 1200.0
    obstacle_stiffness: float = 200.0
    payload_stiffness: float = 250.0
    agent_drag_coef: float = 0.25
    predator_drag_coef: float = 0.25
    payload_drag_coef: float = 0.1
    agent_max_thrust: float = 5.0
    predator_max_speed: float = 3.5
    predator_max_thrust: float = 3.5
    predator_stiffness: float = 200.0
    predator_cooldown_duration: float = 10.0
    predator_cooldown_speed_factor: float = 0.5
    predator_weight: float = 1.0       # w in the guard formula
    ou_decay: float = 0.95
    ou_sigma: float = 0.1

    # spawn geometry (to be calibrated in Phase 2)
    payload_jitter_radius: float = 0.2
    r_agent_min: float = 0.4
    r_agent_max: float = 0.8
    predator_spawn_radius: float = 1.2
    goal_radius: float = 1.8
    goal_angle_jitter: float = 0.3

    # rewards
    progress_coef: float = 10.0
    time_penalty_coef: float = 0.3
    success_threshold: float = 0.3
    success_reward: float = 100.0
    proximity_coef_start: float = 5.0
    proximity_anneal_fraction: float = 0.3
    health_loss_coef: float = 0.8
    health_loss_per_step: float = 5.0
    captured_reward: float = -100.0
    collision_coef: float = 0.1
    max_health: float = 100.0

    # training
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    ent_coefficient: float = 0.001
    rollout_steps: int = 128
    update_epochs: int = 10
    minibatch_size: int = 4096
    num_iterations: int = 250
    lr: float = 3e-4

    # network
    hidden_dim: int = 128
    # own pos/vel, goal offset, payload offset/rel vel, predator offset/rel vel,
    # time remaining, shared health, plus offset + rel vel per teammate
    obs_dim: int = 24
    
    # seed
    seed: int = 0
    
    # scripted policy
    scripted_push_threshold: float = 0.3
    scripted_avoid_margin: float = 0.25
    scripted_avoid_gain: float = 2.0