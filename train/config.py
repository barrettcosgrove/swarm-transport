from dataclasses import dataclass
import torch

@dataclass
class Config:
    # environment
    device: str = "cpu"
    num_envs: int = 64
    n_agents: int = 5
    # tools/calibrate.py recommends 486 for the DESIGN 50-70% traversal target.
    # Held deliberately slack while the scripted controller is still marginal,
    # so failures read as "couldn't do it" rather than "ran out of clock".
    max_steps: int = 900
    dt: float = 0.05

    # sizes and masses
    agent_radius: float = 0.1
    agent_mass: float = 1.0
    predator_radius: float = 0.15
    predator_mass: float = 1.0
    payload_halfsize: torch.Tensor = torch.tensor([0.2, 0.2])
    payload_mass: float = 5.0

    # static geometry. Walls are four boxes 1.0 thick whose inner faces sit at
    # +/- 7.5, clearing the 5.4 goal ring plus 0.6 payload jitter.
    wall_center: torch.Tensor = torch.tensor([[8.0, 0.0], [-8.0, 0.0], [0.0, 8.0], [0.0, -8.0]])
    wall_halfsize: torch.Tensor = torch.tensor([[0.5, 8.5], [0.5, 8.5], [8.5, 0.5], [8.5, 0.5]])

    # physics.step has no notion of obstacle_active, so a disabled obstacle has
    # to be relocated out of reach rather than masked off.
    #
    # Radius 4.3, spanning 3.8-4.8 along their axis. The inner edge clears the
    # 3.6 predator spawn ring, which matters because the predator takes a
    # uniformly random angle and would otherwise sometimes spawn inside a box.
    # The outer edge now reaches into the success region (success_threshold
    # 0.75 puts its inner edge at 5.4 - 0.75 = 4.65), so some goal angles have
    # part of their success annulus behind an obstacle. Measured cost is about
    # 3 points of scripted win rate, and DESIGN.md wants obstacles on the
    # direct path, so this is left as-is rather than shrunk to fit.
    obstacle_center: torch.Tensor = torch.tensor([[4.3, 0.0], [-4.3, 0.0], [0.0, 4.3], [0.0, -4.3]])
    obstacle_halfsize: torch.Tensor = torch.full((4, 2), 0.5)
    obstacle_active: torch.Tensor = torch.ones(4, dtype=torch.bool)

    # physics constants
    body_stiffness: float = 200.0
    wall_stiffness: float = 1200.0
    obstacle_stiffness: float = 200.0
    payload_stiffness: float = 250.0
    agent_drag_coef: float = 0.25
    predator_drag_coef: float = 0.25
    payload_drag_coef: float = 0.1
    agent_max_thrust: float = 5.0
    # a hard velocity cap, applied in physics.integrate. Without it the real
    # top speed is thrust/drag = 14, four times the nominal value this used to
    # carry, and the predator covers 0.7 units per step against a 0.25 contact
    # distance -- deep enough to one-shot an agent, or skip clean over one.
    predator_max_speed: float = 6.0
    predator_max_thrust: float = 3.5
    # how close an agent has to be to take damage. Larger than the bodies
    # actually touching (agent_radius + predator_radius = 0.25) so contact is
    # not missed when a fast agent steps clean past the predator.
    predator_capture_radius: float = 0.4
    predator_cooldown_duration: float = 10.0
    predator_cooldown_speed_factor: float = 0.5
    predator_weight: float = 1.0       # w in the guard formula
    # steps the predator sticks with a target before reconsidering. 20 is one
    # second at dt 0.05 -- long enough that fleeing actually shakes it.
    predator_target_commit_steps: float = 20.0
    ou_decay: float = 0.95
    ou_sigma: float = 0.1

    # spawn geometry (to be calibrated in Phase 2)
    payload_jitter_radius: float = 0.6
    r_agent_min: float = 1.2
    r_agent_max: float = 2.4
    predator_spawn_radius: float = 3.6
    goal_radius: float = 5.4
    goal_angle_jitter: float = 0.3

    # rewards
    progress_coef: float = 10.0
    time_penalty_coef: float = 0.3
    # payload-center to goal-center. The payload's half-diagonal is 0.283, so
    # at the old 0.3 it had to be almost exactly centered on the goal; 0.75
    # only asks it to overlap.
    success_threshold: float = 0.75
    success_reward: float = 100.0
    proximity_coef_start: float = 5.0
    proximity_anneal_fraction: float = 0.3
    health_loss_coef: float = 0.8
    # flat drain per damage event. With the cooldown gating events to one per
    # predator_cooldown_duration steps, max_health / this = contacts survived.
    health_loss_per_step: float = 10.0
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
    
    # seed
    seed: int = 0
    
    # scripted policy
    scripted_push_threshold: float = 0.3
    scripted_avoid_margin: float = 0.25
    scripted_avoid_gain: float = 2.0
    # 3.0 is roughly four steps of warning at the predator's closing speed --
    # a 1.5 radius reacts about two steps out, which is already too late.
    scripted_evade_radius: float = 3.0
    scripted_evade_gain: float = 3.0
    scripted_evade_tangent: float = 0.6

    @property
    def obs_dim(self) -> int:
        """Width of one agent's observation vector, from scenario.observe.

        Derived rather than stored: it depends on n_agents, and a stale
        constant here would only surface as a shape mismatch at the policy's
        first forward pass.

        own pos/vel (4), goal offset (2), payload offset/rel vel (4),
        predator offset/rel vel (4), time remaining (1), shared health (1),
        plus offset + rel vel for each teammate (4 each).
        """
        return 16 + 4 * (self.n_agents - 1)