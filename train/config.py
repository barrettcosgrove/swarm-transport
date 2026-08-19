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

    # static geometry. Walls are four boxes 3.0 thick whose inner faces sit at
    # +/- 8.5. A goal center can sit 6.3 + 0.6 payload jitter = 6.9 from the
    # origin, and its success circle reaches 7.65, so the old 7.5 faces would
    # have clipped it.
    #
    # The thickness is a containment guarantee, not decoration. A penalty force
    # resolves a deep overlap along the shortest way out, so once a body's
    # center passes a wall's midline the nearest face is the OUTER one and the
    # wall starts pushing it away from the arena -- at 1200 stiffness, hard
    # enough that it never comes back. Walls were 1.0 thick, putting that
    # midline at 9.0, only 0.6 beyond first contact at 8.4; agents cleared it
    # in a single step and 21% of them left the arena for good. At half-thickness
    # 1.5 the midline sits at 10.0, 1.6 past first contact, against a capped
    # step of agent_max_speed * dt = 1.0. Keep that margin if either is retuned;
    # tests/test_physics.py asserts it.
    #
    # The long halfsize is each wall's outer face, so the four boxes still
    # overlap at the corners and seal them.
    wall_center: torch.Tensor = torch.tensor([[10.0, 0.0], [-10.0, 0.0], [0.0, 10.0], [0.0, -10.0]])
    wall_halfsize: torch.Tensor = torch.tensor([[1.5, 11.5], [1.5, 11.5], [11.5, 1.5], [11.5, 1.5]])

    # physics.step has no notion of obstacle_active, so a disabled obstacle has
    # to be relocated out of reach rather than masked off.
    #
    # No centers here: scenario.reset samples them per episode from the ring
    # constants under "spawn geometry". A fixed layout put a box outer face at
    # 4.8 against a success region starting at 4.65, so goal angles near an
    # obstacle axis had the goal marker clipping into it.
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
    # thrust/drag, the fastest an agent can drive itself. As a cap it does not
    # restrain the engine at all; it stops CONTACT forces from launching an
    # agent faster than the engine ever could. Without it a wall rebound
    # reached 59 units/s -- 3 units per step against a 3.0 thick wall -- which
    # is how agents ended up outside the arena. Measured in-arena speeds are
    # p90 10.1, p99 17.2, p99.9 22.7, so this trims the contact-force tail and
    # leaves ordinary motion untouched. A tighter 10.0 would have clipped 10%
    # of normal movement.
    agent_max_speed: float = 20.0
    # a hard velocity cap, applied in physics.integrate. Without it the real
    # top speed is thrust/drag = 14, four times the nominal value this used to
    # carry, and the predator covers 0.7 units per step against a 0.25 contact
    # distance -- deep enough to one-shot an agent, or skip clean over one.
    #
    # 3.5 rather than 6.0 so a thrusting agent can break contact during the
    # cooldown window. At 6.0 it closed 0.3 units/step and ran down a clump
    # that had not yet learned to thrust; fleeing was not a behaviour the
    # policy could express, only a penalty it absorbed.
    predator_max_speed: float = 3.5
    predator_max_thrust: float = 3.5
    # how close an agent has to be to take damage. Larger than the bodies
    # actually touching (agent_radius + predator_radius = 0.25) so contact is
    # not missed when a fast agent steps clean past the predator.
    predator_capture_radius: float = 0.4
    # Sets the survival budget together with max_health, and that budget has to
    # outlast a traversal or the task is unwinnable however well the policy
    # plays: max_health / health_loss_per_step events, one per cooldown, is
    # 15 * 40 = 600 steps against ~340 for a push to the goal. At 25 it was 260
    # against 340, so every episode ended in capture before arrival and there
    # was nothing for preserving health to buy. tests/test_reward_invariants.py
    # asserts the margin.
    #
    # Also the window an agent has to break contact after being hit. Too short
    # and evasion is not a behaviour the policy can express, only a penalty it
    # absorbs.
    predator_cooldown_duration: float = 40.0
    predator_cooldown_speed_factor: float = 0.5
    predator_weight: float = 0.5       # w in the guard formula
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
    # 6.3 rather than 5.4 so the obstacle band can end at 4.3 and still leave
    # the goal's success circle clear. Below about 6.1 there is no radius on
    # the payload->goal ray that clears both the predator and the goal, which
    # would have forced obstacles off the direct path entirely.
    goal_radius: float = 6.3
    goal_angle_jitter: float = 0.3

    # Obstacle ring, sampled per episode by scenario.reset. Every bound is
    # derived from a box's circumradius, 0.5 * sqrt(2) = 0.707 -- the
    # conservative circular bound on a halfsize-0.5 box, which makes each
    # clearance a single distance comparison regardless of relative angle.
    #
    #   band_min 3.5    clears the agent annulus. r_agent_max 2.4 +
    #                   agent_radius 0.1 + 0.707 = 3.21, rounded up.
    #   band_max 4.3    clears the goal. The payload must be able to rest
    #                   anywhere inside the success circle, so an obstacle
    #                   needs success_threshold 0.75 + payload circumradius
    #                   0.283 + 0.707 = 1.74 of room, and 6.3 - 4.3 = 2.0.
    #   corridor 0.35   clears the predator, which sits at radius 3.6 on
    #                   theta. Closest approach across the band is
    #                   3.6 * sin(0.35) = 1.24 against 0.857 required
    #                   (predator_radius 0.15 + 0.707). Deliberately no wider:
    #                   goal_angle_jitter is 0.3, so the payload's line to the
    #                   goal still runs straight into a box some episodes,
    #                   which is the point of having obstacles at all.
    #   min_gap 0.75    DESIGN.md's clearance rule. Two obstacles this far
    #                   apart at r >= 3.5 are at least 2 * 3.5 * sin(0.375) =
    #                   2.56 apart, leaving 1.15 surface-to-surface -- wider
    #                   than the payload (0.4) plus two agent diameters. No
    #                   arrangement spaced this way can seal off a region.
    #   n_sectors 6     the arc left after the wedge is 2*pi - 0.7 = 5.58, so
    #                   at most 5.58 / 0.75 = 7 sectors can hold the gap. Six
    #                   slots for four obstacles is what varies the layout:
    #                   which sectors are occupied changes per episode, so
    #                   some crowd one side and leave the other open.
    obstacle_band_min: float = 3.5
    obstacle_band_max: float = 4.3
    obstacle_corridor_half_angle: float = 0.35
    obstacle_min_angular_gap: float = 0.75
    obstacle_n_sectors: int = 6

    # rewards
    progress_coef: float = 50
    # Bounded by DESIGN.md's inequality max_steps * |R_time| < |R_capture|:
    # if the clock costs more than dying, a policy that cannot win is right to
    # end the episode early, and it will. The bound is 250 / 900 = 0.28 at these
    # values, and 0.10 leaves the same kind of headroom DESIGN's original 0.3
    # against 100 did. It read 0.22 against a 900-step, -150 board, so
    # 198 > 150 and suicide was strictly optimal -- the derivation in DESIGN.md
    # was still written against the old 250-step, -100 numbers and had silently
    # stopped applying. tests/test_reward_invariants.py now asserts it.
    time_penalty_coef: float = 0.10
    # payload-center to goal-center. The payload's half-diagonal is 0.283, so
    # at the old 0.3 it had to be almost exactly centered on the goal; 0.75
    # only asks it to overlap.
    success_threshold: float = 0.75
    success_reward: float = 450.0
    proximity_coef_start: float = 8.0
    # the fraction of training over which the coefficient decays linearly to
    # zero: coef = start * (1 - min(progress / fraction, 1)). Anything above 1.0
    # never finishes, leaving a permanent floor of start * (1 - 1/fraction) --
    # at the 5.0 this used to hold, 8.0 only ever reached 6.4, so the crutch
    # never came off and proximity and alignment went on pulling all n_agents
    # onto the same single point behind the payload for the whole run.
    proximity_anneal_fraction: float = 0.6
    # Per-unit distance closed toward the standoff point behind the payload.
    # Same shape as proximity, different target: payload center vs the place a
    # push actually helps. Hovering pays 0. Does not anneal -- pushing is the
    # task, same as progress. 8.0 matches proximity_coef_start, so a 0.05-unit
    # close-in is +0.4/step, 4x the time penalty.
    push_coef: float = 8.0
    # Staging offset behind the payload, along the payload->goal line. Default is
    # payload_halfsize.max() + agent_radius + 0.15, the same number the scripted
    # controller already used inline.
    push_standoff: float = 0.45
    # Kept so the anneal invariant still has something to check, but no longer
    # consumed by compute_reward: a telescoping cosine summed to ~0 over a run
    # and could not teach a standing push. That job is push_coef above.
    alignment_coef_start: float = 8.0
    alignment_anneal_fraction: float = 0.6
    health_loss_coef: float = 1.0
    # How much of a damage event is billed to the agents actually inside
    # predator_capture_radius, with the remainder still shared by the team.
    #
    # A wholly shared penalty is why no agent ever learned to evade. update_health
    # drains on any() over agents, so one agent's retreat moved its own reward by
    # nothing at all, while proximity and alignment were private and did respond.
    # The gradient said "chase the shaping, treat health as weather", and the
    # policy did: measured cos(action, toward predator) was +0.087, drifting
    # toward the thing killing it.
    #
    # The private share is flat, NOT split among the agents in range. Splitting
    # would keep the team total constant but make an agent's own bill shrink as
    # teammates joined it in the danger zone, paying for exactly the bunching
    # that is already a problem here. Flat means being in range always costs the
    # same, and the team total rises with crowding.
    health_loss_blame_fraction: float = 0.7
    # flat drain per damage event. With the cooldown gating events to one per
    # predator_cooldown_duration steps, max_health / this = contacts survived.
    health_loss_per_step: float = 10.0
    # Deep enough that dying beats neither winning nor waiting: against the
    # inequality above it has to exceed max_steps * time_penalty_coef = 90, and
    # against a full health drain (-150) it is the worse outcome, which is the
    # ordering DESIGN.md section 11 asks for.
    captured_reward: float = -250.0
    collision_coef: float = 0.002
    # A private "you are about to be hit" penalty, and the one signal an
    # individual agent can act on alone. DESIGN.md omits predator-distance
    # shaping because a permanent version is a permanent reason to avoid the
    # payload the predator guards; that objection is why this is exactly zero
    # beyond the radius rather than a global gradient. 1.0 is 2.5x the capture
    # radius, far tighter than the scripted controller's evade radius of 3.0, so
    # it fires only when contact is imminent and leaves ordinary pushing alone.
    predator_danger_radius: float = 1.0
    # per-step cost at zero distance, falling quadratically to zero at the
    # radius, and only while the predator can actually deal damage. Matched
    # to the time penalty so it is a reason to step away, not the dominant
    # private term that made approaching the guarded payload net-negative.
    threat_coef: float = 0.10
    # 15 events at 40 steps each = 600 steps of life, against ~340 for a
    # traversal. At 100 the budget was 260 and arrival was impossible.
    max_health: float = 150.0

    # observation scaling. Positions / 8.5 maps the arena inner faces to
    # [-1, 1]; velocities / 5.0 maps ordinary motion similarly. Offsets are
    # computed in world units first, then divided, so a relative vector
    # cannot pick up a scale from only one of its two ends.
    obs_pos_scale: float = 8.5   # arena half-extent
    obs_vel_scale: float = 5.0   # typical agent speed

    # training
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    ent_coefficient: float = 0.001
    rollout_steps: int = 128
    update_epochs: int = 10
    # T=128 * E=64 * N=5 = 40,960 samples an iteration, so this is 10
    # minibatches per epoch. The MAPPO paper finds heavy minibatch splitting
    # costs performance and stability; 20480 (2 minibatches) is the obvious
    # one-variable experiment if value loss will not settle.
    minibatch_size: int = 4096
    num_iterations: int = 400
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    # Off for the first run so a failure has one candidate cause. The +/-100
    # terminal spikes next to -0.1--0.8 per-step terms are exactly the regime
    # value normalization exists for, so this is the first thing to flip.
    use_value_norm: bool = True

    # network
    hidden_dim: int = 128

    # checkpointing and logging
    checkpoint_interval: int = 25
    checkpoint_dir: str = "train/checkpoints"
    log_path: str = "outputs/training_history.json"
    
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
    def n_obstacles(self) -> int:
        """How many obstacles a reset places, read off obstacle_halfsize.

        Derived rather than stored so the count and the sizes cannot drift
        apart -- a mismatch would only surface as a broadcast error deep in
        physics.step.
        """
        return self.obstacle_halfsize.shape[0]

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

    @property
    def state_dim(self) -> int:
        """Width of the centralized critic's input: every agent's observation
        concatenated, plus a one-hot agent id.

        The one-hot is what distinguishes the n_agents rows of an environment's
        critic input -- without it a deterministic network must return the same
        value for every agent, which is wrong, since compute_reward gives each
        agent a private proximity and collision term.

        Grows linearly with n_agents while the sample budget does not. If value
        loss will not settle, this is a candidate cause, and the fix is a
        permutation-invariant encoder rather than raw concatenation.
        """
        return self.n_agents * self.obs_dim + self.n_agents