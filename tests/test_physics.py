"""
tests/test_physics.py

Two categories of test:
  1. Structural tests (settling, batch independence, determinism, no tunneling) --
     these check that the SIMULATION MECHANICS are sound, independent of any
     specific numbers.
  2. Hand-derivation regression tests -- these lock in the exact worked examples
     derived by hand during development, so a future refactor can't silently
     break something that is correct right now.

Run with: pytest tests/test_physics.py -v
"""
import dataclasses
import torch
import pytest

from env.world import WorldState
from env.physics import (
    integrate,
    circle_circle_forces,
    circle_box_static_forces,
    circle_box_dynamic_forces,
    box_box_forces,
    step,
)
from train.config import Config


# ---------------------------------------------------------------- fixtures / helpers

STIFFNESS = dict(body_stiffness=100.0, wall_stiffness=200.0,
                  obstacle_stiffness=100.0, payload_stiffness=150.0)
DRAG = dict(agent_drag_coef=0.25, predator_drag_coef=0.25, payload_drag_coef=0.1)
THRUST = dict(agent_max_thrust=5.0, predator_max_thrust=3.5)


def make_test_world(num_envs, seed=0, n_agents=2):
    """
    A minimal, internally-consistent WorldState for physics testing.
    Does NOT need to match the real spawn rules in scenario.py -- it only
    needs correct shapes and non-degenerate starting positions. Obstacles
    are present but parked far away (inactive), matching the technique from
    DESIGN.md: relocate rather than zero-size, so they contribute exactly
    zero force without physics.py needing to know "active" is a concept.
    """
    g = torch.Generator().manual_seed(seed)
    E = num_envs
    agent_pos = (torch.rand(E, n_agents, 2, generator=g) - 0.5) * 2.0
    agent_vel = torch.zeros(E, n_agents, 2)
    predator_pos = torch.tensor([3.0, 3.0]).expand(E, 2).clone()
    predator_vel = torch.zeros(E, 2)
    payload_pos = torch.zeros(E, 2)
    payload_vel = torch.zeros(E, 2)
    wall_center = torch.tensor([[10.0, 0.0], [-10.0, 0.0], [0.0, 10.0], [0.0, -10.0]])
    wall_halfsize = torch.tensor([[0.5, 10.0], [0.5, 10.0], [10.0, 0.5], [10.0, 0.5]])
    obstacle_center = torch.full((E, 1, 2), 1e6)     # relocated far away -> zero force
    obstacle_halfsize = torch.ones(1, 2) * 0.2
    obstacle_active = torch.zeros(E, 1, dtype=torch.bool)

    return WorldState(
        agent_pos=agent_pos, agent_vel=agent_vel, agent_radius=0.1, agent_mass=1.0,
        predator_pos=predator_pos, predator_vel=predator_vel, predator_radius=0.15, predator_mass=1.0,
        payload_pos=payload_pos, payload_vel=payload_vel,
        payload_halfsize=torch.tensor([0.3, 0.3]), payload_mass=5.0,
        wall_center=wall_center, wall_halfsize=wall_halfsize,
        obstacle_center=obstacle_center, obstacle_halfsize=obstacle_halfsize,
        obstacle_active=obstacle_active,
    )


def slice_env(world, idx):
    """Pull environment `idx` out of a batched WorldState as its own num_envs=1 WorldState."""
    E = world.agent_pos.shape[0]
    kwargs = {}
    for f in dataclasses.fields(world):
        val = getattr(world, f.name)
        if isinstance(val, torch.Tensor) and val.dim() > 0 and val.shape[0] == E:
            kwargs[f.name] = val[idx:idx + 1].clone()
        else:
            kwargs[f.name] = val
    return WorldState(**kwargs)


def make_agent_actions(E, n_agents, n_steps):
    """
    Deterministic, closed-form actions -- NOT randomly sampled. This is
    deliberate: a random tensor generated once for the whole trajectory
    consumes a different amount of the RNG stream depending on E, so
    environment 0's slice would differ between an E=64 run and an E=1 run
    even though nothing in the physics is actually wrong. A closed-form
    function of (t, env_index, agent_index) sidesteps that entirely --
    env 0's actions are identical no matter how many other environments
    exist, because the formula never looks at them.
    """
    t = torch.arange(n_steps).float().view(-1, 1, 1, 1)
    e = torch.arange(E).float().view(1, -1, 1, 1)
    n = torch.arange(n_agents).float().view(1, 1, -1, 1)
    x = torch.sin(0.3 * t + 0.1 * e + 0.5 * n)
    y = torch.cos(0.3 * t + 0.1 * e + 0.5 * n)
    return torch.stack([x, y], dim=-1).squeeze(-2)


def make_predator_actions(E, n_steps):
    t = torch.arange(n_steps).float().view(-1, 1, 1)
    e = torch.arange(E).float().view(1, -1, 1)
    x = torch.sin(0.2 * t + 0.15 * e)
    y = torch.cos(0.2 * t + 0.15 * e)
    return torch.stack([x, y], dim=-1).squeeze(-2)


def run_trajectory(world, n_agents, n_steps, dt):
    E = world.agent_pos.shape[0]
    actions = make_agent_actions(E, n_agents, n_steps)
    pred_actions = make_predator_actions(E, n_steps)
    for t in range(n_steps):
        world = step(world, actions[t], pred_actions[t], dt, **THRUST, **DRAG, **STIFFNESS)
    return world


# ---------------------------------------------------------------- structural tests

def test_settling():
    """No thrust, drag applied explicitly each step -> velocity decays smoothly to zero.

    integrate() itself knows nothing about drag -- drag is a force computed by
    the caller and summed in, same as every other force. This test exercises
    that composition directly, in isolation from every other force source.
    """
    pos = torch.zeros(1, 1, 2)
    vel = torch.tensor([[[2.0, 0.0]]])
    drag_coef = 1.0
    speeds = []
    for _ in range(500):
        drag_force = -drag_coef * vel
        pos, vel = integrate(pos, vel, drag_force, mass=1.0, dt=0.02)
        speeds.append(vel.norm().item())

    assert speeds[-1] < 1e-3
    # monotonic decay -- no oscillation, no energy gain
    assert all(speeds[i] >= speeds[i + 1] - 1e-6 for i in range(len(speeds) - 1))


def test_batch_independence():
    """Environment i inside a batch of 64 must match the same environment run alone.

    This is the test that catches a reduction happening along the wrong axis --
    exactly the bug class found repeatedly while building physics.py. The fix
    of slicing env 0 straight out of the batched WorldState (rather than
    separately constructing a num_envs=1 world) guarantees identical starting
    conditions regardless of any RNG-stream subtlety.
    """
    world64 = make_test_world(num_envs=64, seed=0)
    world1 = slice_env(world64, 0)

    result64 = run_trajectory(world64, n_agents=2, n_steps=50, dt=0.05)
    result1 = run_trajectory(world1, n_agents=2, n_steps=50, dt=0.05)

    assert torch.allclose(result64.agent_pos[0], result1.agent_pos[0], atol=1e-4)
    assert torch.allclose(result64.predator_pos[0], result1.predator_pos[0], atol=1e-4)
    assert torch.allclose(result64.payload_pos[0], result1.payload_pos[0], atol=1e-4)


def test_determinism():
    """Same seed, same actions -> identical trajectory, every run, exactly."""
    world_a = make_test_world(num_envs=4, seed=0)
    world_b = make_test_world(num_envs=4, seed=0)

    result_a = run_trajectory(world_a, n_agents=2, n_steps=30, dt=0.05)
    result_b = run_trajectory(world_b, n_agents=2, n_steps=30, dt=0.05)

    assert torch.equal(result_a.agent_pos, result_b.agent_pos)
    assert torch.equal(result_a.predator_pos, result_b.predator_pos)
    assert torch.equal(result_a.payload_pos, result_b.payload_pos)


def test_no_tunneling():
    """A body approaching a wall at a realistic speed must not pass through it.

    Uses parameters representative of the actual project (moderate speed,
    ordinary dt, a wall with real thickness), not an artificially extreme
    case. Penetration into the wall is expected and correct for a
    penalty-force system -- the assertion is that the body stays on the
    near side of the wall's CENTER, not that penetration is exactly zero.
    """
    wall_center = torch.tensor([[5.0, 0.0]])
    wall_halfsize = torch.tensor([[0.5, 10.0]])   # wall spans x: [4.5, 5.5]
    pos = torch.tensor([[[4.0, 0.0]]])
    vel = torch.tensor([[[5.0, 0.0]]])
    dt = 0.05
    max_x = pos[0, 0, 0].item()

    for _ in range(60):
        force = circle_box_static_forces(pos, 0.1, wall_center, wall_halfsize, 200.0)
        drag = -0.3 * vel
        pos, vel = integrate(pos, vel, force + drag, mass=1.0, dt=dt)
        max_x = max(max_x, pos[0, 0, 0].item())

    assert max_x < wall_center[0, 0].item()


# ---------------------------------------------------------------- containment at the real constants
#
# test_no_tunneling above uses gentle stand-in numbers (stiffness 200, 0.25
# units per step). The real arena runs at stiffness 1200 and up to 1.0 units
# per step, a regime where the penalty force's deep-overlap branch turns
# around and starts ejecting bodies. These tests pin containment and the
# margin that buys it at the constants the project actually ships.

def wall_geometry(config, wall_index=0):
    """Inner face, midline and thin half-thickness of one wall.

    A wall is thin on one axis and long on the other, the long axis only
    being there to seal the corners, so the thin axis is the one carrying
    the face a body runs into.
    """
    center, halfsize = config.wall_center[wall_index], config.wall_halfsize[wall_index]
    thin_axis = int(torch.argmin(halfsize))
    midline = abs(float(center[thin_axis]))
    return midline - float(halfsize[thin_axis]), midline, float(halfsize[thin_axis])


def make_arena_world(config, directions):
    """One agent per environment at the origin, inside the real walls.

    Everything else is parked at 1e6 so the only forces in play are thrust,
    drag and the wall -- the same relocate-rather-than-mask technique the
    obstacles already use.
    """
    E = directions.shape[0]
    far = torch.full((E, 2), 1e6)
    return WorldState(
        agent_pos=torch.zeros(E, 1, 2), agent_vel=torch.zeros(E, 1, 2),
        agent_radius=config.agent_radius, agent_mass=config.agent_mass,
        predator_pos=far.clone(), predator_vel=torch.zeros(E, 2),
        predator_radius=config.predator_radius, predator_mass=config.predator_mass,
        payload_pos=far.clone(), payload_vel=torch.zeros(E, 2),
        payload_halfsize=config.payload_halfsize, payload_mass=config.payload_mass,
        wall_center=config.wall_center, wall_halfsize=config.wall_halfsize,
        obstacle_center=torch.full((E, 1, 2), 1e6),
        obstacle_halfsize=torch.ones(1, 2) * 0.5,
        obstacle_active=torch.zeros(E, 1, dtype=torch.bool),
    )


EIGHT_DIRECTIONS = torch.tensor([
    [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
    [1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0],
])


def test_sustained_thrust_at_the_real_constants_cannot_cross_a_wall():
    """The case the arena actually has to survive: full thrust straight into
    a wall, held for an entire episode, at stiffness 1200 and the real cap.

    Crossing a wall's midline is a point of no return -- past it the nearest
    way out of the box is the outer face, so the wall starts pushing the
    body away from the arena and never stops. So the assertion is on the
    midline, not on the face: penetration into a wall is normal for a
    penalty-force system, crossing its center is terminal.

    Measured reach is 8.97, which the current midline at 10.0 clears
    comfortably. Against the 1.0 thick walls this replaced, whose midline sat
    at 9.0, plain thrust and nothing else came within 0.03 of that same
    threshold -- the old geometry had no margin at all.
    """
    config = Config()
    world = make_arena_world(config, EIGHT_DIRECTIONS)
    # diagonals normalized, so a corner run is not fed 1.41x the thrust an
    # axis run gets
    actions = (EIGHT_DIRECTIONS / EIGHT_DIRECTIONS.norm(dim=-1, keepdim=True)).unsqueeze(1)
    predator_actions = torch.zeros(EIGHT_DIRECTIONS.shape[0], 2)
    _, midline, _ = wall_geometry(config)

    max_reach = 0.0
    for _ in range(600):
        world = step(
            world, actions, predator_actions, config.dt,
            config.agent_max_thrust, config.predator_max_thrust,
            config.agent_drag_coef, config.predator_drag_coef, config.payload_drag_coef,
            config.body_stiffness, config.wall_stiffness,
            config.obstacle_stiffness, config.payload_stiffness,
            predator_max_speed=config.predator_max_speed,
            agent_max_speed=config.agent_max_speed,
        )
        max_reach = max(max_reach, float(world.agent_pos.abs().max()))

    assert max_reach < midline, \
        f"an agent reached {max_reach:.2f}, past the wall midline at {midline}"


@pytest.mark.parametrize("agent_max_speed, contained", [(Config().agent_max_speed, True), (None, False)])
def test_the_cap_contains_an_agent_that_is_already_moving_too_fast(agent_max_speed, contained):
    """The case the cap is actually for.

    Thrust alone tops out at 18.8 in the test above and never threatens the
    wall, so that run would pass with or without a cap. What the cap defends
    against is a body arriving at the wall carrying a speed that no thrust
    produced -- an agent squeezed between the payload and a wall, or bounced
    off one. Rollouts before this fix measured 59 units/s that way, so the
    launch here is 60.

    Parametrized against no cap to keep the guarantee attributable: the same
    launch tunnels straight through and ends up hundreds of units outside.
    """
    config = Config()
    world = make_arena_world(config, EIGHT_DIRECTIONS)
    directions = EIGHT_DIRECTIONS / EIGHT_DIRECTIONS.norm(dim=-1, keepdim=True)
    world = dataclasses.replace(world, agent_vel=(directions * 60.0).unsqueeze(1))
    predator_actions = torch.zeros(EIGHT_DIRECTIONS.shape[0], 2)
    _, midline, _ = wall_geometry(config)

    max_reach = 0.0
    for _ in range(400):
        world = step(
            world, directions.unsqueeze(1), predator_actions, config.dt,
            config.agent_max_thrust, config.predator_max_thrust,
            config.agent_drag_coef, config.predator_drag_coef, config.payload_drag_coef,
            config.body_stiffness, config.wall_stiffness,
            config.obstacle_stiffness, config.payload_stiffness,
            predator_max_speed=config.predator_max_speed,
            agent_max_speed=agent_max_speed,
        )
        max_reach = max(max_reach, float(world.agent_pos.abs().max()))

    assert (max_reach < midline) == contained, \
        f"reached {max_reach:.2f} against a midline at {midline}"


def test_the_wall_pushes_inward_up_to_its_midline():
    """Where the restoring force is trustworthy, and where it stops being so.

    From first contact to the midline the wall pushes a body back into the
    arena. Past the midline the minimum-translation resolution picks the
    outer face instead and the force flips outward -- inherent to a penalty
    method, which cannot tell which side a body entered from. The fix is not
    to make that region behave, it is to make it unreachable, which is what
    the wall thickness buys and what the margin assertions below pin.
    """
    config = Config()
    inner_face, midline, _ = wall_geometry(config)
    center = config.wall_center[0:1]
    halfsize = config.wall_halfsize[0:1]

    contact = inner_face - config.agent_radius
    inward_band = torch.linspace(contact + 1e-3, midline, 200).view(1, -1, 1)
    pos = torch.cat([inward_band, torch.zeros_like(inward_band)], dim=-1)

    force = circle_box_static_forces(pos, config.agent_radius, center, halfsize, config.wall_stiffness)
    assert (force[..., 0] < 0).all(), "the wall failed to push a contacting body back inward"

    # and the documented reversal past it, kept as a live fact rather than a
    # comment so that a future collision rewrite that removes it shows up here
    beyond = torch.tensor([[[midline + 0.1, 0.0]]])
    force_beyond = circle_box_static_forces(beyond, config.agent_radius, center, halfsize, config.wall_stiffness)
    assert float(force_beyond[0, 0, 0]) > 0, \
        "past the midline the force no longer reverses -- the margin guards below may be obsolete"


def test_the_geometry_leaves_room_for_a_full_speed_step():
    """The two relationships the containment above rests on. They are cheap
    to state and easy to break by retuning any one constant in isolation, so
    they fail loudly here rather than silently letting agents out.
    """
    config = Config()
    inner_face, midline, thin_halfsize = wall_geometry(config)

    # one capped step must not carry a body from first contact past the midline
    travel_budget = thin_halfsize + config.agent_radius
    assert config.agent_max_speed * config.dt < travel_budget, (
        f"a step of {config.agent_max_speed * config.dt} clears the "
        f"{travel_budget} from first contact to the wall midline"
    )

    # the cap must not exceed what thrust alone can reach, or contact forces
    # get to launch agents faster than they could ever drive themselves
    assert config.agent_max_speed <= config.agent_max_thrust / config.agent_drag_coef

    # every wall is the same thickness, so checking one is checking all four
    for index in range(config.wall_center.shape[0]):
        assert wall_geometry(config, index) == (inner_face, midline, thin_halfsize)


# ---------------------------------------------------------------- hand-derivation regression tests

def test_integrate_matches_hand_derivation():
    """Constant thrust, mass=1, dt=0.1 -- traced by hand: velocity climbs by
    exactly 1 unit per step, position increments grow correspondingly."""
    pos = torch.zeros(1, 2)
    vel = torch.zeros(1, 2)
    force = torch.tensor([[10.0, 0.0]])

    pos, vel = integrate(pos, vel, force, mass=1.0, dt=0.1)
    assert torch.allclose(vel, torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(pos, torch.tensor([[0.1, 0.0]]))

    pos, vel = integrate(pos, vel, force, mass=1.0, dt=0.1)
    assert torch.allclose(vel, torch.tensor([[2.0, 0.0]]))
    assert torch.allclose(pos, torch.tensor([[0.3, 0.0]]))

    pos, vel = integrate(pos, vel, force, mass=1.0, dt=0.1)
    assert torch.allclose(vel, torch.tensor([[3.0, 0.0]]))
    assert torch.allclose(pos, torch.tensor([[0.6, 0.0]]))


def test_circle_circle_forces_matches_hand_derivation():
    """Two agents (radius 1) and a predator (radius 1.5), positions and
    penetrations traced by hand: agent 0 overlaps the predator by 1.5,
    agent 1 by 0.5, giving forces (-150,0)/(50,0) on the agents and
    (100,0) -- the combined reaction -- on the predator."""
    agent_pos = torch.tensor([[[0.0, 0.0], [3.0, 0.0]]])
    predator_pos = torch.tensor([[[1.0, 0.0]]])

    force_a, force_b = circle_circle_forces(agent_pos, 1.0, predator_pos, 1.5, 100.0)

    assert torch.allclose(force_a, torch.tensor([[[-150.0, 0.0], [50.0, 0.0]]]), atol=1e-3)
    assert torch.allclose(force_b, torch.tensor([[[100.0, 0.0]]]), atol=1e-3)


def test_circle_circle_forces_single_body_predator():
    """Same physics as above, but exercising the (E,2) single-body shape
    predator_pos actually has in WorldState -- not (E,1,2). This is the
    exact shape that silently broke before the squeeze/unsqueeze fix."""
    agent_pos = torch.tensor([[[0.0, 0.0], [3.0, 0.0]]])
    predator_pos = torch.tensor([[1.0, 0.0]])   # (E, 2), no body-count axis

    force_a, force_b = circle_circle_forces(agent_pos, 1.0, predator_pos, 1.5, 100.0)

    assert force_b.shape == (1, 2)   # must NOT have picked up a phantom axis
    assert torch.allclose(force_a, torch.tensor([[[-150.0, 0.0], [50.0, 0.0]]]), atol=1e-3)
    assert torch.allclose(force_b, torch.tensor([[100.0, 0.0]]), atol=1e-3)


def test_circle_box_static_forces_face_contact():
    """Box center (5,5) halfsize (2,3); circle at (7.5,6) r=1.
    closest point (7,6), distance 0.5, penetration 0.5 -> force (50,0)."""
    circle_pos = torch.tensor([[[7.5, 6.0]]])
    box_center = torch.tensor([[5.0, 5.0]])
    box_halfsize = torch.tensor([[2.0, 3.0]])

    force = circle_box_static_forces(circle_pos, 1.0, box_center, box_halfsize, 100.0)
    assert torch.allclose(force, torch.tensor([[[50.0, 0.0]]]), atol=1e-3)


def test_circle_box_static_forces_corner_contact():
    """Same box; circle at (8,9) r=1.5. Closest point is the corner (7,8),
    distance sqrt(2), penetration 1.5 - sqrt(2), pushed diagonally."""
    circle_pos = torch.tensor([[[8.0, 9.0]]])
    box_center = torch.tensor([[5.0, 5.0]])
    box_halfsize = torch.tensor([[2.0, 3.0]])

    force = circle_box_static_forces(circle_pos, 1.5, box_center, box_halfsize, 100.0)
    dist = 2.0 ** 0.5
    penetration = 1.5 - dist
    expected = 100.0 * penetration * (1 / dist)
    assert torch.allclose(force, torch.tensor([[[expected, expected]]]), atol=1e-3)


def test_circle_box_static_forces_interior_case():
    """Circle center INSIDE the box -- the case that silently returned zero
    force before the interior-branch fix. Box center (0,0) halfsize (2,2).
    Circle at (0.5,0.5) r=0.3: nearest exit is 1.5 along either axis (tie,
    resolves to x), penetration = 1.5 + 0.3 = 1.8 -> force (180, 0)."""
    circle_pos = torch.tensor([[[0.5, 0.5]]])
    box_center = torch.tensor([[0.0, 0.0]])
    box_halfsize = torch.tensor([[2.0, 2.0]])

    force = circle_box_static_forces(circle_pos, 0.3, box_center, box_halfsize, 100.0)
    assert torch.allclose(force, torch.tensor([[[180.0, 0.0]]]), atol=1e-3)
    assert force.abs().sum() > 0   # regression guard: this used to be exactly zero


def test_circle_box_static_forces_no_contact_is_zero():
    circle_pos = torch.tensor([[[100.0, 100.0]]])
    box_center = torch.tensor([[0.0, 0.0]])
    box_halfsize = torch.tensor([[2.0, 2.0]])

    force = circle_box_static_forces(circle_pos, 1.0, box_center, box_halfsize, 100.0)
    assert torch.allclose(force, torch.zeros_like(force))


def test_circle_box_dynamic_forces_matches_hand_derivation():
    """Same face-contact geometry as the static test, but against a dynamic
    box (the payload): the circle's force matches the static case exactly,
    and the box receives the equal-and-opposite reaction."""
    circle_pos = torch.tensor([[[7.5, 6.0]]])
    box_center = torch.tensor([[5.0, 5.0]])
    box_halfsize = torch.tensor([2.0, 3.0])

    force_a, force_b = circle_box_dynamic_forces(circle_pos, 1.0, box_center, box_halfsize, 100.0)
    assert torch.allclose(force_a, torch.tensor([[[50.0, 0.0]]]), atol=1e-3)
    assert torch.allclose(force_b, torch.tensor([[-50.0, 0.0]]), atol=1e-3)


def test_circle_box_dynamic_forces_single_body_predator():
    """Same shape guard as the circle-circle equivalent: predator_pos is
    (E,2), not (E,1,2). This is the exact call site that broke before the fix."""
    predator_pos = torch.tensor([[7.5, 6.0]])   # (E, 2)
    box_center = torch.tensor([[5.0, 5.0]])
    box_halfsize = torch.tensor([2.0, 3.0])

    force_a, force_b = circle_box_dynamic_forces(predator_pos, 1.0, box_center, box_halfsize, 100.0)
    assert force_a.shape == (1, 2)
    assert torch.allclose(force_a, torch.tensor([[50.0, 0.0]]), atol=1e-3)
    assert torch.allclose(force_b, torch.tensor([[-50.0, 0.0]]), atol=1e-3)


def test_box_box_forces_matches_hand_derivation():
    """Box1 center (0,0) halfsize (2,1); box2 center (3,0.5) halfsize (2,1).
    overlap_x = 1, overlap_y = 1.5 -> resolves along x (smaller overlap),
    penetration 1, direction (-1,0) -> force (-100, 0)."""
    box1_center = torch.tensor([[0.0, 0.0]])
    box1_halfsize = torch.tensor([2.0, 1.0])
    box2_center = torch.tensor([[3.0, 0.5]])
    box2_halfsize = torch.tensor([2.0, 1.0])

    force = box_box_forces(box1_center, box1_halfsize, box2_center, box2_halfsize, 100.0)
    assert torch.allclose(force, torch.tensor([[-100.0, 0.0]]), atol=1e-3)


def test_box_box_forces_no_overlap_is_zero():
    box1_center = torch.tensor([[0.0, 0.0]])
    box1_halfsize = torch.tensor([1.0, 1.0])
    box2_center = torch.tensor([[10.0, 10.0]])
    box2_halfsize = torch.tensor([1.0, 1.0])

    force = box_box_forces(box1_center, box1_halfsize, box2_center, box2_halfsize, 100.0)
    assert torch.allclose(force, torch.zeros_like(force))