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