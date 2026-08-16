"""
tests/test_env.py

Tests the full env.py / scenario.py surface -- reset(), step(), and the
observation/reward/done pipeline they call into. Complements test_physics.py,
which tests world.py + physics.py in isolation.

Run with: pytest tests/test_env.py -v
"""
import dataclasses
import math
import torch
import pytest

from env.env import Env
from env.world import WorldState
from train.config import Config
import env.scenario as scenario


# ---------------------------------------------------------------- fixtures

def make_test_config(num_envs=4, n_agents=3, max_steps=50, seed=0):
    """
    A Config with a short max_steps -- deliberately small so rollout tests
    exercise truncation (and therefore reset_at) quickly, rather than
    needing hundreds of steps before any environment resets.
    """
    return Config(
        num_envs=num_envs,
        n_agents=n_agents,
        max_steps=max_steps,
        seed=seed,
    )


def assert_finite(t, name):
    assert torch.isfinite(t).all(), f"{name} contains NaN or Inf"


# ---------------------------------------------------------------- shape / sanity tests

def test_reset_produces_correct_shapes():
    config = make_test_config()
    env = Env(config)
    obs = env.reset()

    assert obs.shape == (config.num_envs, config.n_agents, config.obs_dim)
    assert_finite(obs, "reset observation")


def test_step_produces_correct_shapes():
    config = make_test_config()
    env = Env(config)
    env.reset()

    actions = torch.rand(config.num_envs, config.n_agents, 2) * 2 - 1
    obs, reward, terminated, truncated, info = env.step(actions, training_progress=0.0)

    assert obs.shape == (config.num_envs, config.n_agents, config.obs_dim)
    assert reward.shape == (config.num_envs, config.n_agents)
    assert terminated.shape == (config.num_envs,)
    assert truncated.shape == (config.num_envs,)
    assert terminated.dtype == torch.bool
    assert truncated.dtype == torch.bool

    assert_finite(obs, "step observation")
    assert_finite(reward, "step reward")


def test_step_info_contents():
    """The info dict is what the renderer and any logging read, so pin its
    keys, shapes, and the one relationship that holds between them."""
    config = make_test_config()
    env = Env(config)
    env.reset()

    actions = torch.rand(config.num_envs, config.n_agents, 2) * 2 - 1
    _, _, _, _, info = env.step(actions, training_progress=0.0)

    for key in ("payload_dist", "success", "captured", "world_state", "scenario_state",
                "final_observation"):
        assert key in info, f"info is missing {key}"

    for key in ("payload_dist", "success", "captured"):
        assert info[key].shape == (config.num_envs,), f"info[{key}] has the wrong shape"

    assert info["final_observation"].shape == (config.num_envs, config.n_agents, config.obs_dim)
    assert_finite(info["final_observation"], "info final_observation")

    assert_finite(info["payload_dist"], "info payload_dist")
    assert torch.equal(info["success"], info["payload_dist"] < config.success_threshold)

    assert isinstance(info["world_state"], WorldState)
    assert isinstance(info["scenario_state"], scenario.ScenarioState)


def test_step_returns_the_post_reset_observation():
    """A rollout loop assigns obs = next_obs and acts on it next step, so the
    returned observation has to describe the state the simulation is actually
    in. Returning the pre-reset one would hand the policy a stale board on
    every terminal step; that pre-reset view lives in info["final_observation"]
    instead, where GAE can bootstrap from it.
    """
    config = make_test_config(max_steps=3)
    env = Env(config)
    env.reset()

    actions = torch.zeros(config.num_envs, config.n_agents, 2)
    saw_a_reset = False
    for _ in range(config.max_steps + 1):
        obs, _, terminated, truncated, info = env.step(actions, training_progress=0.0)
        done = terminated | truncated

        # the returned observation must match a fresh observe() of the CURRENT
        # state, which is what the next step will actually advance
        expected = scenario.observe(env.world_state, env.scenario_state, config)
        assert torch.equal(obs, expected), "the returned observation is not the current one"

        if done.any():
            saw_a_reset = True
            assert not torch.equal(obs[done], info["final_observation"][done]), \
                "a reset environment returned its pre-reset observation"
            # untouched environments see the same thing either way
            assert torch.equal(obs[~done], info["final_observation"][~done])

    assert saw_a_reset, "no environment ended, so the post-reset path was never exercised"


def test_truncation_fires_at_max_steps():
    """With no predator interference and max_steps small, every environment
    should truncate by the time step_count reaches max_steps, if nothing
    else ends the episode first."""
    config = make_test_config(max_steps=5)
    env = Env(config)
    env.reset()

    actions = torch.zeros(config.num_envs, config.n_agents, 2)   # no thrust, just wait it out
    truncated_any = torch.zeros(config.num_envs, dtype=torch.bool)
    for _ in range(config.max_steps + 1):
        _, _, terminated, truncated, _ = env.step(actions, training_progress=0.0)
        truncated_any |= truncated

    assert truncated_any.all(), "expected every environment to truncate within max_steps + 1 calls"


# ---------------------------------------------------------------- observation regression

def test_observe_own_pos_matches_world_state():
    """The first slice of the observation vector should be each agent's own
    position, taken directly from world_state -- no transformation at all."""
    config = make_test_config(num_envs=1, n_agents=2)
    world_state, scenario_state = scenario.reset(config.num_envs, config, torch.Generator().manual_seed(0))

    obs = scenario.observe(world_state, scenario_state, config)
    own_pos_from_obs = obs[..., 0:2]

    assert torch.allclose(own_pos_from_obs, world_state.agent_pos, atol=1e-5)


def test_observe_goal_offset_matches_manual_calc():
    """The goal-offset slice should equal goal_pos - agent_pos, computed by
    hand against the same world/scenario state observe() receives."""
    config = make_test_config(num_envs=1, n_agents=2)
    world_state, scenario_state = scenario.reset(config.num_envs, config, torch.Generator().manual_seed(0))

    obs = scenario.observe(world_state, scenario_state, config)
    goal_offset_from_obs = obs[..., 4:6]   # adjust the slice if your field order differs

    expected = scenario_state.goal_pos.unsqueeze(1) - world_state.agent_pos
    assert torch.allclose(goal_offset_from_obs, expected, atol=1e-5)


# ---------------------------------------------------------------- reset_at correctness

def test_reset_at_leaves_unflagged_environments_untouched():
    """The core invariant behind staggered termination: resetting some
    environments in a batch must not change anything about the others.
    Uses step_count as a deterministic marker rather than relying on
    positions merely being "probably different" after a reset.
    """
    config = make_test_config(num_envs=4)
    generator = torch.Generator().manual_seed(0)
    world_state, scenario_state = scenario.reset(config.num_envs, config, generator)

    # give every environment a distinct, nonzero step_count so a reset to 0
    # is unambiguous
    scenario_state = dataclasses.replace(
        scenario_state, step_count=torch.tensor([10, 20, 30, 40])
    )
    agent_pos_before = world_state.agent_pos.clone()

    needs_reset = torch.tensor([True, False, True, False])
    new_world, new_scenario = scenario.reset_at(world_state, scenario_state, needs_reset, config, generator)

    # flagged environments (0, 2): step_count reset to 0
    assert new_scenario.step_count[0] == 0
    assert new_scenario.step_count[2] == 0

    # unflagged environments (1, 3): step_count and position completely unchanged
    assert new_scenario.step_count[1] == 20
    assert new_scenario.step_count[3] == 40
    assert torch.equal(new_world.agent_pos[1], agent_pos_before[1])
    assert torch.equal(new_world.agent_pos[3], agent_pos_before[3])


def test_reset_at_no_environments_flagged_changes_nothing():
    """A needs_reset mask of all False should be a complete no-op."""
    config = make_test_config(num_envs=4)
    generator = torch.Generator().manual_seed(0)
    world_state, scenario_state = scenario.reset(config.num_envs, config, generator)

    needs_reset = torch.zeros(config.num_envs, dtype=torch.bool)
    new_world, new_scenario = scenario.reset_at(world_state, scenario_state, needs_reset, config, generator)

    assert torch.equal(new_world.agent_pos, world_state.agent_pos)
    assert torch.equal(new_scenario.step_count, scenario_state.step_count)


# ---------------------------------------------------------------- obstacle spawn invariants

SPAWN_SEEDS = (0, 1, 2)


def point_box_distance(points, box_center, box_halfsize):
    """(E, P, B) exterior distance from each point to each box, 0 when inside.

    Exact for axis-aligned boxes, where the spawn constants are derived from
    a box's circumradius. Checking against the exact distance means these
    assertions fail on a real overlap rather than on the conservative bound
    happening to be tight.
    """
    delta = (points.unsqueeze(2) - box_center.unsqueeze(1)).abs()          # (E, P, B, 2)
    return torch.norm(torch.clamp(delta - box_halfsize, min=0.0), dim=-1)


def box_box_gap(center_a, halfsize_a, center_b, halfsize_b):
    """(E, A, B) surface-to-surface separation between two sets of boxes."""
    delta = (center_a.unsqueeze(2) - center_b.unsqueeze(1)).abs()          # (E, A, B, 2)
    halfsize_sum = halfsize_a.unsqueeze(1) + halfsize_b.unsqueeze(0)       # (A, B, 2)
    return torch.norm(torch.clamp(delta - halfsize_sum, min=0.0), dim=-1)


def spawn_batch(seed, num_envs=256):
    config = make_test_config(num_envs=num_envs, n_agents=5, seed=seed)
    world_state, scenario_state = scenario.reset(
        num_envs, config, torch.Generator().manual_seed(seed)
    )
    return config, world_state, scenario_state


def passage_width(config):
    """What has to fit through a gap: the payload, flanked by an agent a side."""
    return 2 * float(config.payload_halfsize.max()) + 4 * config.agent_radius


def test_obstacle_layout_varies_across_environments_and_resets():
    """The direct regression guard. Obstacle centers used to be one config
    constant broadcast over the batch, so every environment of every episode
    got the same four boxes in the same places."""
    config, world_state, _ = spawn_batch(seed=0)
    centers = world_state.obstacle_center

    assert not torch.allclose(centers[0], centers[1]), "two environments share a layout"
    assert centers.std(dim=0).min() > 0.1, \
        "an obstacle lands in the same place in every environment"

    generator = torch.Generator().manual_seed(0)
    first, _ = scenario.reset(config.num_envs, config, generator)
    second, _ = scenario.reset(config.num_envs, config, generator)
    assert not torch.allclose(first.obstacle_center, second.obstacle_center), \
        "a second reset reproduced the first layout"


def test_obstacle_layout_is_reproducible_from_the_generator():
    """Layouts must come from the env's generator and nothing else -- a
    *_like sampler would read the global RNG and quietly make seeded rollouts
    irreproducible, which is a bug this codebase has already hit once."""
    config = make_test_config(num_envs=16)

    first, _ = scenario.reset(config.num_envs, config, torch.Generator().manual_seed(7))
    second, _ = scenario.reset(config.num_envs, config, torch.Generator().manual_seed(7))

    assert torch.equal(first.obstacle_center, second.obstacle_center)


def test_obstacles_clear_every_other_entity():
    """Obstacles are placed last, so nothing already on the board may be
    overlapping one when the episode starts."""
    for seed in SPAWN_SEEDS:
        config, world_state, scenario_state = spawn_batch(seed)
        centers, halfsize = world_state.obstacle_center, world_state.obstacle_halfsize

        agent_gap = point_box_distance(world_state.agent_pos, centers, halfsize)
        assert agent_gap.min() > config.agent_radius, f"agent inside an obstacle (seed {seed})"

        predator_gap = point_box_distance(world_state.predator_pos.unsqueeze(1), centers, halfsize)
        assert predator_gap.min() > config.predator_radius, \
            f"predator inside an obstacle (seed {seed})"

        payload_gap = box_box_gap(world_state.payload_pos.unsqueeze(1),
                                   config.payload_halfsize.unsqueeze(0), centers, halfsize)
        assert payload_gap.min() > 0.0, f"payload inside an obstacle (seed {seed})"

        # the goal needs more room than a point: the payload has to be able to
        # rest anywhere inside the success circle without an obstacle in the
        # way, and the marker the renderer draws is that whole circle
        goal_gap = point_box_distance(scenario_state.goal_pos.unsqueeze(1), centers, halfsize)
        payload_reach = config.success_threshold + float(torch.norm(config.payload_halfsize))
        assert goal_gap.min() > payload_reach, f"obstacle clipping the goal (seed {seed})"


def test_obstacles_never_seal_a_passage():
    """DESIGN's clearance rule. Every obstacle pair leaves a gap the payload
    and a flanking agent per side can pass through, which is what makes
    walling off a region impossible for any arrangement at all."""
    for seed in SPAWN_SEEDS:
        config, world_state, _ = spawn_batch(seed)
        centers, halfsize = world_state.obstacle_center, world_state.obstacle_halfsize

        gap = box_box_gap(centers, halfsize, centers, halfsize)            # (E, K, K)
        gap = gap.masked_fill(torch.eye(config.n_obstacles, dtype=torch.bool), float("inf"))

        assert gap.min() > passage_width(config), f"obstacle pair too tight (seed {seed})"


def test_obstacles_leave_the_perimeter_ring_clear():
    """The clear ring at the boundary is the payload's guaranteed detour
    around any interior cluster, so obstacles must stay out of it."""
    for seed in SPAWN_SEEDS:
        config, world_state, _ = spawn_batch(seed)

        inner_face = min(
            abs(float(center[int(torch.argmin(halfsize))])) - float(halfsize.min())
            for center, halfsize in zip(config.wall_center, config.wall_halfsize)
        )
        extent = (world_state.obstacle_center.abs() + world_state.obstacle_halfsize).max()

        assert float(extent) < inner_face - passage_width(config), \
            f"obstacle encroaching on the perimeter ring (seed {seed})"


def test_reset_at_resamples_obstacles_only_for_flagged_environments():
    config = make_test_config(num_envs=4)
    generator = torch.Generator().manual_seed(0)
    world_state, scenario_state = scenario.reset(config.num_envs, config, generator)
    before = world_state.obstacle_center.clone()

    needs_reset = torch.tensor([True, False, True, False])
    new_world, _ = scenario.reset_at(world_state, scenario_state, needs_reset, config, generator)

    assert torch.equal(new_world.obstacle_center[1], before[1])
    assert torch.equal(new_world.obstacle_center[3], before[3])
    assert not torch.allclose(new_world.obstacle_center[0], before[0])
    assert not torch.allclose(new_world.obstacle_center[2], before[2])


# ---------------------------------------------------------------- rollout smoke test

def test_long_rollout_never_produces_nan():
    """Run many steps -- comfortably more than max_steps, so multiple
    environments truncate and get reset multiple times over -- and confirm
    nothing ever goes non-finite. This is the test that exercises reset_at
    happening repeatedly, interleaved with ordinary stepping, the way it
    actually will during training.
    """
    config = make_test_config(num_envs=8, max_steps=20)
    env = Env(config)
    obs = env.reset()
    assert_finite(obs, "initial observation")

    for step_idx in range(300):
        actions = torch.rand(config.num_envs, config.n_agents, 2) * 2 - 1
        training_progress = step_idx / 300
        obs, reward, terminated, truncated, _ = env.step(actions, training_progress)

        assert_finite(obs, f"observation at step {step_idx}")
        assert_finite(reward, f"reward at step {step_idx}")
        assert not (terminated & truncated).any(), \
            f"step {step_idx}: an environment reported both terminated and truncated"


# ---------------------------------------------------------------- containment
#
# DESIGN.md's stated reason for having walls at all: a body that cannot leave
# keeps observations bounded, and an unbounded state space is bad input for a
# network. These pin that end to end, through env.step rather than at the
# physics layer, because the failure that motivated them was a config value
# not being passed down -- exactly the kind of gap a physics-only test misses.

def wall_midline(config):
    """Distance from the origin to the center of a wall.

    Not the inner face. A penalty force resolves an overlap along the
    shortest way out, so a body past a wall's center gets pushed toward the
    OUTER face and is expelled from the arena for good. Sinking into a wall
    is ordinary; reaching its center is not survivable.
    """
    return min(
        abs(float(center[int(torch.argmin(halfsize))]))
        for center, halfsize in zip(config.wall_center, config.wall_halfsize)
    )


def sustained_thrust_actions(config):
    """One fixed compass direction per agent, held for the whole rollout.

    Random per-step actions average out to almost no displacement and never
    put a body near a wall, so they cannot catch a containment bug -- the
    rollout smoke test above ran 300 steps of them without ever coming close.
    Committing each agent to one direction and never letting up is what
    drives every body into a wall and holds it there.
    """
    angles = torch.arange(config.num_envs * config.n_agents).float()
    angles = angles * (2 * math.pi / (config.num_envs * config.n_agents))
    directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    return directions.view(config.num_envs, config.n_agents, 2)


def test_sustained_thrust_cannot_push_a_body_out_of_the_arena():
    """Every agent drives into a wall and stays there for several episodes.

    This is the regression test for agents escaping: before the speed cap
    and the thicker walls, bodies crossed a wall's midline, were ejected by
    the very force meant to contain them, and reached positions past 370.
    """
    config = make_test_config(num_envs=8, n_agents=5, max_steps=100)
    env = Env(config)
    env.reset()
    actions = sustained_thrust_actions(config)
    midline = wall_midline(config)

    max_agent = 0.0
    max_payload = 0.0
    max_obs = 0.0
    for step_idx in range(400):
        obs, _, _, _, info = env.step(actions, training_progress=step_idx / 400)
        world = info["world_state"]
        max_agent = max(max_agent, float(world.agent_pos.abs().max()))
        max_payload = max(max_payload, float(world.payload_pos.abs().max()))
        # health is the trailing channel and lives on a 0-100 scale of its own,
        # so it is excluded rather than folded into a spatial bound
        max_obs = max(max_obs, float(obs[..., :-1].abs().max()))

    assert max_agent < midline, f"an agent reached {max_agent:.1f}, past the wall midline at {midline}"

    # the payload has no speed cap and rests entirely on the geometric margin,
    # so it gets its own assertion rather than riding on the agents'
    assert max_payload < midline, f"the payload reached {max_payload:.1f}, past the wall midline"

    # the point of all of the above. Every remaining channel is a position, an
    # offset between two of them, or a relative velocity, so the widest any of
    # them can legitimately get is two arenas or two speed caps. One escaped
    # agent used to drag this past 370, and it poisons its four teammates'
    # observations too, through the relative-position block.
    bound = max(2 * midline, 2 * config.agent_max_speed)
    assert max_obs < bound, f"observations reached {max_obs:.1f} with bodies still inside the arena"