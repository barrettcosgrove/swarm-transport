"""
tests/test_env.py

Tests the full env.py / scenario.py surface -- reset(), step(), and the
observation/reward/done pipeline they call into. Complements test_physics.py,
which tests world.py + physics.py in isolation.

Run with: pytest tests/test_env.py -v
"""
import dataclasses
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

    for key in ("payload_dist", "success", "captured", "world_state", "scenario_state"):
        assert key in info, f"info is missing {key}"

    for key in ("payload_dist", "success", "captured"):
        assert info[key].shape == (config.num_envs,), f"info[{key}] has the wrong shape"

    assert_finite(info["payload_dist"], "info payload_dist")
    assert torch.equal(info["success"], info["payload_dist"] < config.success_threshold)

    assert isinstance(info["world_state"], WorldState)
    assert isinstance(info["scenario_state"], scenario.ScenarioState)


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