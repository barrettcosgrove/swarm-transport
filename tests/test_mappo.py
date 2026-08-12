"""
tests/test_mappo.py

The acceptance checks for train/mappo.py. Complements test_env.py, which covers
the environment these read from.

RL code fails silently -- a backward GAE loop, a misaligned flatten, an
unregistered log_std -- so these target the specific things that produce
plausible numbers while being wrong, rather than the things that would crash:

  shapes            the dimensions every downstream reshape assumes
  finiteness        200 steps of buffer contents, all tensors
  GAE closed form   a hand-built buffer whose advantages have an exact answer
  termination       that a terminal step does not bootstrap across the boundary
  checkpoints       that log_std survives a save/load round trip

Run with: pytest tests/test_mappo.py -v
"""
import dataclasses
import json
import math
import os
import shutil

import torch

from train.config import Config
from train.mappo import Actor, Critic, RolloutBuffer, ValueNormalizer, MAPPOTrainer


ACTION_DIM = 2


def make_test_config(num_envs=4, n_agents=5, rollout_steps=8, max_steps=50, **overrides):
    """A Config sized so a full iteration runs in about a second.

    n_agents stays at the real 5 so obs_dim and state_dim keep the values the
    rest of the project quotes (32 and 165).
    """
    settings = {
        "num_envs": num_envs,
        "n_agents": n_agents,
        "rollout_steps": rollout_steps,
        "max_steps": max_steps,
        "num_iterations": 2,
        "update_epochs": 2,
        "minibatch_size": num_envs * n_agents * rollout_steps // 2,
        "seed": 0,
    }
    settings.update(overrides)
    return dataclasses.replace(Config(), **settings)


def make_buffer(rollout_steps, num_envs, n_agents=2, obs_dim=4):
    return RolloutBuffer(rollout_steps, num_envs, n_agents, obs_dim, ACTION_DIM,
                         torch.device("cpu"))


# ---------------------------------------------------------------- check 1: shapes

def test_dimensions_match_the_documented_values():
    """obs_dim and state_dim are quoted throughout DESIGN.md and the trainer.
    Both are derived properties, so pin them at the real agent count."""
    config = Config()
    assert config.obs_dim == 32
    assert config.state_dim == 165
    assert config.rollout_steps * config.num_envs * config.n_agents == 40_960


def test_one_iteration_produces_the_expected_shapes():
    config = make_test_config()
    T, E, N = config.rollout_steps, config.num_envs, config.n_agents
    trainer = MAPPOTrainer(config)

    trainer.obs = trainer.env.reset()
    assert trainer.obs.shape == (E, N, config.obs_dim)
    assert trainer.critic.build_state(trainer.obs).shape == (E, N, config.state_dim)

    last_value, outcomes = trainer.collect_rollout(training_progress=0.0)
    assert last_value.shape == (E, N)

    assert trainer.buffer.obs.shape == (T, E, N, config.obs_dim)
    assert trainer.buffer.actions.shape == (T, E, N, ACTION_DIM)
    for name in ("log_probs", "values", "rewards", "terminated", "truncated", "trunc_values"):
        assert getattr(trainer.buffer, name).shape == (T, E, N), f"{name} has the wrong shape"

    advantages, returns = trainer.buffer.compute_gae(last_value, config.gamma, config.gae_lambda)
    assert advantages.shape == (T, E, N)
    assert returns.shape == (T, E, N)

    flat = trainer.buffer.flatten(trainer.critic, advantages, returns)
    rows = T * E * N
    assert flat["obs"].shape == (rows, config.obs_dim)
    assert flat["state"].shape == (rows, config.state_dim)
    assert flat["actions"].shape == (rows, ACTION_DIM)
    for name in ("log_probs", "advantages", "returns"):
        assert flat[name].shape == (rows,), f"flat[{name}] has the wrong shape"

    assert set(outcomes) == {"episodes_completed", "success_rate", "capture_rate", "timeout_rate"}


def test_actor_never_sees_more_than_its_own_observation():
    """A hard deployment constraint, not a preference: each agent runs its own
    ONNX model in the browser with local perception only. The actor's input
    width must be obs_dim, so handing it a joint state has to fail."""
    config = Config()
    actor = Actor(config.obs_dim, ACTION_DIM, config.hidden_dim)

    assert actor.net[0].in_features == config.obs_dim

    state = torch.zeros(4, config.n_agents, config.state_dim)
    try:
        actor(state)
    except RuntimeError:
        pass
    else:
        raise AssertionError("the actor accepted a joint state as input")


def test_actor_forward_is_the_mean_only():
    """forward has to stay traceable as obs -> mean for torch.onnx.export, so
    it must be deterministic and must not consume log_std."""
    config = Config()
    actor = Actor(config.obs_dim, ACTION_DIM, config.hidden_dim)
    obs = torch.randn(3, config.n_agents, config.obs_dim)

    assert torch.equal(actor(obs), actor(obs)), "forward is stochastic"
    with torch.no_grad():
        actor.log_std += 5.0
    assert torch.equal(actor(obs), actor.net(obs)), "forward is not just the network"


def test_critic_distinguishes_agents_within_an_environment():
    """Without the one-hot every row of an environment's critic input is
    identical, and a deterministic network must return identical values -- wrong,
    because compute_reward gives each agent a private proximity term."""
    config = Config()
    critic = Critic(config.obs_dim, config.n_agents, config.hidden_dim)
    obs = torch.randn(2, config.n_agents, config.obs_dim)

    values = critic(obs)
    assert values.shape == (2, config.n_agents)
    assert values[0].std() > 0.0, "the critic returned the same value for every agent"

    # the joint half is shared, only the one-hot tail differs
    state = critic.build_state(obs)
    joint_width = config.n_agents * config.obs_dim
    assert torch.equal(state[0, 0, :joint_width], state[0, 1, :joint_width])
    assert not torch.equal(state[0, 0, joint_width:], state[0, 1, joint_width:])


def test_critic_build_state_accepts_both_leading_sizes():
    """Called with num_envs during a rollout and rollout_steps*num_envs when the
    buffer is flattened, so it must read the batch size off the input."""
    config = Config()
    critic = Critic(config.obs_dim, config.n_agents, config.hidden_dim)

    for batch in (config.num_envs, 7 * config.num_envs):
        obs = torch.randn(batch, config.n_agents, config.obs_dim)
        assert critic.build_state(obs).shape == (batch, config.n_agents, config.state_dim)


# ---------------------------------------------------------------- check 2: no NaNs

def test_two_hundred_steps_of_rollout_stay_finite():
    """Random actions rather than a policy, so this isolates the buffer and the
    environment from anything the network is doing."""
    config = make_test_config(rollout_steps=200, max_steps=20)
    trainer = MAPPOTrainer(config)
    trainer.obs = trainer.env.reset()

    last_value, _ = trainer.collect_rollout(training_progress=0.5)
    advantages, returns = trainer.buffer.compute_gae(last_value, config.gamma, config.gae_lambda)
    flat = trainer.buffer.flatten(trainer.critic, advantages, returns)

    for name in ("obs", "actions", "log_probs", "values", "rewards", "trunc_values"):
        tensor = getattr(trainer.buffer, name)
        assert torch.isfinite(tensor).all(), f"buffer.{name} is not finite"

    assert torch.isfinite(advantages).all(), "advantages are not finite"
    assert torch.isfinite(returns).all(), "returns are not finite"
    for name, tensor in flat.items():
        assert torch.isfinite(tensor).all(), f"flat[{name}] is not finite"

    # with max_steps 20 and 200 steps, every environment must have truncated
    assert trainer.buffer.truncated.any(), "no truncation in 200 steps at max_steps=20"


def test_stored_action_is_raw_and_the_environment_gets_a_clamped_copy():
    """The PPO ratio re-evaluates the stored action, so it has to be the raw
    sample log_prob was computed against. Nothing in physics.step clamps, hence
    the separate clamped copy."""
    config = make_test_config(rollout_steps=64)
    trainer = MAPPOTrainer(config)
    trainer.obs = trainer.env.reset()
    trainer.collect_rollout(training_progress=0.0)

    # at std 1.0 over 64*4*5 samples, |a| > 1 is overwhelmingly likely
    assert trainer.buffer.actions.abs().max() > 1.0, \
        "stored actions look clamped -- the ratio would compare different actions"


# ---------------------------------------------------------------- check 3: GAE closed form

def test_gae_matches_the_closed_form_with_no_terminations():
    """rewards=1, values=0, lambda=1, nothing done: the advantage at t is just
    the discounted sum of the remaining rewards, sum(gamma^k) for k < T-t."""
    T, E, N = 6, 2, 2
    gamma, gae_lambda = 0.99, 1.0

    buffer = make_buffer(T, E, N)
    buffer.rewards.fill_(1.0)
    buffer.values.zero_()
    last_value = torch.zeros(E, N)

    advantages, returns = buffer.compute_gae(last_value, gamma, gae_lambda)

    for t in range(T):
        expected = sum(gamma ** k for k in range(T - t))
        assert torch.allclose(advantages[t], torch.full((E, N), expected), atol=1e-5), \
            f"advantage at t={t} does not match the geometric sum"

    # values are zero, so returns are the advantages
    assert torch.allclose(returns, advantages, atol=1e-6)


def test_gae_runs_backward_in_time():
    """The direct guard on loop direction. With a single reward at the last
    step, only the earlier steps can see it -- a forward loop would put the
    accumulation at the wrong end."""
    T, E, N = 4, 1, 1
    buffer = make_buffer(T, E, N)
    buffer.rewards[T - 1] = 1.0

    advantages, _ = buffer.compute_gae(torch.zeros(E, N), gamma=1.0, gae_lambda=1.0)

    # every step's advantage is that one future reward, undiscounted
    assert torch.allclose(advantages, torch.ones(T, E, N), atol=1e-6)


# ---------------------------------------------------------------- check 4: termination isolation

def test_terminal_step_does_not_bootstrap():
    """One of two environments terminates mid-rollout. Its advantage at the
    terminal step must be exactly reward - value: no next_value leaks in, and
    the GAE chain does not carry momentum across the boundary."""
    T, E, N = 5, 2, 2
    gamma, gae_lambda = 0.99, 0.95
    terminal_t = 2

    buffer = make_buffer(T, E, N)
    buffer.rewards.fill_(1.0)
    buffer.values.fill_(3.0)
    buffer.terminated[terminal_t, 0] = True

    advantages, _ = buffer.compute_gae(torch.full((E, N), 3.0), gamma, gae_lambda)

    assert torch.allclose(advantages[terminal_t, 0], torch.full((N,), 1.0 - 3.0), atol=1e-6), \
        "the terminal step bootstrapped from a future value"

    # the other environment, at the same t, bootstraps normally
    assert not torch.allclose(advantages[terminal_t, 0], advantages[terminal_t, 1]), \
        "termination in one environment leaked into the other"

    # The boundary is one-directional. Nothing from AFTER the terminal step
    # reaches it (asserted above), but the terminal step is still part of the
    # episode that preceded it, so its advantage does flow backward.
    delta_before = 1.0 + gamma * 3.0 - 3.0
    expected_before = delta_before + gamma * gae_lambda * float(advantages[terminal_t, 0, 0])
    assert torch.allclose(advantages[terminal_t - 1, 0],
                          torch.full((N,), expected_before), atol=1e-6), \
        "the step before termination lost the terminal step's advantage"


def test_truncation_bootstraps_from_the_cached_final_value():
    """A truncated episode did not really end, so it bootstraps from the value
    of where it stopped -- trunc_values, not values[t+1], which after the
    auto-reset describes a different episode entirely."""
    T, E, N = 4, 1, 1
    gamma, gae_lambda = 0.99, 0.95
    trunc_t = 1

    buffer = make_buffer(T, E, N)
    buffer.rewards.fill_(1.0)
    buffer.values.fill_(2.0)
    buffer.truncated[trunc_t] = True
    buffer.trunc_values[trunc_t] = 10.0

    advantages, _ = buffer.compute_gae(torch.full((E, N), 2.0), gamma, gae_lambda)

    expected = 1.0 + gamma * 10.0 - 2.0
    assert torch.allclose(advantages[trunc_t], torch.full((E, N), expected), atol=1e-6), \
        "truncation did not use the cached final value"


def test_termination_wins_over_truncation():
    """compute_done makes the two mutually exclusive, but the ordering of the
    two torch.where calls must not depend on that."""
    T, E, N = 3, 1, 1
    buffer = make_buffer(T, E, N)
    buffer.rewards.fill_(1.0)
    buffer.values.fill_(2.0)
    buffer.terminated[1] = True
    buffer.truncated[1] = True
    buffer.trunc_values[1] = 50.0

    advantages, _ = buffer.compute_gae(torch.full((E, N), 2.0), gamma=0.99, gae_lambda=0.95)

    assert torch.allclose(advantages[1], torch.full((E, N), 1.0 - 2.0), atol=1e-6), \
        "truncation overrode termination"


def test_clear_wipes_stale_truncation_bootstraps():
    """trunc_values is written through a boolean mask, so anything not
    overwritten this iteration survives from the last one."""
    buffer = make_buffer(3, 2)
    buffer.trunc_values.fill_(99.0)
    buffer.clear()
    assert not buffer.trunc_values.any(), "clear left stale trunc_values behind"


# ---------------------------------------------------------------- check 5: checkpoint round trip

def test_checkpoint_round_trip_restores_every_tensor():
    """Including log_std and agent_ids, which are only carried because one is a
    registered Parameter and the other a registered buffer."""
    from train.checkpoints import load_checkpoint

    config = make_test_config()
    trainer = MAPPOTrainer(config)

    # move every parameter off its initial value so a no-op load cannot pass
    with torch.no_grad():
        for param in list(trainer.actor.parameters()) + list(trainer.critic.parameters()):
            param.add_(torch.randn_like(param))

    path = f"{config.checkpoint_dir}/_test_roundtrip.pt"
    trainer.save(path, iteration=7)

    fresh_actor = Actor(config.obs_dim, ACTION_DIM, config.hidden_dim)
    fresh_critic = Critic(config.obs_dim, config.n_agents, config.hidden_dim)
    iteration = load_checkpoint(path, fresh_actor, fresh_critic)

    assert iteration == 7

    for (name, saved), (_, loaded) in zip(trainer.actor.state_dict().items(),
                                          fresh_actor.state_dict().items()):
        assert torch.equal(saved, loaded), f"actor.{name} did not round trip"
    for (name, saved), (_, loaded) in zip(trainer.critic.state_dict().items(),
                                          fresh_critic.state_dict().items()):
        assert torch.equal(saved, loaded), f"critic.{name} did not round trip"

    assert "log_std" in fresh_actor.state_dict()
    assert torch.equal(trainer.actor.log_std, fresh_actor.log_std)

    # same observation, same mean -- the check that actually matters downstream
    obs = torch.randn(2, config.n_agents, config.obs_dim)
    assert torch.equal(trainer.actor(obs), fresh_actor(obs))

    os.remove(path)


def test_checkpoint_loads_without_optimizers():
    """export_onnx.py and render.py want weights only, and must not have to
    build optimizers to get them."""
    from train.checkpoints import load_checkpoint

    config = make_test_config()
    trainer = MAPPOTrainer(config)
    path = f"{config.checkpoint_dir}/_test_weights_only.pt"
    trainer.save(path, iteration=3)

    actor = Actor(config.obs_dim, ACTION_DIM, config.hidden_dim)
    critic = Critic(config.obs_dim, config.n_agents, config.hidden_dim)
    assert load_checkpoint(path, actor, critic) == 3

    os.remove(path)


# ---------------------------------------------------------------- value normalizer

def test_value_normalizer_round_trips_and_tracks_statistics():
    normalizer = ValueNormalizer()
    sample = torch.randn(4096) * 17.0 + 5.0

    normalizer.update(sample)
    assert abs(float(normalizer.mean) - float(sample.mean())) < 0.1
    assert abs(float(normalizer.var.sqrt()) - float(sample.std(unbiased=False))) < 0.5

    normalized = normalizer.normalize(sample)
    assert abs(float(normalized.mean())) < 0.05
    assert torch.allclose(normalizer.denormalize(normalized), sample, atol=1e-3)


def test_value_normalizer_batched_update_matches_a_single_pass():
    """Two updates over halves must give the same statistics as one update over
    the whole -- the property that makes the parallel-variance form usable."""
    data = torch.randn(2048) * 3.0 - 1.0

    incremental = ValueNormalizer()
    incremental.update(data[:1024])
    incremental.update(data[1024:])

    one_shot = ValueNormalizer()
    one_shot.update(data)

    assert abs(float(incremental.mean) - float(one_shot.mean)) < 1e-4
    assert abs(float(incremental.var) - float(one_shot.var)) < 1e-3


# ---------------------------------------------------------------- check 6: smoke

def test_short_training_run_is_finite_and_writes_its_log():
    config = make_test_config(rollout_steps=16, max_steps=20, num_iterations=3,
                              log_path="outputs/_test_history.json",
                              checkpoint_dir="train/checkpoints/_test_smoke")
    history = MAPPOTrainer(config).train(verbose=False)

    assert len(history) == 3
    for record in history:
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
                    "mean_action_std", "mean_episode_return", "learning_rate"):
            assert key in record, f"the log is missing {key}"
            assert math.isfinite(record[key]), f"{key} is not finite"
        assert record["state_dim"] == config.state_dim
        assert record["n_agents"] == config.n_agents

    assert os.path.exists(config.log_path)
    with open(config.log_path) as f:
        assert len(json.load(f)) == 3

    # num_iterations is the final iteration, so a checkpoint is always written
    assert os.path.exists(f"{config.checkpoint_dir}/checkpoint_latest.pt")

    shutil.rmtree(config.checkpoint_dir)
    os.remove(config.log_path)


def test_learning_rate_anneals_to_zero_across_the_run():
    config = make_test_config(rollout_steps=8, max_steps=20, num_iterations=4,
                              log_path="outputs/_test_lr.json",
                              checkpoint_dir="train/checkpoints/_test_lr")
    history = MAPPOTrainer(config).train(verbose=False)

    rates = [record["learning_rate"] for record in history]
    assert rates == sorted(rates, reverse=True), "the learning rate did not decrease"
    assert rates[0] == config.lr

    shutil.rmtree(config.checkpoint_dir)
    os.remove(config.log_path)


def test_seeding_makes_a_rollout_reproducible():
    """Normal.sample() reads the global RNG, so this fails unless the trainer
    seeds it -- the environment's own generator is not enough."""
    config = make_test_config(rollout_steps=8, max_steps=20)

    first = MAPPOTrainer(config)
    first.obs = first.env.reset()
    first.collect_rollout(training_progress=0.0)

    second = MAPPOTrainer(config)
    second.obs = second.env.reset()
    second.collect_rollout(training_progress=0.0)

    assert torch.equal(first.buffer.actions, second.buffer.actions)
    assert torch.equal(first.buffer.rewards, second.buffer.rewards)
