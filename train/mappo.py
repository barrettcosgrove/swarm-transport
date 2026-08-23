"""
train/mappo.py

MAPPO for the swarm-transport agents. Decentralized actor on local
observations, centralized critic on joint observations, both shared across the
homogeneous agents.

Adapted from a single-script version validated on VMAS navigation (3 agents, 64
envs): episode return -0.19 -> ~3.0 over 250 iterations, action std annealing
1.0 -> ~0.55 with no collapse. The maths is unchanged; the structure is not.

This pass is agents only. The predator stays scripted in
scenario.predator_policy and is driven entirely inside env.step, so there is one
actor and one critic here and nothing else.

Usage:
    python -m train.mappo
"""
import dataclasses
import json
import os
import time
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from env import scenario
from env.env import Env
from train.checkpoints import save_checkpoint, load_checkpoint
from train.config import Config


# ---------------------------------------------------------------- actor

class Actor(nn.Module):
    """The deployed policy: one agent's local observation in, action mean out.

    Shared across agents, which is what makes a single ONNX file enough for a
    whole team. nn.Linear maps over the last dimension only, so an
    (E, n_agents, obs_dim) batch flows through untouched -- there is no loop
    over agents anywhere in this file.

    Deliberately never given access to the joint observation. Each agent runs
    its own copy of this network in the browser with local perception only, so
    global state in the actor's input is not a design preference, it is
    undeployable.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        # A registered Parameter, not a free tensor alongside the module. In the
        # single-script version it lived outside the network and had to be
        # appended to the optimizer's parameter list by hand; forgetting that
        # meant exploration noise never updated, with no error anywhere. As a
        # Parameter it is picked up by parameters(), state_dict() and therefore
        # checkpointing, automatically.
        #
        # State-independent: the spread is a property of the policy's current
        # confidence, not of the observation.
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        """Observation -> action mean. Nothing else.

        Sampling and log_std stay out of here so torch.onnx.export traces
        exactly the graph the browser needs. Putting the distribution in
        forward would drag a sampler into the exported graph.
        """
        return self.net(obs)

    def distribution(self, obs):
        mean = self.net(obs)
        # log_std is (action_dim,) and broadcasts against (..., action_dim)
        return torch.distributions.Normal(mean, self.log_std.exp())

    def act(self, obs):
        """Sample an action and its log-probability.

        log_prob sums over the action axis: x and y are one joint action, so
        their log-probabilities add. The agent axis is preserved.
        """
        dist = self.distribution(obs)
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1)

    def evaluate(self, obs, action):
        """Re-score an action taken by an older version of this policy.

        This is what the PPO ratio is built from, so it must be given the raw
        sampled action -- the same value log_prob was computed against during
        the rollout, not the clamped copy handed to the environment.
        """
        dist = self.distribution(obs)
        return dist.log_prob(action).sum(-1), dist.entropy().sum(-1)


# ---------------------------------------------------------------- critic

class Critic(nn.Module):
    """Centralized value function. Training-only, never exported.

    Input is every agent's observation concatenated, plus a one-hot agent id.
    The one-hot is the only thing distinguishing the n_agents rows belonging to
    one environment: without it they are byte-identical, and a deterministic
    network would have to return the same value for all of them. That would be
    wrong, because compute_reward gives each agent a private approach_reward
    and collision_reward.
    """

    def __init__(self, obs_dim, n_agents, hidden_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.state_dim = n_agents * obs_dim + n_agents

        self.net = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # register_buffer, not a plain attribute: it moves with .to(device) and
        # appears in state_dict without being trainable.
        self.register_buffer("agent_ids", torch.eye(n_agents).unsqueeze(0))   # (1, N, N)

    def build_state(self, obs):
        """(B, N, obs_dim) -> (B, N, state_dim).

        A method rather than a free function: assembling the joint observation
        is part of what a centralized critic is, and nothing outside this class
        needs it.

        B is read off the input rather than taken from config, because this is
        called with two different leading sizes -- num_envs during the rollout,
        and rollout_steps * num_envs when the buffer is flattened.
        """
        batch = obs.shape[0]
        joint = obs.reshape(batch, 1, self.n_agents * self.obs_dim).expand(batch, self.n_agents, -1)
        ids = self.agent_ids.expand(batch, -1, -1)
        return torch.cat([joint, ids], dim=-1)

    def forward(self, obs):
        """(B, N, obs_dim) -> (B, N) values, one per agent."""
        return self.net(self.build_state(obs)).squeeze(-1)

    def value_from_state(self, state):
        """Values from an already-built state, as the update loop has.

        Skips rebuilding the joint observation for minibatches that stored the
        state directly.
        """
        return self.net(state).squeeze(-1)


# ---------------------------------------------------------------- value normalization

class ValueNormalizer:
    """Running mean/std over value targets, for normalizing the critic's scale.

    The reward table puts +/-100 terminal spikes next to -0.1--0.8 per-step
    terms. That dynamic range is the regime this exists for, and the MAPPO
    paper reports value normalization never hurts and often helps a lot.

    Off by default (config.use_value_norm) so the first training run is the
    simplest possible thing and a failure has one candidate cause. Flipping it
    is then a clean one-variable experiment.

    Not an nn.Module: it holds statistics, not parameters, and giving it a
    state_dict is enough for checkpointing.
    """

    def __init__(self, epsilon=1e-5, device="cpu"):
        self.epsilon = epsilon
        self.mean = torch.zeros((), device=device)
        self.var = torch.ones((), device=device)
        self.count = torch.tensor(epsilon, device=device)

    def update(self, returns):
        """Chan's parallel variance update -- exact, and batched.

        Welford one sample at a time would mean a Python loop over 40,960
        values an iteration.
        """
        batch = returns.detach().reshape(-1)
        batch_count = torch.tensor(float(batch.numel()), device=batch.device)
        batch_mean = batch.mean()
        batch_var = batch.var(unbiased=False)

        delta = batch_mean - self.mean
        total = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        new_var = (m_a + m_b + delta.pow(2) * self.count * batch_count / total) / total

        self.mean, self.var, self.count = new_mean, new_var, total

    def normalize(self, values):
        return (values - self.mean) / torch.sqrt(self.var + self.epsilon)

    def denormalize(self, values):
        return values * torch.sqrt(self.var + self.epsilon) + self.mean

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count, "epsilon": self.epsilon}

    def load_state_dict(self, state):
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]
        self.epsilon = state["epsilon"]

    def to(self, device):
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        self.count = self.count.to(device)
        return self


# ---------------------------------------------------------------- rollout storage

class RolloutBuffer:
    """Preallocated storage for one iteration of experience, plus GAE.

    Every tensor is allocated once at construction and overwritten in place
    each iteration, so a 40,960-sample iteration does no allocation in the hot
    loop.
    """

    def __init__(self, rollout_steps, num_envs, n_agents, obs_dim, action_dim, device):
        T, E, N = rollout_steps, num_envs, n_agents
        self.rollout_steps = T
        self.num_envs = E
        self.n_agents = N
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        self.obs = torch.zeros(T, E, N, obs_dim, device=device)
        self.actions = torch.zeros(T, E, N, action_dim, device=device)
        self.log_probs = torch.zeros(T, E, N, device=device)
        self.values = torch.zeros(T, E, N, device=device)
        self.rewards = torch.zeros(T, E, N, device=device)
        self.terminated = torch.zeros(T, E, N, dtype=torch.bool, device=device)
        self.truncated = torch.zeros(T, E, N, dtype=torch.bool, device=device)
        # the value of where a truncated episode actually stopped, which the
        # post-reset observation at t+1 no longer describes
        self.trunc_values = torch.zeros(T, E, N, device=device)

    def clear(self):
        """Zero everything. Only trunc_values strictly needs it -- it is written
        through a boolean mask, so a stale entry from a previous iteration would
        survive at any (t, e, n) that does not truncate this time.
        """
        for tensor in (self.obs, self.actions, self.log_probs, self.values,
                       self.rewards, self.trunc_values):
            tensor.zero_()
        self.terminated.zero_()
        self.truncated.zero_()

    def insert(self, step, obs, action, log_prob, value, reward, terminated, truncated,
               trunc_value=None):
        self.obs[step] = obs
        self.actions[step] = action
        self.log_probs[step] = log_prob
        self.values[step] = value
        self.rewards[step] = reward
        self.terminated[step] = terminated
        self.truncated[step] = truncated
        if trunc_value is not None:
            self.trunc_values[step] = trunc_value

    def compute_gae(self, last_value, gamma, gae_lambda):
        """Generalized advantage estimation over the stored rollout.

        Runs backward over time. Forward is silently wrong -- it produces
        finite, plausible-looking numbers that are not advantages.

        last_gae is an (E, N) tensor, not a scalar: environments are at
        different points in their episodes at the same t, so each one carries
        its own accumulator and the torch.where calls keep them independent.
        """
        T = self.rewards.shape[0]
        advantages = torch.zeros_like(self.rewards)
        last_gae = torch.zeros_like(self.rewards[0])              # (E, N)

        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else self.values[t + 1]
            # Truncation swaps in the cached bootstrap; termination then
            # overwrites it with zero. Termination must be applied LAST so it
            # wins if both are somehow set. compute_done already makes them
            # mutually exclusive via `& ~terminated`, but the ordering here does
            # not lean on that.
            next_value = torch.where(self.truncated[t], self.trunc_values[t], next_value)
            next_value = torch.where(self.terminated[t], torch.zeros_like(next_value), next_value)
            # A separate concern from next_value: that one answers "what number
            # do I bootstrap from", this one answers "may the GAE chain carry
            # momentum across this boundary". Same triggers, different jobs.
            next_nonterminal = (~(self.terminated[t] | self.truncated[t])).float()

            delta = self.rewards[t] + gamma * next_value - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values
        return advantages, returns

    def flatten(self, critic, advantages, returns):
        """Collapse (T, E, N, ...) to (T*E*N, ...), six aligned tensors.

        Row k of every tensor refers to the same (timestep, environment, agent).
        That holds only because all six derive from tensors with an identical
        leading shape, flattened in identical order -- nothing checks it at
        runtime, and a reshape that reorders one of them would corrupt the
        update with no error.

        Advantages are normalized across the whole batch, all agents together.
        """
        T, E, N = self.rollout_steps, self.num_envs, self.n_agents
        state_dim = critic.state_dim

        with torch.no_grad():
            # via (T*E, N, obs_dim), because build_state needs the agent axis
            # intact to concatenate a joint observation and a one-hot
            state = critic.build_state(self.obs.reshape(-1, N, self.obs_dim)).reshape(-1, state_dim)

        adv = advantages.reshape(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        return {
            "obs": self.obs.reshape(-1, self.obs_dim),
            "state": state,
            "actions": self.actions.reshape(-1, self.action_dim),
            "log_probs": self.log_probs.reshape(-1),
            "advantages": adv,
            "returns": returns.reshape(-1),
        }

    def minibatches(self, batch_size, minibatch_size, generator):
        """Yield index tensors covering a shuffled epoch."""
        order = torch.randperm(batch_size, generator=generator, device=self.device)
        for start in range(0, batch_size, minibatch_size):
            yield order[start:start + minibatch_size]


# ---------------------------------------------------------------- logging

class TrainingLogger:
    """One record per iteration, written to JSON periodically.

    Written every `write_every` iterations rather than only at the end so a run
    that crashes at iteration 180 keeps its history.

    What to actually read here:
      mean_action_std  the clearest health signal. On the validated VMAS run it
                       annealed smoothly 1.0 -> ~0.55. Collapse toward zero is
                       premature determinism; growth means the entropy bonus is
                       overpowering the policy loss.
      approx_kl,
      clip_fraction    "is the policy moving too far per update".
      value_loss       watched alongside state_dim, because that is the leading
                       suspect if it will not settle.
    """

    def __init__(self, path, write_every=10):
        self.path = path
        self.write_every = write_every
        self.history = []
        self.start_time = time.time()

    def log(self, record):
        record = dict(record)
        record["wall_time_s"] = time.time() - self.start_time
        self.history.append(record)
        if len(self.history) % self.write_every == 0:
            self.write()
        return record

    def write(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.history, f, indent=2)
        return self.path

    @staticmethod
    def format(record):
        return (f"iter {record['iteration']:>4}  "
                f"steps {record['total_env_steps']:>9}  "
                f"return {record['mean_episode_return']:>8.2f}  "
                f"succ {record['success_rate']:>5.1%}  "
                f"cap {record['capture_rate']:>5.1%}  "
                f"to {record['timeout_rate']:>5.1%}  "
                f"len {record['mean_episode_length']:>6.0f}  "
                f"pos {record['position_ratio']:>5.2f}  "
                f"peff {record['push_efficiency']:>5.1%}  "
                f"eva {record['evade_cosine']:>5.2f}  "
                f"thr {record['mean_threat_agents']:>4.2f}  "
                f"pi {record['policy_loss']:>7.4f}  "
                f"v {record['value_loss']:>9.3f}  "
                f"ent {record['entropy']:>6.3f}  "
                f"kl {record['approx_kl']:>7.4f}  "
                f"clip {record['clip_fraction']:>5.1%}  "
                f"std {record['mean_action_std']:>5.3f}")


# ---------------------------------------------------------------- trainer

class MAPPOTrainer:

    ACTION_DIM = 2      # force (x, y)

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)

        # Normal.sample() draws from the global RNG, so without this the
        # policy's exploration is irreproducible even though Env has its own
        # generator. Nothing in this file uses a *_like sampler, for the same
        # reason scenario.predator_policy avoids them.
        torch.manual_seed(config.seed)
        # a dedicated generator for minibatch shuffling, so changing the number
        # of epochs does not shift the action noise stream
        self.shuffle_generator = torch.Generator(device=config.device)
        self.shuffle_generator.manual_seed(config.seed)

        self.env = Env(config)

        self.actor = Actor(config.obs_dim, self.ACTION_DIM, config.hidden_dim).to(self.device)
        self.critic = Critic(config.obs_dim, config.n_agents, config.hidden_dim).to(self.device)

        # Separate optimizers, matching the validated version. The two losses
        # are independent, and one optimizer over both parameter sets would
        # share Adam's state scaling across networks with very different
        # gradient magnitudes.
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=config.lr)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=config.lr)

        self.buffer = RolloutBuffer(config.rollout_steps, config.num_envs, config.n_agents,
                                    config.obs_dim, self.ACTION_DIM, self.device)
        self.logger = TrainingLogger(config.log_path)
        self.value_normalizer = ValueNormalizer(device=self.device) if config.use_value_norm else None

        self.obs = None
        self.total_env_steps = 0
        self.best_win_rate = -1.0

        # episode bookkeeping, carried across iterations because an episode is
        # far longer than a rollout: at max_steps 900 and rollout_steps 128 a
        # single episode spans about seven iterations
        self.episode_return = torch.zeros(config.num_envs, device=self.device)
        self.episode_returns = deque(maxlen=100)
        self.episode_length = torch.zeros(config.num_envs, device=self.device)
        self.episode_lengths = deque(maxlen=100)
        # filled on the first collect_rollout from info["reward_terms"] keys,
        # so adding a term does not require a matching edit here
        self.episode_term_return = None
        self.episode_term_returns = None

    # ------------------------------------------------------------ rollout

    def collect_rollout(self, training_progress):
        """Fill the buffer with rollout_steps of experience.

        env.reset() is never called here. The environment auto-resets
        internally via reset_at, and resetting from the outside would throw
        away every episode still in flight.
        """
        self.buffer.clear()
        N = self.config.n_agents
        successes = captures = timeouts = episodes = 0
        spread_sum = pred_sum = push_events_sum = 0.0
        contact_cosine_sum = contact_count = 0.0
        push_signed_sum = push_abs_sum = 0.0
        agent_payload_dist_sum = 0.0
        behind_count = front_count = 0
        force_goal_sum = force_mag_sum = 0.0
        evade_cosine_sum = evade_count = 0.0
        threat_agents_sum = 0.0
        n_beh_steps = 0

        for step in range(self.config.rollout_steps):
            with torch.no_grad():
                action, log_prob = self.actor.act(self.obs)
                value = self.critic(self.obs)
                if self.value_normalizer is not None:
                    value = self.value_normalizer.denormalize(value)

            # The RAW action is stored: it is what log_prob was computed
            # against, and what the PPO ratio has to re-evaluate. Nothing in
            # physics.step clamps, so the environment gets a clamped copy --
            # storing the clamped one instead would make the ratio compare
            # log-probabilities of two different actions.
            clipped = action.clamp(-1.0, 1.0)
            # evade_cosine has to be scored on the board the action was chosen
            # against. info["world_state"] is post-step.
            pre_ws = self.env.world_state
            next_obs, reward, terminated, truncated, info = self.env.step(clipped, training_progress)

            # terminated/truncated are per-environment; the buffer is per-agent
            term_n = terminated.unsqueeze(-1).expand(-1, N)
            trunc_n = truncated.unsqueeze(-1).expand(-1, N)

            self.buffer.insert(step, self.obs, action, log_prob, value, reward, term_n, trunc_n)

            if truncated.any():
                with torch.no_grad():
                    final_values = self.critic(info["final_observation"])       # (E, N)
                    if self.value_normalizer is not None:
                        final_values = self.value_normalizer.denormalize(final_values)
                self.buffer.trunc_values[step][trunc_n] = final_values[trunc_n]

            terms = info["reward_terms"]
            if self.episode_term_return is None:
                self.episode_term_return = {k: torch.zeros(self.config.num_envs, device=self.device)
                                            for k in terms}
                self.episode_term_returns = {k: deque(maxlen=100) for k in terms}
            for name, term in terms.items():
                self.episode_term_return[name] += term.mean(-1)

            # per-step behavioural diagnostics: team spread and predator
            # distance distinguish "learning to evade" from "learning to quit",
            # and neither is visible in the return. Push events count actual
            # contact (push_reward != 0), which reward_push cannot: that one
            # mixes how often agents touch the payload with how hard they push.
            ws = info["world_state"]
            centroid = ws.agent_pos.mean(dim=1, keepdim=True)
            spread_sum += (ws.agent_pos - centroid).norm(dim=-1).mean().item()
            pred_dist = (ws.agent_pos - ws.predator_pos.unsqueeze(1)).norm(dim=-1)
            pred_sum += pred_dist.min(dim=-1).values.mean().item()
            push_events_sum += (terms["push"] != 0).float().sum(-1).mean().item()

            agent_payload = ws.agent_pos - ws.payload_pos.unsqueeze(1)
            agent_payload_dist = agent_payload.norm(dim=-1)
            goal_dir = info["scenario_state"].goal_pos - ws.payload_pos
            goal_dir = goal_dir / goal_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            contact_range = self.config.payload_halfsize.max() + self.config.agent_radius
            in_contact = agent_payload_dist <= contact_range
            contact_cosine = (
                agent_payload * goal_dir.unsqueeze(1)
            ).sum(-1) / agent_payload_dist.clamp(min=1e-6)
            contact_cosine_sum += contact_cosine[in_contact].sum().item()
            contact_count += in_contact.sum().item()
            agent_payload_dist_sum += agent_payload_dist.mean().item()
            behind, front = scenario.payload_side_masks(
                ws, info["scenario_state"], self.config.push_zone_radius)
            behind_count += int(behind.sum())
            front_count += int(front.sum())
            goalward, magnitude = scenario.payload_goalward_force(
                ws, info["scenario_state"], self.config)
            force_goal_sum += goalward.sum().item()
            force_mag_sum += magnitude.sum().item()
            evade_cos, in_danger = scenario.predator_evade_masks(
                pre_ws, clipped, self.config.predator_danger_radius)
            evade_cosine_sum += evade_cos[in_danger].sum().item()
            evade_count += int(in_danger.sum())
            threat_agents_sum += (terms["threat"] != 0).float().sum(-1).mean().item()
            if self.config.push_coef:
                push_penetration = terms["push"] / self.config.push_coef
            else:
                push_penetration = scenario.goalward_push_penetration(
                    ws, info["scenario_state"], self.config)
            # Use the same contact samples and denominator for both push
            # metrics, so signed / absolute directly measures cancellation.
            push_signed_sum += push_penetration[in_contact].sum().item()
            push_abs_sum += push_penetration[in_contact].abs().sum().item()
            n_beh_steps += 1

            self.episode_return += reward.mean(-1)
            self.episode_length += 1
            done = terminated | truncated
            if done.any():
                self.episode_returns.extend(self.episode_return[done].tolist())
                self.episode_lengths.extend(self.episode_length[done].tolist())
                self.episode_return = torch.where(done, torch.zeros_like(self.episode_return),
                                                  self.episode_return)
                self.episode_length = torch.where(done, torch.zeros_like(self.episode_length),
                                                  self.episode_length)
                for name in self.episode_term_return:
                    self.episode_term_returns[name].extend(
                        self.episode_term_return[name][done].tolist())
                    self.episode_term_return[name] = torch.where(
                        done, torch.zeros_like(self.episode_term_return[name]),
                        self.episode_term_return[name])
                # outcomes come from info, not from reward: a return of +90 is
                # not distinguishable from a timeout that happened to go well
                successes += int((terminated & info["success"]).sum())
                captures += int((terminated & info["captured"]).sum())
                timeouts += int(truncated.sum())
                episodes += int(done.sum())

            self.obs = next_obs
            self.total_env_steps += self.config.num_envs

        with torch.no_grad():
            last_value = self.critic(self.obs)
            if self.value_normalizer is not None:
                last_value = self.value_normalizer.denormalize(last_value)

        def _mean(values):
            return sum(values) / len(values) if values else 0.0

        outcomes = {
            "episodes_completed": episodes,
            "success_rate": successes / episodes if episodes else 0.0,
            "capture_rate": captures / episodes if episodes else 0.0,
            "timeout_rate": timeouts / episodes if episodes else 0.0,
            "mean_episode_length": _mean(self.episode_lengths),
            "mean_team_spread": spread_sum / n_beh_steps,
            "mean_min_predator_dist": pred_sum / n_beh_steps,
            "mean_push_events": push_events_sum / n_beh_steps,
            "mean_contact_cosine": contact_cosine_sum / contact_count if contact_count else 0.0,
            "mean_push_abs": push_abs_sum / contact_count if contact_count else 0.0,
            "mean_push_signed": push_signed_sum / contact_count if contact_count else 0.0,
            "mean_agent_payload_dist": agent_payload_dist_sum / n_beh_steps,
            "position_ratio": behind_count / max(front_count, 1),
            "push_efficiency": force_goal_sum / force_mag_sum if force_mag_sum else 0.0,
            "evade_cosine": evade_cosine_sum / evade_count if evade_count else 0.0,
            "mean_threat_agents": threat_agents_sum / n_beh_steps,
        }
        if self.episode_term_returns is not None:
            for name, values in self.episode_term_returns.items():
                outcomes[f"reward_{name}"] = _mean(values)
        return last_value, outcomes

    # ------------------------------------------------------------ update

    def update(self, flat):
        """update_epochs passes over the flattened batch, in minibatches."""
        cfg = self.config
        batch_size = flat["advantages"].shape[0]

        if self.value_normalizer is not None:
            self.value_normalizer.update(flat["returns"])
            value_targets = self.value_normalizer.normalize(flat["returns"])
        else:
            value_targets = flat["returns"]

        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                  "approx_kl": 0.0, "clip_fraction": 0.0}
        n_updates = 0

        for _ in range(cfg.update_epochs):
            for idx in self.buffer.minibatches(batch_size, cfg.minibatch_size,
                                               self.shuffle_generator):
                obs_b = flat["obs"][idx]
                action_b = flat["actions"][idx]
                old_log_prob_b = flat["log_probs"][idx]
                adv_b = flat["advantages"][idx]
                state_b = flat["state"][idx]
                target_b = value_targets[idx]

                new_log_prob, entropy = self.actor.evaluate(obs_b, action_b)
                entropy = entropy.mean()

                log_ratio = new_log_prob - old_log_prob_b
                ratio = torch.exp(log_ratio)

                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                actor_loss = policy_loss - cfg.ent_coefficient * entropy

                critic_loss = F.mse_loss(self.critic.value_from_state(state_b), target_b)

                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                # BETWEEN backward and step. After step it runs without error
                # and does nothing at all -- the optimizer has already consumed
                # the unclipped gradients.
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.optimizer_actor.step()
                self.optimizer_critic.step()

                with torch.no_grad():
                    totals["policy_loss"] += policy_loss.item()
                    totals["value_loss"] += critic_loss.item()
                    totals["entropy"] += entropy.item()
                    totals["approx_kl"] += ((ratio - 1) - log_ratio).mean().item()
                    totals["clip_fraction"] += ((ratio - 1).abs() > cfg.clip_epsilon).float().mean().item()
                n_updates += 1

        return {key: value / n_updates for key, value in totals.items()}

    # ------------------------------------------------------------ outer loop

    def train(self, verbose=True, max_iterations=None, anneal_lr=True):
        """Train for all configured iterations or stop after a shorter preview.

        max_iterations limits how many iterations execute without changing
        config.num_iterations. That distinction matters because num_iterations
        remains the denominator for training_progress and the learning-rate
        schedule, so a 30-iteration preview can represent iterations 1--30 of a
        planned 250-iteration run rather than compressing the full schedule
        into 30 iterations.

        Set anneal_lr=False for a constant-learning-rate experiment. Reward
        curriculum progress still uses config.num_iterations.
        """
        cfg = self.config
        self.obs = self.env.reset()

        stop_at = cfg.num_iterations if max_iterations is None else min(max_iterations, cfg.num_iterations)
        if stop_at < 1:
            raise ValueError("max_iterations must be at least 1")

        for iteration in range(1, stop_at + 1):
            progress = (iteration - 1) / cfg.num_iterations

            last_value, outcomes = self.collect_rollout(progress)
            advantages, returns = self.buffer.compute_gae(last_value, cfg.gamma, cfg.gae_lambda)
            team_adv = advantages.mean(-1, keepdim=True)
            adv_var = advantages.var()
            outcomes["advantage_private_fraction"] = (
                float((advantages - team_adv).var() / adv_var) if float(adv_var) > 0.0 else 0.0)
            flat = self.buffer.flatten(self.critic, advantages, returns)

            # Linear anneal uses the configured full-run horizon even when
            # max_iterations stops a preview early. It can be disabled to
            # isolate constant learning rate as a single-variable experiment.
            lr_now = cfg.lr * (1.0 - progress) if anneal_lr else cfg.lr
            for optimizer in (self.optimizer_actor, self.optimizer_critic):
                for group in optimizer.param_groups:
                    group["lr"] = lr_now

            losses = self.update(flat)

            returns_tensor = torch.tensor(list(self.episode_returns)) if self.episode_returns \
                else torch.zeros(1)
            is_final = iteration == stop_at
            eval_stats = {}
            if cfg.eval_interval and (iteration % cfg.eval_interval == 0 or is_final):
                eval_stats = self.evaluate_policy()
                if eval_stats["eval_win_rate"] >= self.best_win_rate:
                    self.best_win_rate = eval_stats["eval_win_rate"]
                    self.save(os.path.join(cfg.checkpoint_dir, "checkpoint_best.pt"), iteration)

            record = {
                "iteration": iteration,
                "total_env_steps": self.total_env_steps,
                "mean_episode_return": float(returns_tensor.mean()),
                "episode_return_std": float(returns_tensor.std(unbiased=False)),
                "mean_action_std": float(self.actor.log_std.exp().mean()),
                "learning_rate": lr_now,
                "n_agents": cfg.n_agents,
                "obs_dim": cfg.obs_dim,
                "state_dim": cfg.state_dim,
                **outcomes,
                **losses,
                **eval_stats,
            }
            self.logger.log(record)
            if verbose:
                print(TrainingLogger.format(record), flush=True)

            if iteration % cfg.checkpoint_interval == 0 or is_final:
                self.save_periodic(iteration, final=is_final)

        self.logger.write()
        return self.logger.history

    def evaluate_policy(self):
        """Blocking eval on a fresh env batch. Does not touch the training env."""
        from tools.evaluate import evaluate

        cfg = dataclasses.replace(self.config, num_envs=self.config.eval_num_envs)
        actor = self.actor
        actor.eval()

        def policy(world_state, scenario_state, config, actor=actor):
            with torch.no_grad():
                return actor(scenario.observe(world_state, scenario_state, config)).clamp(-1.0, 1.0)

        result = evaluate(cfg, policy)
        actor.train()
        return result.as_log()

    # ------------------------------------------------------------ checkpoints

    def save(self, path, iteration):
        return save_checkpoint(path, iteration, self.actor, self.critic,
                               self.optimizer_actor, self.optimizer_critic,
                               self.config, self.value_normalizer, self.logger.history)

    def load(self, path, map_location=None):
        return load_checkpoint(path, self.actor, self.critic,
                               self.optimizer_actor, self.optimizer_critic,
                               self.value_normalizer,
                               map_location or self.config.device)

    def save_periodic(self, iteration, final=False):
        """checkpoint_latest.pt every interval, numbered snapshots more rarely.

        Overwriting one latest file is what makes a frequent interval cheap;
        the numbered copies exist so a run that degrades late still has an
        earlier policy to fall back on, without leaving 250 files on disk.
        """
        cfg = self.config
        paths = [os.path.join(cfg.checkpoint_dir, "checkpoint_latest.pt")]
        keep_interval = 4 * cfg.checkpoint_interval
        if iteration % keep_interval == 0 or final:
            paths.append(os.path.join(cfg.checkpoint_dir, f"checkpoint_{iteration}.pt"))
        for path in paths:
            self.save(path, iteration)
        return paths


if __name__ == "__main__":
    config = Config()
    print(f"n_agents {config.n_agents}  obs_dim {config.obs_dim}  state_dim {config.state_dim}  "
          f"samples/iter {config.rollout_steps * config.num_envs * config.n_agents}")
    MAPPOTrainer(config).train()
