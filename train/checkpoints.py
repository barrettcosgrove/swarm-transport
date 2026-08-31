"""
train/checkpoints.py

Save and load training state as plain functions, with no dependency on
MAPPOTrainer.

Kept separate for the consumers: render.py and evaluate.py want weights and
nothing else, and constructing a trainer to get them would mean building an
environment, two optimizers and a rollout buffer just to read tensors off disk.
MAPPOTrainer calls into here periodically; the direction of the dependency
never reverses.

Usage:
    from train.checkpoints import load_checkpoint
    from train.mappo import Actor, Critic

    actor = Actor(config.obs_dim, 2, config.hidden_dim)
    critic = Critic(config.obs_dim, config.n_agents, config.hidden_dim)
    iteration = load_checkpoint("train/checkpoints/checkpoint_latest.pt", actor, critic)
"""
import dataclasses
import os

import torch


def save_checkpoint(path, iteration, actor, critic, optimizer_actor, optimizer_critic,
                    config, value_normalizer=None, history=None):
    """Write everything needed to either resume training or just run the policy.

    actor.state_dict() carries log_std and critic.state_dict() carries
    agent_ids automatically, because the first is a registered Parameter and
    the second a registered buffer. Nothing here has to know they exist.

    Both optimizer states are saved so a resumed run keeps Adam's momentum
    estimates instead of restarting cold, and the config is saved as a plain
    dict so a checkpoint records the constants it was trained under.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    payload = {
        "iteration": iteration,
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "optimizer_actor": optimizer_actor.state_dict() if optimizer_actor is not None else None,
        "optimizer_critic": optimizer_critic.state_dict() if optimizer_critic is not None else None,
        "config": dataclasses.asdict(config) if dataclasses.is_dataclass(config) else config,
        "value_normalizer": value_normalizer.state_dict() if value_normalizer is not None else None,
        "history": history,
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path, actor, critic, optimizer_actor=None, optimizer_critic=None,
                    value_normalizer=None, map_location="cpu"):
    """Restore into already-constructed modules and return the iteration reached.

    map_location defaults to cpu so a GPU-trained checkpoint loads on a laptop
    for rendering or eval. Optimizers and the normalizer are optional:
    the weights-only consumers pass neither.
    """
    payload = torch.load(path, map_location=map_location)

    actor.load_state_dict(payload["actor"])
    critic.load_state_dict(payload["critic"])

    if optimizer_actor is not None and payload.get("optimizer_actor") is not None:
        optimizer_actor.load_state_dict(payload["optimizer_actor"])
    if optimizer_critic is not None and payload.get("optimizer_critic") is not None:
        optimizer_critic.load_state_dict(payload["optimizer_critic"])
    if value_normalizer is not None and payload.get("value_normalizer") is not None:
        value_normalizer.load_state_dict(payload["value_normalizer"])

    return payload["iteration"]


def load_history(path, map_location="cpu"):
    """The training history a checkpoint was written with, or None.

    Separate from load_checkpoint so a resumed run can pick the log back up
    without the caller having to reach into the raw payload.
    """
    payload = torch.load(path, map_location=map_location)
    return payload.get("history")
