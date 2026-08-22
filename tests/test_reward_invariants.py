"""
tests/test_reward_invariants.py

Arithmetic the reward constants have to satisfy for the task to be winnable and
for winning to be what the return actually favours. Pure config arithmetic --
no rollouts, no network.

These exist because the constraints used to live only in DESIGN.md prose, and
prose does not fail CI. The time-penalty inequality was derived against a
250-step episode with a -100 capture, then max_steps grew to 900 and the bound
tightened from 0.4 to 0.28 with nothing to notice that 0.22 had stopped being
safe -- at 900 * 0.22 = 198 against 150, ending the episode early was strictly
better than running out the clock, and the policy learned exactly that.

Run with: pytest tests/test_reward_invariants.py -v
"""
import torch

from train.config import Config


# Payload cruising speed in units/sec under a sustained single-agent push,
# measured by tools/calibrate.py. Re-measure and update this whenever dt,
# stiffness, mass, drag or thrust change -- it is the bridge between the
# physics constants and the survival budget below.
MEASURED_PUSH_SPEED = 0.37


def traversal_steps(config):
    """Steps a single agent needs to shove the payload from spawn to the goal."""
    return config.goal_radius / MEASURED_PUSH_SPEED / config.dt


def survival_budget_steps(config):
    """Steps between the predator's first hit and the health pool emptying.

    One damage event per cooldown, so the budget is events * cooldown. Slightly
    pessimistic: update_health sets the cooldown to its full duration and then
    decrements once per step, so events are actually cooldown + 1 apart.
    """
    events = config.max_health / config.health_loss_per_step
    return events * config.predator_cooldown_duration


def test_dying_costs_more_than_running_out_the_clock():
    """DESIGN.md section 11: max_steps * |R_time| < |R_capture|.

    Violate this and suicide is not a quirk of exploration, it is the correct
    play for any policy that cannot reliably reach the goal -- and every policy
    starts out unable to. Asserted with headroom rather than as a bare
    inequality: at parity the two outcomes are worth the same, and the shaping
    terms decide.
    """
    config = Config()
    clock = config.max_steps * config.time_penalty_coef
    assert clock < 0.6 * abs(config.captured_reward), (
        f"running the clock costs {clock:.1f} against {abs(config.captured_reward):.1f} "
        f"for being captured, so ending the episode early is the better outcome. "
        f"time_penalty_coef must stay under "
        f"{0.6 * abs(config.captured_reward) / config.max_steps:.3f} at max_steps="
        f"{config.max_steps}"
    )


def test_capture_is_the_worst_outcome():
    """The ordering DESIGN.md asks for: clock < full health drain < capture.

    Capture has to be worse than the drain that leads to it, or the terminal
    penalty is not what deters the behaviour draining the pool.
    """
    config = Config()
    clock = config.max_steps * config.time_penalty_coef
    drain = config.health_loss_coef * config.max_health
    assert clock < drain < abs(config.captured_reward), (
        f"expected clock {clock:.1f} < health drain {drain:.1f} < capture "
        f"{abs(config.captured_reward):.1f}"
    )


def test_survival_budget_outlasts_a_traversal():
    """The team must be able to survive long enough to finish the job.

    Nothing enforced this before, and it was false: 260 steps of health against
    ~340 to reach the goal meant every episode ended in capture no matter how
    well the payload was pushed. A policy cannot learn to protect a health pool
    that buys it nothing, so this is upstream of every reward-shaping question.

    1.5x rather than 1.0x because a real episode is not a straight-line shove:
    obstacles, evasion detours and imperfect push angles all cost steps, and the
    measured speed is an unobstructed best case.
    """
    config = Config()
    budget = survival_budget_steps(config)
    needed = traversal_steps(config)
    assert budget > 1.5 * needed, (
        f"survival budget {budget:.0f} steps against {needed:.0f} for a traversal. "
        f"Raise max_health or predator_cooldown_duration, or shorten the task"
    )


def test_episode_is_long_enough_to_finish():
    """max_steps has to leave room for a traversal too, not just the health pool."""
    config = Config()
    assert config.max_steps > 1.5 * traversal_steps(config), (
        f"max_steps {config.max_steps} against {traversal_steps(config):.0f} "
        f"steps for a traversal"
    )


def test_shaping_coefficients_anneal_to_zero():
    """coef = start * (1 - min(progress / fraction, 1)), so a fraction above 1.0
    never reaches zero and leaves a floor of start * (1 - 1/fraction).

    Both fractions sat at 5.0, which pinned an 8.0 coefficient at 6.4 forever.
    Approach is an exploration crutch that pulls agents onto the standoff
    point behind the payload. 0.0 is a legal sentinel meaning "do not anneal"
    (the 250-run that kept it on taught return-to-the-box). A value in (0, 1]
    still completes; anything above 1.0 is the old floor bug. Push is not
    annealed: it is the term that pays during sustained contact.
    """
    config = Config()
    assert config.approach_anneal_fraction == 0.0, (
        "approach_anneal_fraction is the constant-coefficient sentinel; "
        f"got {config.approach_anneal_fraction}"
    )
    fraction = config.alignment_anneal_fraction
    assert 0.0 < fraction <= 1.0, (
        f"alignment_anneal_fraction={fraction} never completes: the "
        f"coefficient bottoms out at {100 * (1 - 1 / fraction):.0f}% of "
        f"its starting value"
    )


def test_blame_fraction_is_a_fraction():
    """Outside [0, 1] the split stops being a split: above 1 the shared branch
    goes positive and paying the team for taking damage, below 0 it pays the
    agent that was actually hit.
    """
    config = Config()
    assert 0.0 <= config.health_loss_blame_fraction <= 1.0


def test_threat_zone_leaves_room_to_push():
    """The threat radius has to exceed the capture radius to be a warning rather
    than a synonym for the damage it is meant to pre-empt, and has to stay well
    inside the arena or it becomes the permanent repulsion DESIGN.md rejects.
    """
    config = Config()
    assert config.predator_danger_radius > config.predator_capture_radius
    # an agent pushing the payload sits about this far from its center; the
    # threat term must not make that position unaffordable on its own
    push_standoff = float(config.payload_halfsize.max()) + config.agent_radius
    worst_case_per_step = config.threat_coef
    assert worst_case_per_step < config.progress_coef * push_standoff, (
        "threat term outweighs what pushing the payload can pay"
    )
