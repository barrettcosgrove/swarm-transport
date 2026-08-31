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
import pytest
import torch

from env.scenario import reset, reward_terms
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
    """DESIGN.md section 3: max_steps * |R_time| < |R_capture|.

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


def test_approach_stays_constant():
    """0.0 is a sentinel: reward_terms keeps approach_coef at start for the
    whole run. A decaying schedule taught flee-and-stay-gone. Alignment is
    not a reward term -- that job is push_coef.
    """
    config = Config()
    assert config.approach_anneal_fraction == 0.0, (
        "approach_anneal_fraction is the constant-coefficient sentinel; "
        f"got {config.approach_anneal_fraction}"
    )
    world_state, scenario_state = reset(
        1, Config(num_envs=1), torch.Generator().manual_seed(0))
    assert "alignment" not in reward_terms(world_state, scenario_state, 0.0, config)


def test_blame_fraction_is_a_fraction():
    """Outside [0, 1] the split stops being a split: above 1 the shared branch
    goes positive and paying the team for taking damage, below 0 it pays the
    agent that was actually hit.
    """
    config = Config()
    assert 0.0 <= config.health_loss_blame_fraction <= 1.0
    assert 0.0 <= config.progress_blame_fraction <= 1.0


def predator_spin_up_closure(config):
    """Net ground a predator at its speed cap makes up while an agent
    accelerates from rest against linear drag.

    Same integration as tools/threat_calibrate.py. Finite because agent
    thrust/drag is 20 against predator_max_speed 3.5 -- the agent eventually
    outruns it, so only the spin-up window decides the outcome.
    """
    a = config.agent_max_thrust / config.agent_mass
    k = config.agent_drag_coef / config.agent_mass
    v = 0.0
    closure = 0.0
    worst = 0.0
    for _ in range(200):
        v = min(v + (a - k * v) * config.dt, config.agent_max_speed)
        closure += (config.predator_max_speed - v) * config.dt
        worst = max(worst, closure)
    return worst


def test_danger_radius_clears_the_spin_up_closure():
    """An agent fleeing from rest loses ~1.209 units before it matches the
    predator's speed cap. Any danger radius below capture + that closure is a
    warning that arrives after capture is already committed -- the 1.0 radius
    sat below this floor and evasion was not a behaviour the policy could
    express.
    """
    config = Config()
    floor = config.predator_capture_radius + predator_spin_up_closure(config)
    assert config.predator_danger_radius > floor, (
        f"predator_danger_radius {config.predator_danger_radius} is below the "
        f"escape floor {floor:.3f}"
    )


def test_threat_zone_leaves_room_to_push():
    """The threat radius has to exceed the capture radius to be a warning rather
    than a synonym for the damage it is meant to pre-empt, and has to stay well
    inside the arena or it becomes the permanent repulsion DESIGN.md rejects.

    The per-step bound is threat_coef against progress_coef * push_standoff.
    At 1.0 vs 15 the margin is 15x -- still a reason to stay and push rather
    than a standing order to leave the crate. The per-step test alone missed
    variant B; test_threat_integral_stays_well_under_winning is the one that
    would have caught the accumulating cost.
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


# Fraction of agent-steps with nonzero closing-rate threat, measured by
# tools/threat_calibrate.py against variant D at radius 3.0. Production is
# radius 2.5, so this is a pessimistic envelope -- a wider field, and most
# live steps pay a fraction of threat_coef. Re-measure if danger_radius
# grows or predator_max_speed changes. The integral is an upper bound, not
# the realized episode sum.
MEASURED_THREAT_DUTY = 0.3855

# Fraction of agent-steps with nonzero camping term, measured by
# tools/threat_calibrate.py against variant D at camp_radius 0.8. Unused
# while camp_coef is 0; kept so the gated integral below has a duty if
# the term is turned back on. Re-measure whenever predator_camp_radius
# changes.
MEASURED_CAMP_DUTY = 0.4089


def test_threat_integral_stays_well_under_winning():
    """The distance-field form of variant B passed the per-step bound
    (2.0 < 15) and still dominated the return: episode threat -138 against
    success 450. Duty counts any nonzero step, so this envelope is loose;
    a win is 450 and variant B's realized -138 already inverted the
    objective. Requiring the envelope under one win is what would have
    rejected that coefficient.
    """
    config = Config()
    budget = config.max_steps * config.threat_coef * MEASURED_THREAT_DUTY
    assert budget < config.success_reward, (
        f"threat integral {budget:.1f} against success {config.success_reward}. "
        f"Lower threat_coef or re-measure MEASURED_THREAT_DUTY"
    )


def test_camp_radius_is_tighter_than_the_danger_ring():
    """The camping term is the capture-bubble tax, not a second danger field.
    It has to exceed capture_radius (otherwise it is a synonym for the hit
    it is meant to pre-empt) and stay inside 2x that and well inside the
    closing-gated ring. Wider is variant B: everyone leaves the crate.
    """
    config = Config()
    assert config.predator_camp_radius > config.predator_capture_radius, (
        "camp_radius must be a warning, not a synonym for capture"
    )
    assert config.predator_camp_radius <= 2.0 * config.predator_capture_radius, (
        f"predator_camp_radius {config.predator_camp_radius} is wider than "
        f"2x capture {config.predator_capture_radius}; that taxes the push zone"
    )
    assert config.predator_camp_radius < config.predator_danger_radius, (
        f"predator_camp_radius {config.predator_camp_radius} must stay inside "
        f"the danger ring {config.predator_danger_radius}"
    )


def test_camp_coef_is_off():
    """Variant E cannot both spare the pushers and evict a hunter on the
    crate. The term stays in reward_terms so the log key does not vanish.
    """
    assert Config().camp_coef == 0.0


def test_camp_integral_stays_well_under_winning():
    """Same envelope as threat. Vacuous while camp_coef is 0; the bound
    matters the moment someone turns it back on.
    """
    config = Config()
    if config.camp_coef <= 0.0:
        pytest.skip("camp_coef is off; integral is tautological")
    budget = config.max_steps * config.camp_coef * MEASURED_CAMP_DUTY
    assert budget < config.success_reward, (
        f"camp integral {budget:.1f} against success {config.success_reward}. "
        f"Lower camp_coef or re-measure MEASURED_CAMP_DUTY"
    )


def test_flee_coef_is_off():
    """Variant G's action bonus lost to C (hunted cosine +0.20 -> +0.09).
    Do not turn it back on in the same run as the approach-while-camping
    gate.
    """
    assert Config().flee_coef == 0.0


def test_flee_integral_stays_well_under_winning():
    """Only the hunted agent is paid, peak flee_coef per step. Vacuous
    while flee_coef is 0; if the envelope reached a win, fleeing forever
    would beat delivering the crate.
    """
    config = Config()
    if config.flee_coef <= 0.0:
        pytest.skip("flee_coef is off; integral is tautological")
    budget = config.max_steps * config.flee_coef / config.n_agents
    assert budget < config.success_reward, (
        f"flee envelope {budget:.1f} against success {config.success_reward}. "
        f"Lower flee_coef"
    )
    push_standoff = float(config.payload_halfsize.max()) + config.agent_radius
    assert config.flee_coef < config.progress_coef * push_standoff, (
        "flee term outweighs what pushing the payload can pay"
    )
