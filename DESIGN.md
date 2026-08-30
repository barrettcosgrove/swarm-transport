# Design notes

Numbers live in `train/config.py` and are asserted by `tests/test_reward_invariants.py`. This file is the reasoning — why a choice exists, what was rejected, and what each change did. The current recipe: 5 agents, 900 steps, closing-gated threat at radius 2.5, 70% progress and health blame, approach constant, camp and flee off.

See [README.md](README.md) for the running system, eval numbers, and how to train.

## 1. Intent

A custom vectorized multi-agent simulator in PyTorch, trained with MAPPO. Five agents haul a heavy crate to a goal while a scripted predator guards it. Health is a shared pool. Episodes end on delivery, capture, or timeout.

The load-bearing tension: agents must cluster to push, and clustering is what the predator punishes. If either side of that is free — a decoy that can run forever, or a crate that one agent moves as fast as five — the demo collapses.

A hand-written controller exists because a custom env fails in a specific way: training is flat and there are five places to blame (physics, reward, obs, spawn, algorithm). If the scripted policy cannot move the crate to the goal, no amount of MAPPO will.

## 2. Simulation

**Penalty forces, not impulses.** Soft springs vectorize: overlap matrix, mask, sum, integrate once. Impulses are order-dependent and need several solver passes when two agents and a crate touch at once — the common case, and a poor fit for 128 batched envs. Soft contact is also the right mechanic: this is sustained pushing, not bouncing.

**No rotation.** No angular velocity, torque, or inertia. Collision stays a tensor sum (circle–circle, circle–AABB, AABB min-penetration). A rotating box would need SAT and contact manifolds for a visual effect the demo does not need.

**Walls are thick AABBs, not a clamp and not a reward.** A physical wall keeps observations bounded. A penalty for leaving the arena burns sample budget on a lesson physics can enforce. A hard clamp makes agents stick. Thin walls failed: once a center crossed the wall midline, the nearest face was the outer one and stiffness 1200 ejected the body for good. Half-thickness 1.5 puts the midline at 10.0, 1.6 past first contact, against a capped step of `agent_max_speed × dt = 1.0`. Uncapped contact rebounds hit ~59 units/s; the agent cap is 20.

**Heavy payload + drag, not a friction threshold.** A μ-threshold is a discontinuity. With mass 5 and drag 0.1, one agent can move the crate slowly and two move it faster. Cooperation is a gradient, not a binary.

**Shared health, not death.** Death plus a time penalty is a suicide exploit: ending the episode early is a reward. Removing an agent mid-episode breaks the fixed observation width. A shared pool of 150, drained 10 per hit, keeps the count fixed and gives a smooth “closer is worse” signal. Capture radius 0.4 is larger than body touch (0.25) so a fast agent cannot step clean through the predator.

**Predator is a guard, and slower than the agents.** Target = argmin(`dist_to_predator + 0.5 × dist_to_payload`). Pure pursuit has a free decoy: one agent runs away forever, the rest push. A guard lives on the crate, so a decoy has to work near it. Speed 3.5 vs agent 20 (1.75 on cooldown) is the evasion window — at 6.0 the predator closed faster than a clump could learn to thrust. Cooldown is a speed cap, not a thrust cut, so a cooling hunter still turns. OU noise (decay 0.95, σ 0.1) stops the policy overfitting a deterministic opponent.

Per-step argmin switched targets ~73 times an episode: the moment an agent fled it stopped being nearest, and the predator turned on whoever stayed to push. A 20-step lock-on made fleeing a real behavior.

**Relative vectors, not lidar.** Identity matters (teammate vs predator). Lidar returns a distance. Positions / 8.5, velocities / 5.0, health / 150. Raw health 0–150 flipped agents between flee and suicide. Cooldown is not observed — agents already see predator pos/vel, and adding the bit (variant F) lost the basin. Actor is local, 32-dim; critic is 165-dim (5×32 + one-hot id). The one-hot is load-bearing: without it a deterministic critic must return the same value for every agent, and approach/push/collision are private.

**Spawn is valid by construction.** Payload → agents (annulus 1.2–2.4) → predator (r=3.6) → goal (r=6.3) → four obstacles in a 6-sector polar ring (band 3.5–4.3, corridor 0.35 rad, min gap 0.75). No rejection loop, no scipy reachability. Gaps are structural: nothing spaced that way can seal a region. Obstacles are allowed on the payload–goal line; that is the point of having them.

## 3. Reward

Each term buys a behavior. Shared terms barely move an individual's gradient; private terms are what the policy actually follows.

**Get to the crate → private approach.** Telescoping delta to the standoff behind the payload. An agent hovering at the crate closes nothing and earns nothing. This is not an anneal-off crutch: a 400-run that faded approach at 0.6 of training finished at 12.5% win / 87.5% timeout (flee and stay gone). Constant +8 taught return-to-the-box.

**Stay on it and drive it → private push + progress blame.** Approach goes silent once the standoff is held, because the target translates with the crate. Push pays this agent's goalward contact penetration — zero without overlap, so it cannot be farmed by hovering. Progress is a team outcome; a wholly shared term is why no agent learned side selection (pusher and blocker got the same number). 70% is billed to agents inside radius 0.5 and *split* among them so the team total stays the old shared term.

**Don't ignore the hunter, don't abandon the crate → health blame + closing-gated threat.** `update_health` drains on `any()`, so a shared health penalty does not move when one agent retreats. Measured `cos(action, toward predator)` was +0.087: chase the shaping, treat health as weather. The private share is *flat*, not split — splitting would shrink an agent's bill as teammates joined it in the danger zone. Threat is `−intrusion² × closing` inside radius 2.5, and identically zero when the predator is parked. An always-on distance field is a standing reason to leave the crate the predator guards (variant B: episode threat −138 against progress ~0, payload abandoned).

**Win > die > wait.** Clock 90 < full drain 150 < capture 250 < win 450. If the clock costs more than dying, a policy that cannot win is right to suicide. At 900 × 0.22 the bound had silently flipped (198 > 150). Tests assert it.

**Deliberately omitted.** No global predator-distance tax (same failure as ungated threat). No agent–agent collision penalty (pushing requires clustering). Alignment as a standing cosine telescoped to ~0 and could not teach a standing push — that job is `push_coef`. Camp and flee terms stay in the log with coefficient 0 so a key never vanishes.

## 4. What shipped and what failed

Production is variant **C + A**: closing-gated threat, progress blame, approach on, camp off, flee off. Seven seeds × 400 iterations. Demo checkpoint: seed 4, **67% win / 11% capture**, position ratio **3.6** (starts near 1.0), push efficiency **41%**. Across seeds: 40–67% win, mean ~53%.

**Shipped**

| Change | Why | Result |
|---|---|---|
| Progress blame 70% | Shared progress hid who was pushing | Control 37–41% / 34–47% cap → A 52–59% / 19–22% |
| Closing-gated threat r=2.5, coef 1.0 | Always-on tax emptied the crate | C 58–64% / 14–21%; seed 4 67% / 11% |
| Approach constant | Anneal taught flee-and-stay-gone | Annealed run 12.5% win / 87.5% timeout |
| Health blame 70% flat | Shared health could not teach evade | Hunted agents start fleeing instead of drifting in |
| LR anneal 3e-4 → 0, value norm on | ±450 terminals next to −0.1 steps | Value loss settles; late training stays stable |
| `checkpoint_best.pt` + 7 seeds | Last iterate is not the best board | Seed 4 best eval 70.5% at iter 300 |

**Rejected**

| Variant | Change | What happened |
|---|---|---|
| B | Ungated threat, r=3 | 8–32% win. Agents left the crate |
| D | Danger radius 3.0 | Seed lottery 37–67% |
| E | camp_coef=0.35, r=0.8 | Cannot spare pushers and evict a hunter on the crate |
| F | Cooldown in the observation | Lost C's basin |
| G | Hunted flee-action bonus | Episode flee +1–2 against progress +250; hunted cosine worse than C |
| H | Zero approach while predator camps | Push collapsed |

Threat was sized against the time penalty, not against progress RMS. Progress is zero-mean and telescopes; threat accumulates. Sizing B against “6.5% of progress” is how it summed to −138 / episode.

The scripted controller is the solvability check and the eval baseline. Behavioral cloning was considered and not needed. Logging grew with the bugs: outcome split, per-term sums, position ratio, push efficiency, hunted cosine, camping time. Render for eyes; JSON for structure. Most training ran on a local CPU; `Config.device` is the only switch.

## 5. Training

MAPPO: shared actor on local obs, centralized critic on the joint plus agent id. γ=0.999 because episodes are 900 steps. 128 envs × 128 steps × 5 agents = 81,920 samples/iteration. Value normalization is on because the ±450 terminals next to −0.1-scale steps are the regime it exists for.

A learned predator is deferred. It needs its own actor and its own critic — a single global value is meaningless across an adversarial team boundary — and it adds two-timescale instability on top of ordinary MAPPO non-stationarity. The scripted guard is the opponent the current recipe was tuned against.
