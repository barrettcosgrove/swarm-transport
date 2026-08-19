DESIGN.md
Multi-agent cooperative transport with an adversarial predator. Trained with MAPPO in PyTorch, deployed to the browser via ONNX and a TypeScript physics reimplementation.

This document is the source of truth for the environment specification. It records decisions and their reasoning, not just conclusions — the reasoning is the part that's hard to reconstruct later.

1. Project overview
Three homogeneous agents cooperatively push a heavy payload to a goal while a predator guards the objective. Agents share a health pool that drains under predator contact; the episode ends in success (payload reaches goal) or failure (health depleted or time expires).

End state: an interactive browser demo where trained policies run client-side via ONNX Runtime Web, physics is reimplemented in TypeScript, and rendering is handled by pixi.js.

Deployment constraint that shapes everything: each agent runs its own ONNX model with only local perception. The actor's input width is fixed at deployment and cannot include global state. This is why the critic is training-only and why observation design matters more than it would in a pure-research setting.

2. Build phases
Each phase has a definition of done. Do not begin a phase until the previous one meets it.

Phase	Deliverable	Done when
0	Design lock	This document exists; zero code written
1	Physics core	Circles collide correctly; all four physics tests pass
2	Env API + scripted controller	A hand-written controller solves the transport task
3	Single-agent RL	PPO reliably solves single-agent transport with a light payload
4	Multi-agent transport	MAPPO with 3 agents measurably beats the single-agent baseline
5	Scripted predator	Agents learn to evade while still transporting
5b	Learned predator (optional)	PPO-trained predator; adversarial training stable
6	Browser	TS physics matches Python golden trajectory within tolerance
7	Stretch	Restricted observation radius, variable agent count, richer obstacles
Why Phase 2 exists and must not be skipped: custom-environment RL fails in a specific way — you build everything, training doesn't work, and you cannot tell whether the bug is in physics, reward, observation, spawn logic, or the algorithm. Five candidate causes, no isolation. A hand-written controller answers a question RL cannot: is this task actually solvable with these physics and this reward? If the scripted controller can't move the payload to the goal, no amount of training will.

Phase 1 internal build order
Do not implement the full collision matrix at once. Isolate one variable at a time:

State representation and integration, no collisions at all. Position/velocity tensors with a batch dimension, force accumulation, semi-implicit Euler, drag. Catches batching bugs before collision logic exists to blame instead.
Circle–circle collision only.
Circle–box, static (walls first, then obstacles).
Circle–box, dynamic (agent pushing payload — first real force transfer between moving bodies).
Box–box (payload against wall/obstacle).
Predator, health, reward, and spawn logic come after all of this.

3. Entities
Dynamic bodies
Entity	Shape	Count	Notes
Agent	Circle	3 (fixed)	Homogeneous, shared policy
Predator	Circle	1	Larger than agents; modestly faster (~1.2–1.4×)
Payload	Box (non-rotating)	1	Heavy, high drag
Static geometry
Entity	Shape	Notes
Walls	4 axis-aligned boxes	Bound the arena
Obstacles	Axis-aligned boxes	Optional; Phase 5+
Goal	Position only	No physics body
Decision: no rotation anywhere in the system. No angular velocity, no torque, no moment of inertia. This is the single highest-leverage simplification available — it eliminates SAT collision, contact manifolds, and inertia tensors, all of which would also need porting to TypeScript.

Decision: payload is a non-rotating box. Chosen for visual/behavioral similarity to VMAS transport. Known tradeoff, recorded honestly: a circle payload's contact force always points through its center, so pushing off-center makes it veer — meaning a single agent pushing badly sends it sideways while two coordinated agents cancel each other's deflection and drive it straight. That's an emergent cooperation incentive that costs zero extra code. A non-rotating box loses most of this: when an agent contacts a flat face, the contact normal is that face's fixed direction regardless of where along the face the push lands, because the box cannot rotate to reveal the difference. The effect only survives near corners. Revisit if single-agent pushing feels too forgiving.

Decision: obstacles are axis-aligned boxes, not circles. This is the one deliberate exception to "all circles." Circles are a poor primitive for walls — you'd chain a dozen to make one barrier, and pairwise cost grows quadratically for something that never moves. Circle-vs-AABB is nearly as simple as circle-vs-circle, and static + axis-aligned means no rotation, no inertia, no box-box case for obstacles, and no reaction force back onto the obstacle.

4. Physics
Integration
Semi-implicit Euler. Forces accumulate per body per step: action thrust, linear drag (-k * velocity), and collision penalty forces. Then vel += (force / mass) * dt, pos += vel * dt.

Collision model: penalty forces, not impulses
Decision: soft penalty (spring) forces.

Verified reference — VMAS uses exactly this, not impulses: softplus-smoothed penetration depth times a stiffness constant, with COLLISION_FORCE = 100, contact_margin = 1e-3, drag = 0.25, dt = 0.1, substeps = 1, default sphere radius 0.05.

Reasoning:

Vectorizes with no iteration. Compute an (E, N, N) pairwise overlap matrix, mask non-overlapping pairs to zero, sum forces per body, integrate once. Impulse resolution is order-dependent and needs multiple solver passes when three bodies touch simultaneously — which is exactly what happens when two agents push a payload. Batching an iterative solver across thousands of environments is genuinely hard engineering; batching a force sum is a one-liner.
Reference implementation available. VMAS's _get_constraint_forces is readable, verified code.
Pushing works naturally. Soft contact suits sustained pushing, the core mechanic. Impulses suit sharp bouncing, which this project doesn't need.
Known costs: visible overlap between bodies (cosmetic; can be masked in the browser renderer with slight deformation), and jitter with stiff contacts if dt is too large. Tuning knobs are stiffness, dt, and substeps.

Collision matrix
Pair	Routine
agent–agent, agent–predator	circle–circle
agent/predator–wall, agent/predator–obstacle	circle–box (static, no reaction on box)
agent–payload, predator–payload	circle–box (dynamic, reaction applies to box)
payload–wall, payload–obstacle	box–box (AABB, resolve along axis of minimum penetration)
Everything stays axis-aligned, so box–box is interval overlap per axis plus a push-out along whichever axis has less penetration. No SAT anywhere.

Circle–AABB collision
Clamp the circle's center componentwise into the box's extents. The result is the closest point on the box — automatically landing on a face or corner with no case analysis:

closest = clamp(center, box_min, box_max)
delta   = center - closest
if |delta| < radius:
    penetration = radius - |delta|
    normal      = delta / |delta|
Roughly six lines, batched as (E, n_circles, n_boxes). The same routine handles arena walls by modeling the boundary as four boxes.

Edge case to handle defensively: if a circle's center ends up inside a box, delta is zero and the normal is undefined. With adequate stiffness this shouldn't occur, but guard it — push out along the axis of least penetration rather than dividing by zero.

Boundaries
Decision: hard walls via the same penalty-force machinery. No separate boundary reward penalty.

Reasoning: a physical wall means agents cannot leave, so observations stay bounded — an unbounded state space is genuinely bad for a network, since position features drift outside the range early layers were calibrated on. A reward penalty instead means early random policies burn real sample budget wandering outside the arena learning a lesson physics could enforce for free, and it adds another coefficient to tune for no benefit. A hard position clamp is the other tempting option, but you'd have to zero the velocity component manually or agents visibly stick to walls.

VMAS does this with x_semidim / y_semidim.

5. Cooperation mechanism
Decision: heavy payload mass + high linear drag + per-step time pressure.

Not a static friction threshold ("payload doesn't move until total force exceeds μ"). Discontinuities are hard to learn through. With mass and drag, one agent can move the payload but slowly; two move it meaningfully faster. Cooperation emerges because it's faster, not because it's binary — smooth gradient throughout.

The central tension: agents must cluster to push effectively, but clustering makes them collectively vulnerable to the predator. This is the heart of the project and the thing that makes the demo compelling.

6. Predator
Phase 4–5: scripted guard
Decision: guard, not chaser. Target selection is a fixed formula, not a learned policy:

target = argmin over agents of ( distance_to_predator + w * distance_to_payload )
With w = 0 this is pure pursuit; increasing w makes the predator defend the payload.

Reasoning: pure pursuit has a degenerate counter-strategy — one agent runs away forever as a permanent decoy while the other two push freely, which makes the central tension evaporate. With guard targeting, agents cannot avoid the predator by staying away, because the thing they need to touch is where the predator lives. A decoy must operate near the payload to draw the predator off it, which means the decoy is genuinely at risk. Role-splitting becomes an emergent strategy with real cost rather than a free exploit.

Cooldown
After dealing damage, the predator slows (~50% max speed) and cannot deal damage for a fixed duration. Prevents permanent pinning of one agent. Implemented as a single scalar per environment:

python
is_cooling_down     = predator_cooldown > 0
effective_max_speed = where(is_cooling_down, max_speed * 0.5, max_speed)
can_deal_damage     = ~is_cooling_down
predator_cooldown   = where(just_dealt_damage, cooldown_duration,
                            clamp(predator_cooldown - 1, min=0))
Fully vectorized, no per-environment branching.

Noise
Decision: correlated (Ornstein-Uhlenbeck) noise on steering, not per-step Gaussian jitter.

Plain per-step noise is high-frequency and averages out over a few steps — a policy learns to ignore it. Correlated noise drifts rather than jumping, so it's harder to predict-and-average-around:

python
noise = decay * noise + randn_like(noise) * sigma   # decay ~0.95
target_direction = target_direction + noise
Also reads better visually: "the predator overcorrects and wanders slightly" rather than "the predator vibrates."

Purpose: prevent the policy from overfitting to an exactly deterministic opponent, which looks great in training and is brittle everywhere else.

Phase 5b: learned predator (deferred)
Barrett's stated preference: the last change before starting the browser version.

What this reopens, recorded so it isn't a surprise:

A second actor network with its own observation and action space, trained simultaneously.
A second, separate centralized critic. The MAPPO critic stops being well-defined across a team boundary — a single "global value" is meaningless when one side's gain is the other's loss.
Genuine two-timescale adversarial instability. Unlike ordinary MAPPO non-stationarity (teammates adapting toward an aligned objective), this is adversarial: cycling and oscillation are common failure modes, not edge cases.
Deliberately sequenced after Phase 4 so a MAPPO bug and an adversarial-training instability never appear in the same debugging session with no way to distinguish them.

7. Capture mechanic
Decision: shared health pool. Not agent death, not instant episode termination.

Health drains while any agent is inside the predator's capture radius. Episode ends at zero health with a large terminal penalty.

Reasoning — each rejected alternative has a specific failure:

Death on capture creates a suicide exploit. If per-step reward is net negative (time penalty), ending the episode early is a reward. Agents can learn to throw themselves at the predator. This is a real, classic RL bug.
Removing an agent mid-episode breaks the fixed observation shape, forces death-masking into MAPPO, and creates variable-length inputs in ONNX.
Shared health keeps agent count fixed forever, gives a smooth gradient of "closer to the predator is worse" rather than a cliff, and has no early-termination exploit.
Visually this still reads well — flash an agent red while it's taking damage.

8. Observations
Decision: explicit relative vectors, not lidar.

Reasoning: lidar's advantages are handling arbitrary obstacle geometry and scaling to unknown entity counts. This environment has ~6 known entities and (initially) no obstacles, so lidar's cost — ray-circle intersection per ray per entity — buys nothing. Worse, lidar destroys identity: a ray returns a distance, not whether it hit a teammate or the predator. Recovering that requires separate lidar sets per entity class, multiplying cost again. Relative vectors are cheap (subtraction), exact, and preserve identity.

Per-agent observation contents
Field	Dim	Frame
Own position	2	Absolute (needed for walls)
Own velocity	2	Absolute
Goal offset	2	Relative to self
Payload offset	2	Relative to self
Payload velocity	2	Relative or absolute
Predator offset	2	Relative to self
Predator velocity	2	Relative or absolute
Other agents (×2)	8	Relative offset + velocity each
Shared health	1	Normalized
Time remaining	1	Normalized
Everything relative except own absolute position. Other agents use fixed index ordering (agent 0 sees agents 1, 2; agent 1 sees agents 0, 2) — standard for parameter-shared policies at this scale, and what VMAS does.

Phase 4: full observability. Get it working first.

Phase 7 ablation: restricted vision radius. Partial observability of the actor is precisely the setting a centralized critic is designed for, and it would finally make the critic earn its keep in a way navigation never did. Done as a second step it becomes a clean ablation: train with full observability, then with radius r, measure the gap, and check whether MAPPO's advantage over IPPO widens as observability shrinks.

Implementation detail if restricting: do not simply zero out the relative vector for unseen agents — zeros are ambiguous with "teammate is exactly at my position." Include an explicit visibility bit per agent alongside the zeroed vector.

If obstacles are added later: bolt on a small obstacle-only lidar. Clean hybrid — use lidar only where its actual advantage (arbitrary geometry) applies.

9. Spawn construction
Principle: valid by construction, no rejection loops. Each step's sampling zone is defined so its constraint holds automatically.

Order
python
# 1. Payload — arena center + jitter. Everything else is defined relative to it.
payload_pos = arena_center + small_jitter()

# 2. Agents — annulus around payload.
#    Inner radius keeps them off the payload; outer keeps them clear of walls.
agent_angles = uniform(0, 2*pi, size=n_agents)
agent_pos    = payload_pos + annulus_radius(agent_angles)

# 3. Predator — free angle, radius strictly beyond the agent annulus.
theta         = uniform(0, 2*pi)
predator_pos  = payload_pos + predator_radius * [cos(theta), sin(theta)]
# predator_radius > r_agent_max + margin  ->  clearance from EVERY agent guaranteed

# 4. Goal — same angle, further out, small jitter.
goal_theta = theta + uniform(-jitter, jitter)
goal_pos   = payload_pos + goal_radius * [cos(goal_theta), sin(goal_theta)]
# goal_radius > predator_radius  ->  predator between payload and goal guaranteed

# 5. Obstacles — grid slots in a band, filtered by exclusion buffers.
Why the angle is sampled before the goal exists: "predator must be between the agents and the goal" appears to need the goal's position. Flipping which quantity is free and which is derived removes the dependency — sample a direction first, place the predator along it, then place the goal further out along the same direction with jitter.

Why this eliminates the last correction step: an earlier version needed a post-hoc nudge (place predator, check distance to every agent, correct if too close). Choosing the predator's radius to exceed the agent annulus's outer radius plus margin makes "predator isn't too close to any agent" structurally true regardless of which angles agents landed at. No per-agent check needed.

Obstacle placement
Candidate positions are grid slots within a band — from just outside the agent annulus to just inside the goal ring. A band rather than a fixed-radius ring, so layouts vary between episodes.

Two filters:

Exclusion buffers. Discard any slot within entity_radius + obstacle_half_size + margin of any already-placed entity. Small distance matrix (few slots × ~6 entities), trivially cheap. Guarantees obstacles never spawn overlapping an entity.
Minimum clearance gap (see below).
Buffers stay local to each entity's footprint — they deliberately do not extend as a blanket corridor between payload and goal. Obstacles should be allowed to sit on the direct path; that's what forces interesting detours.

Clearance rule — the key guarantee
Decision: enforce a minimum surface-to-surface gap between every obstacle pair, and between every obstacle and every wall, equal to the payload's required passage width plus margin.

gap = center_distance - radius_1 - radius_2      # surface to surface, NOT center to center
This is a hard geometric guarantee, not a heuristic: no arrangement of obstacles spaced this way can seal off a region, because sealing requires some gap to be too narrow, and that's been ruled out everywhere at once.

Gap width is a design decision, not just an engineering one:

Narrow (payload_diameter + margin): agents must trail single-file behind the payload, force transmitting agent-to-agent-to-payload through a contact chain. Recommended — the queuing constraint is tactically interesting, and a narrow chokepoint becomes a natural predator ambush point, which is a legitimate and hard-to-counter threat.
Wide (payload_diameter + 2 * agent_diameter): agents can flank the payload even in corridors.
Vary gap widths across pairs rather than fixing every gap at the minimum — some tight chokepoints for tension, some open stretches for maneuvering.

Perimeter margin: maintain a clear ring around the entire arena boundary (no obstacles within some distance of a wall). Two benefits — topologically, the payload can always detour around the outside of any interior obstacle cluster, so interior obstacles can never fully separate two points as long as the boundary ring is unbroken. Tactically, this is a permanent "coward's route": always available, always slower, always more exposed to the guard predator's interior position. It doubles as the agents' fallback evasion corridor.

Reachability check (retained, secondary)
Even with the clearance rule as the primary guarantee, keep the connectivity check — reframed as insurance against implementation bugs (sign flip in the gap formula, off-by-one in which pairs get checked, a forgotten obstacle-wall case), not against the geometric argument being wrong.

Method: rasterize the arena into a coarse grid, mark cells within payload_radius + margin of any obstacle as blocked, then run connected-component labeling. Reachable iff payload's cell and goal's cell share a label.

python
from scipy.ndimage import label
labels, _ = label(free_grid)
reachable = labels[payload_cell] != 0 and labels[payload_cell] == labels[goal_cell]
Measured cost on a representative 40×56 grid (2,240 cells): scipy.ndimage.label = 72 µs; hand-written BFS = 2,633 µs. Even 4,096 environments resetting simultaneously is under 300 ms, once. Not worth optimizing.

Use 4-connectivity (scipy's default). Verified: two cells touching only diagonally are correctly reported as 2 separate components. This is physically correct — the gap at a corner-touch is exactly zero width, so no body with real radius can pass through, no matter how small.

Bonus from label: it returns every component in one call, so verifying each agent can also reach the payload is just labels[agent_cell] == labels[payload_cell] afterward, at no extra cost.

Seeding
Hold a torch.Generator rather than seeding the global RNG, so environment randomness is isolated from policy sampling. Store a per-environment seed to allow replaying a specific episode.

Browser gotcha: do not attempt to reproduce a Python-generated layout bit-for-bit in TypeScript. PyTorch's RNG stream will not match anything in JavaScript, and forcing it is a rabbit hole. Reimplement the same structural sampling rules in TS with a simple seeded PRNG (xorshift or LCG, ~10 lines) and match the distribution, not the stream. The policy doesn't care how the layout was generated. Exact agreement is only required for the physics step, which the golden-trajectory test covers.

10. Arena sizing
Decision: do not fix a number. Fix a ratio, and calibrate in Phase 2.

Reasoning: VMAS units (0.1 radius, 1.0 semidim) are arbitrary until mass, drag, and force scale exist. Any arena size chosen now would be a guess wearing a number's clothing, wrong the moment physics is calibrated.

Procedure:

Build physics with placeholder units (Phase 1).
Run the scripted controller pushing the payload alone, straight line, sustained max effort. Measure actual cruising speed once drag and thrust reach equilibrium — a real number in your units.
Target: unassisted single-agent straight-line traversal from payload start to goal should consume 50–70% of the episode's time budget (max_steps × dt).
Set the goal-ring radius to hit that target given the measured speed.
Why this ratio specifically: a single agent should just barely make it alone if nothing goes wrong. That keeps both mechanisms load-bearing simultaneously — cooperation is meaningfully faster and buys margin for evasion detours, and the predator's interference is what actually threatens the timer rather than distance alone. Too large an arena makes solo transport impossible even with zero interference, teaching agents to give up before learning anything. Too small makes cooperation irrelevant.

Reference point: VMAS navigation uses a 2×2 arena with agent radius 0.1 — 10 agent diameters across. Deliberately tight, because it has no time pressure or evasion requirement.

11. Reward
Anchor: R_success = 100. The absolute scale is arbitrary — only ratios matter, since PPO's advantage normalization washes out absolute magnitude. All reasoning is in cumulative-over-episode terms, which is the currency the constraints are actually stated in.

Term	Scope	Value	Cumulative	Rationale
Success bonus	terminal, shared	+100	+100	Anchor
Capture failure	terminal, shared	−100	−100	Most negative outcome
Time penalty	per-step, shared	≈ −0.3	≈ −75 over 250 steps	Small, but sum stays under capture
Health loss	per HP, shared	≈ −0.8	≈ −80 if 100 HP fully drained	Between time and capture
Payload progress	per unit distance, shared	0.7 * R_success / D	≈ +70	Second-largest positive, under success
Payload proximity	per unit closed this step, private	starts ≈ progress coef, annealed to 0 over first ~30% of training	—	Exploration crutch only
Push alignment	per unit of cosine gained this step, private	≈ time penalty, annealed on the same schedule as proximity	—	Orients agents already converging
Obstacle/wall collision	per-step, private	≈ −0.1	—	Minor deterrent
Predator threat	per-step inside danger radius, private	≈ −0.3 at contact, 0 beyond the radius	—	Imminent-hit warning; gated so it is not a standing repulsion
Constraint chain holds: 75 < 80 < 100 and 70 < 100.

Notes:

Time penalty derives from an inequality: max_steps × |R_time| < |R_capture|. If the clock costs more than dying, a policy that cannot win is right to end the episode early, and every policy starts out unable to win. The original derivation used 250 steps and R_capture = 100, giving |R_time| < 0.4. Those numbers moved (max_steps is 900, capture is −250) and the bound moved with them: |R_time| must stay under 250/900 ≈ 0.28, and 0.10 leaves the same kind of headroom the original 0.3 did. The inequality is now asserted by tests/test_reward_invariants.py, so it cannot silently stop applying the next time a constant is retuned.
Health loss is sized via k × health_pool rather than k × drain_rate, so the cumulative penalty depends only on total HP lost, not on how fast the predator happens to drain it. Keeps the number meaningful regardless of later predator tuning. Structurally, health loss is a dense signal building toward a sparse terminal penalty — the exact mirror of progress's relationship to the success bonus. The drain itself is still a shared pool gated on any() over agents (one event costs 10 HP whether one agent or five are in range); per-agent responsibility is applied at reward time, billing health_loss_blame_fraction of the event to the agents actually inside the capture radius and sharing the rest. The private share is flat, not split among the culprits, so crowding into the danger zone cannot dilute an individual's bill.
Payload progress cannot get a final number until D (payload-to-goal distance) is known from Phase 2 calibration. The ratio is fixed now; the coefficient follows.
Progress and proximity measure different quantities. Progress = payload-to-goal distance. Proximity = agent-to-payload distance. Same shape, different measurement.
Alignment is the cosine between an agent's direction to the payload and the payload's direction to the goal: +1 is directly behind the payload, where a push helps, and −1 is in its path. Signed rather than clamped at zero, so blocking costs what pushing pays. It is a direction, not a position, so it says nothing about closing in — that is proximity's job, and the two anneal independently.
All three dense terms are scored on the change since the previous step, not on the standing value, so each is a per-unit-of-improvement coefficient. That makes their per-step magnitudes far smaller than an absolute-distance term at the same coefficient, and the coefficients need reading in those units.
Symmetric +100/−100 is a default, not a law. If agents are too reckless in rendered rollouts, widening capture to −150 against +100 buys risk-aversion without touching anything else.
Survival budget must outlast a traversal: (max_health / health_loss_per_step) × predator_cooldown_duration > 1.5 × goal_radius / push_speed / dt. If the health pool empties before the payload can arrive, success is unreachable and preserving health has no value — which is how a policy that cannot win correctly learns to die early. Also asserted by tests/test_reward_invariants.py.
Deliberately omitted terms
No global predator-distance shaping. The predator guards the payload, so a permanent distance-from-predator reward would be a permanent incentive to avoid the payload — directly undermining the objective. The original omission assumed the health penalty would supply "don't get caught." That only works if the penalty is private to the agent that was caught; a value identical across all n_agents cannot teach an individual to move. The replacement is two private signals: a radius-gated proximity penalty that is exactly zero beyond predator_danger_radius (so it is "you are about to be hit," not "stay away from the payload"), and the blame-weighted health penalty above. The gate is load-bearing — widening it toward a standing repulsion is how this becomes the term DESIGN originally rejected.

No agent-agent collision penalty. VMAS navigation penalizes this and it's tempting to copy, but cooperative pushing requires clustering near each other and the payload. A collision penalty would actively punish the behavior being encouraged. Penalize agent-obstacle and agent-wall collisions only; leave agent-agent contact free, or penalize only egregious overlap.

Reward hacking risk: payload camping
Rewarding proximity/contact risks agents learning to touch the payload and hover rather than push it.

The total-return math says camping should never win: a pushing policy collects proximity reward and progress reward, so pushing weakly dominates. But policy gradient methods don't find global optima — they follow local gradient signal. Proximity reward is dense, immediate, and stumbled into by pure random exploration (bump into a large slow object near spawn). Progress reward requires sustained, directionally correct pushing, which random exploration rarely produces before coordination exists. A policy can reliably collect proximity reward long before discovering that pushing pays more, and once a behavior is being reinforced, gradient descent concentrates around it. This is a training-dynamics problem, not a total-return problem.

Mitigations, used together:

Score proximity on distance closed per step rather than on distance standing. This is the structural fix, and it does not depend on tuning: a difference telescopes, so the proximity collected over an episode is only ever (initial distance − final distance) no matter what route produced it. An agent parked beside the payload closes nothing and so earns nothing, and the same argument covers alignment, where orbiting to farm the cosine gives back on the way out what it collected on the way in. What survives is a term that still pays a random-exploring agent for stumbling toward the payload — the exploration help that motivated it — while paying nothing at all for having arrived.
Anneal proximity to zero — removes the safe option entirely rather than relying on the policy finding its own way past it.
Keep the progress coefficient large from the start, not just eventually, so pushing is attractive early too.
Watch rendered rollouts periodically during training. This is the one that matters most. A camping policy produces a perfectly healthy-looking, positive, rising reward curve from proximity reward alone — indistinguishable from real progress on the numbers. Ten seconds of GIF is the only direct check.
Known caveat: dynamic range
±100 terminal spikes next to −0.1–−0.8 per-step terms is a wide range for the critic to fit a smooth value function across. This is a known friction point in sparse-plus-dense reward designs, not specific to these numbers. Nothing to fix preemptively — but if value loss struggles to settle, or the learning curve stays noisy well past where navigation smoothed out, look here first.

12. Repo structure
swarm-transport/
├── README.md
├── DESIGN.md                  # this document
├── requirements.txt
├── .gitignore
├── env/
│   ├── world.py               # entity state tensors, integration
│   ├── physics.py             # collision detection + penalty forces
│   ├── scenario.py            # spawn, reward, observation, done
│   └── env.py                 # reset/step/spaces wrapper
├── train/
│   ├── mappo.py
│   ├── config.py
│   └── checkpoints/
├── tests/
│   ├── test_physics.py
│   └── test_env.py
├── tools/
│   ├── scripted_policy.py     # Phase 2 controller
│   ├── render.py              # matplotlib -> GIF playback
│   └── export_onnx.py
└── web/
    ├── src/{physics.ts, policy.ts, render.ts}
    └── models/actor.onnx
The split that matters: physics.py must be a pure function of tensors with no environment knowledge — no rewards, no observations, no episode logic. That's what makes it portable to TypeScript and testable in isolation. Write it knowing it gets ported: no exotic PyTorch idioms, constants named and centralized, no reliance on broadcasting tricks that are hard to reproduce in plain arrays.

.gitignore: .venv/, __pycache__/, *.pyc, node_modules/, train/checkpoints/. Keep the single deployed web/models/actor.onnx in the repo — a few hundred KB, and the web app needs it.

13. Testing
Physics tests (Phase 1, written alongside the code, not after)
Batch independence. Run environment i inside a batch of 64 with a fixed action sequence; run the identical initial state and actions with num_envs=1. Trajectories must match to floating-point tolerance. Write this one first — it catches broadcasting bugs where environments leak into each other, which are otherwise nearly invisible and completely fatal.
Determinism. Same seed, same actions, same trajectory, every time.
No tunneling. Fast body into a wall at the chosen dt doesn't pass through.
Settling. No forces, drag on — velocities decay smoothly to zero without jitter or oscillation.
Golden trajectory test (Phase 6)
Record a fixed initial state plus a fixed action sequence in Python, save positions to JSON, replay the same in TypeScript, assert agreement within tolerance.

The TS physics will diverge from the Python physics in some subtle way — different operation order, float32 vs float64, a sign error in one branch. Without this test that surfaces as "the policy behaves weirdly in the browser" with no diagnostic path.

14. Visualization
matplotlib circle/rect patches → collect frames as arrays → imageio.mimsave("rollout.gif", frames, fps=20).

Deliberately dumb: works headless (no display dependency), drops straight into the README, ~50 lines maximum. Save real rendering effort for pixi.js.

Not optional. Ten seconds of watching catches things unit tests never will: agents vibrating against each other, payload drifting when it should be at rest, predator orbiting instead of intercepting — and most importantly, the payload-camping failure mode that metrics cannot distinguish from success.

15. MAPPO reference configuration
Validated on VMAS navigation (3 agents, 64 parallel envs): episode return −0.19 → ~3.0 plateau over 250 iterations, with action std annealing smoothly from 1.0 to ~0.55–0.62 and no collapse. This is the working baseline to adapt.

Component	Configuration
Actor	Local observation only. Linear(obs) → Tanh → Linear(n_actions), 128 hidden. Shared across homogeneous agents.
Action distribution	Gaussian, state-independent learned log_std (shared across agents)
Critic	Centralized: input = joint observations + one-hot agent id. Same architecture, single scalar output. Training-only.
gamma / lambda	0.99 / 0.95
Clip epsilon	0.2
Entropy coefficient	0.001
Rollout steps	128 per environment
Update epochs	10
Minibatch size	4096 (deliberately large — MAPPO degrades with heavy minibatch splitting)
Learning rate	3e-4, linearly annealed
Grad clipping	max_norm=0.5, applied between backward() and step()
Notes carried forward from the MAPPO paper: avoid heavy minibatch splitting; value normalization "never hurts and often helps" — skip it initially here (returns are small), but revisit for this project, since the ±100 terminal spikes put it squarely in the regime value normalization exists for.

When the predator becomes learned (Phase 5b): it needs its own actor and its own separate centralized critic. A single global value function is meaningless across an adversarial team boundary.

16. AI tooling policy
RL code fails silently. A subtly wrong GAE, a broadcasting bug that averages across environments, a reward sign error — none of these crash. They produce a flat learning curve, leaving you debugging code you didn't write and don't fully understand.

Per-file heuristic: "if this were subtly wrong, would I find out within a minute?"

Yes → AI is fine. Plotting, argparse, config plumbing, the pixi.js render loop, TypeScript scaffolding, GIF export, README formatting. Wrong output is immediately visible.
No → write it yourself. physics.py, the reward function, the observation function, spawn logic, GAE, the update loop.
Three practices that preserve understanding without giving up leverage:

AI as reviewer, not author. "Here's my collision resolution, what's wrong with it" is far more valuable than "write me collision resolution."
Ask for tests, not implementations. A generated test is checkable against your own understanding; a generated implementation asks you to trust it.
Delete any line you can't explain — at the level of explaining what expand does to memory strides, not "roughly gets the gist."
This repo is a portfolio piece. Being able to defend every design decision in an interview is worth more than shipping two weeks sooner — which is the reason this document records reasoning rather than just conclusions.

17. Open questions
Deferred deliberately, with the phase where each gets resolved:

Question	Resolve in
Arena dimensions, goal-ring radius	Phase 2 (calibrate against scripted controller)
Masses, radii, drag coefficients, force scale	Phase 2
D-dependent progress coefficient	Phase 2
Predator speed multiplier, capture radius, cooldown duration, damage rate	Phase 5 (tune from rendered rollouts)
Guard weight w in predator targeting	Phase 5
Whether box payload is too forgiving of sloppy pushing	Phase 4 (revert to circle if so)
Obstacle count and gap-width distribution	Phase 5
Whether cornering is unfair	Phase 5 (watch rollouts; tune speed or add turn-rate cap)
Value normalization	Phase 4 (add if value loss struggles)
Restricted observation radius	Phase 7 ablation
Learned vs scripted predator	Phase 5b
