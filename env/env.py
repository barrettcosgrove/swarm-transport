from . import world
from . import scenario
from . import physics
from train.config import Config
from .world import WorldState
from .scenario import ScenarioState
import torch
import dataclasses

"""
1. predator_policy()          -> predator action + updated scenario state (noise saved)
2. physics.step()             -> new world state
3. scenario.update_health()   -> updated health + cooldown + prev_health saved
4. scenario.compute_reward()  -> reward tensor
5. scenario.compute_done()    -> terminated, truncated
6. scenario.observe()         -> the pre-reset observation, kept in info
7. build the info dict from the pre-reset state
8. scenario.reset_at()        -> respawn the environments that finished
9. dataclasses.replace on scenario_state to advance step_count and re-baseline
   the shaping terms: prev_payload_dist, prev_agent_payload_dist, prev_alignment
10. scenario.observe()        -> the observation actually returned, post-reset
"""

class Env:
    def __init__(self, config):
        self.config = config
        self.generator = torch.Generator(device=config.device)
        self.generator.manual_seed(config.seed)
        self.world_state, self.scenario_state = scenario.reset(config.num_envs, config, self.generator)
        self.total_steps = 0
    
    def step(self, agent_action, training_progress=None):
        self.total_steps += self.config.num_envs
        if training_progress is None:
            training_progress = min(self.total_steps / (self.config.num_iterations * self.config.rollout_steps * self.config.num_envs), 1.0)
        predator_action, self.scenario_state = scenario.predator_policy(self.world_state, self.scenario_state, self.config, self.generator)
        
        predator_max_speed = scenario.effective_predator_max_speed(self.scenario_state, self.config)
        self.world_state = physics.step(self.world_state, agent_action, predator_action, self.config.dt, self.config.agent_max_thrust, self.config.predator_max_thrust, self.config.agent_drag_coef, self.config.predator_drag_coef, self.config.payload_drag_coef, self.config.body_stiffness, self.config.wall_stiffness, self.config.obstacle_stiffness, self.config.payload_stiffness, predator_max_speed, agent_max_speed=self.config.agent_max_speed)
        self.scenario_state = scenario.update_health(self.world_state, self.scenario_state, self.config)
        
        terms = scenario.reward_terms(self.world_state, self.scenario_state, training_progress, self.config)
        reward = torch.stack(tuple(terms.values())).sum(0)
        terminated, truncated = scenario.compute_done(self.world_state, self.scenario_state, self.config)
        needs_reset = terminated | truncated
        final_observation = scenario.observe(self.world_state, self.scenario_state, self.config)
        
        payload_dist = torch.norm(self.world_state.payload_pos - self.scenario_state.goal_pos, dim=-1)
        info = {
            "payload_dist": payload_dist,
            "success": payload_dist < self.config.success_threshold,
            "captured": self.scenario_state.health <= 0.0,
            "world_state": self.world_state,
            "scenario_state": self.scenario_state,
            "reward_terms": terms,
            # the last observation of the episode that just ended. GAE has to
            # bootstrap a truncated episode from the value of where it actually
            # stopped, which the returned observation no longer describes.
            "final_observation": final_observation,
        }
        
        
        self.world_state, self.scenario_state = scenario.reset_at(self.world_state, self.scenario_state, needs_reset, self.config, self.generator)
        
        # step_count advances after reset_at, so a freshly respawned environment
        # reports step_count 1 on its first observation rather than 0 -- a
        # one-step offset in time_remaining. Left alone deliberately:
        # compute_done and the renderer both read step_count under this
        # convention, and moving the increment would shift truncation by a step.
        current_payload_dist = torch.norm(self.world_state.payload_pos - self.scenario_state.goal_pos , dim=-1)
        current_agent_payload_dist, current_alignment = scenario.agent_payload_geometry(
            self.world_state.agent_pos, self.world_state.payload_pos, self.scenario_state.goal_pos)
        self.scenario_state = dataclasses.replace(self.scenario_state, step_count=self.scenario_state.step_count + 1, prev_payload_dist=current_payload_dist, prev_agent_payload_dist=current_agent_payload_dist, prev_alignment=current_alignment)
        
        # observed AFTER reset_at, so what a rollout loop carries into the next
        # step matches the state the simulation is actually in. Returning the
        # pre-reset observation would feed the policy a stale board on every
        # terminal step. Costs one extra observe(), which is subtractions and a
        # gather -- negligible against physics.step.
        observation = scenario.observe(self.world_state, self.scenario_state, self.config)
        
        return observation, reward, terminated, truncated, info
    
    def reset(self):
        self.world_state, self.scenario_state = scenario.reset(self.config.num_envs, self.config, self.generator)
        return scenario.observe(self.world_state, self.scenario_state, self.config)
    
    
    