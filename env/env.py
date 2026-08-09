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
6. scenario.observe()         -> observation tensor
7. dataclasses.replace on scenario_state to advance step_count and prev_payload_dist
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
        predator_action, self.scenario_state = scenario.predator_policy(self.world_state, self.scenario_state, self.config)
        
        self.world_state = physics.step(self.world_state, agent_action, predator_action, self.config.dt, self.config.agent_max_thrust, self.config.predator_max_thrust, self.config.agent_drag_coef, self.config.predator_drag_coef, self.config.payload_drag_coef, self.config.body_stiffness, self.config.wall_stiffness, self.config.obstacle_stiffness, self.config.payload_stiffness)
        self.scenario_state = scenario.update_health(self.world_state, self.scenario_state, self.config)
        
        reward = scenario.compute_reward(self.world_state, self.scenario_state, training_progress, self.config)
        terminated, truncated = scenario.compute_done(self.world_state, self.scenario_state, self.config)
        needs_reset = terminated | truncated
        observation = scenario.observe(self.world_state, self.scenario_state, self.config)
        
        payload_dist = torch.norm(self.world_state.payload_pos - self.scenario_state.goal_pos, dim=-1)
        info = {
            "payload_dist": payload_dist,
            "success": payload_dist < self.config.success_threshold,
            "captured": self.scenario_state.health <= 0.0,
            "world_state": self.world_state,
            "scenario_state": self.scenario_state,
        }
        
        
        self.world_state, self.scenario_state = scenario.reset_at(self.world_state, self.scenario_state, needs_reset, self.config, self.generator)
        
        current_payload_dist = torch.norm(self.world_state.payload_pos - self.scenario_state.goal_pos , dim=-1)
        self.scenario_state = dataclasses.replace(self.scenario_state, step_count=self.scenario_state.step_count + 1, prev_payload_dist=current_payload_dist)
        
        return observation, reward, terminated, truncated, info
    
    def reset(self):
        self.world_state, self.scenario_state = scenario.reset(self.config.num_envs, self.config, self.generator)
        return scenario.observe(self.world_state, self.scenario_state, self.config)
    
    
    