import torch 
import dataclasses
import torch.nn.functional as F

def integrate(pos, vel, force, mass, dt):
    acceleration = force / mass
    new_vel = vel + acceleration * dt
    new_pos = pos + new_vel * dt
    return new_pos, new_vel

def circle_circle_forces(pos_a, radius_a, pos_b, radius_b, stiffness):
    squeeze_a = pos_a.dim() == 2
    squeeze_b = pos_b.dim() == 2
    pa = pos_a.unsqueeze(1) if squeeze_a else pos_a
    pb = pos_b.unsqueeze(1) if squeeze_b else pos_b
    
    delta = pa.unsqueeze(1) - pb.unsqueeze(2)
    dist_centers = torch.norm(delta, dim=-1)
    dist_centers = torch.clamp(dist_centers, min=1e-6)
    sum_radii = radius_a + radius_b
    
    penetration = sum_radii - dist_centers
    penetration = torch.clamp(penetration, min=0.0)
    
    force_magnitude = stiffness * penetration
    direction = delta / dist_centers.unsqueeze(-1)
    force_pair = force_magnitude.unsqueeze(-1) * direction
    
    force_a = force_pair.sum(dim=1)
    force_b = -force_pair.sum(dim=2)
    
    if squeeze_a: force_a = force_a.squeeze(1)
    if squeeze_b: force_b = force_b.squeeze(1)
    
    return force_a, force_b

def _circle_box_forces_core(cp, bc, bh, circle_radius, stiffness):
    box_min = bc - bh
    box_max = bc + bh
    
    closest = torch.clamp(cp, min=box_min, max=box_max)
    delta_ext = cp - closest
    dist = torch.norm(delta_ext, dim=-1)
    inside = dist < 1e-9
    dist_safe = torch.clamp(dist, min=1e-6)
    pen_ext = torch.clamp(circle_radius - dist_safe, min=0.0)
    dir_ext = delta_ext / dist_safe.unsqueeze(-1)
    
    to_min = cp - box_min
    to_max = box_max - cp
    exit_per_axis = torch.min(to_min, to_max)
    toward_max = to_max < to_min
    exit_dist, axis = torch.min(exit_per_axis, dim=-1)
    
    axis_mask = F.one_hot(axis, num_classes=2).float()
    sign = torch.where(toward_max, torch.ones_like(to_max), -torch.ones_like(to_max))
    dir_in  = (sign * axis_mask)
    pen_in  = exit_dist + circle_radius
    
    penetration = torch.where(inside, pen_in, pen_ext)
    direction   = torch.where(inside.unsqueeze(-1), dir_in, dir_ext)
    return stiffness * penetration.unsqueeze(-1) * direction
    

def circle_box_static_forces(circle_pos, circle_radius, box_center, box_halfsize, stiffness):
    squeeze_output = circle_pos.dim() == 2
    circle_pos_ = circle_pos.unsqueeze(1) if squeeze_output else circle_pos
    
    if box_center.dim() == 3:
        bc = box_center.unsqueeze(1)
        cp = circle_pos_.unsqueeze(2)
        bh = box_halfsize if box_halfsize.dim() == 3 else box_halfsize.unsqueeze(0)
    else:
        bc = box_center
        cp = circle_pos_.unsqueeze(2)
        bh = box_halfsize
    force = _circle_box_forces_core(cp, bc, bh, circle_radius, stiffness)
    force = force.sum(dim=2)
    return force.squeeze(1) if squeeze_output else force

def circle_box_dynamic_forces(circle_pos, circle_radius, box_center, box_halfsize, stiffness):
    squeeze_output = circle_pos.dim() == 2
    cp = circle_pos.unsqueeze(1) if squeeze_output else circle_pos
    bc = box_center.unsqueeze(1) 
    bh = box_halfsize
    force_a = _circle_box_forces_core(cp, bc, bh, circle_radius, stiffness)
    force_b = -force_a.sum(dim=1)
    if squeeze_output: force_a = force_a.squeeze(1)
    return force_a, force_b

def box_box_forces(payload_center, payload_halfsize, box_center, box_halfsize, stiffness):
    payload_center = payload_center.unsqueeze(1)
    delta = payload_center - box_center
    halfsize_sum = payload_halfsize + box_halfsize
    overlap = halfsize_sum - torch.abs(delta)
    overlap = torch.clamp(overlap, min=0.0)
    
    penetration, _ = torch.min(overlap, dim=-1)
    mask = (overlap == penetration.unsqueeze(-1)).float()
    direction = torch.sign(delta) * mask
    
    force_magnitude = stiffness * penetration.unsqueeze(-1)
    force = force_magnitude * direction
    return force.sum(dim=1)

def step(world_state, agent_actions, predator_actions, dt, agent_max_thrust, predator_max_thrust, agent_drag_coef, predator_drag_coef, payload_drag_coef, body_stiffness,wall_stiffness, obstacle_stiffness, payload_stiffness):
    agent_thrust = agent_actions * agent_max_thrust
    agent_drag = -agent_drag_coef * world_state.agent_vel
    
    predator_thrust = predator_actions * predator_max_thrust
    predator_drag = -predator_drag_coef * world_state.predator_vel
    
    payload_drag = -payload_drag_coef * world_state.payload_vel
    
    #agent-agent forces
    force_agent_agent, _ = circle_circle_forces(world_state.agent_pos, world_state.agent_radius, world_state.agent_pos, world_state.agent_radius, body_stiffness)
    
    #agent-predator forces
    force_agent_from_predator, force_predator_from_agent = circle_circle_forces(world_state.agent_pos, world_state.agent_radius, world_state.predator_pos, world_state.predator_radius, body_stiffness)

    #agent-payload forces
    force_agent_from_payload, force_payload_from_agent = circle_box_dynamic_forces(world_state.agent_pos, world_state.agent_radius, world_state.payload_pos, world_state.payload_halfsize, payload_stiffness)
    
    #agent-wall forces
    force_agent_wall = circle_box_static_forces(world_state.agent_pos, world_state.agent_radius, world_state.wall_center, world_state.wall_halfsize, wall_stiffness)
    
    #agent-obstacle forces
    force_agent_obstacle = circle_box_static_forces(world_state.agent_pos, world_state.agent_radius, world_state.obstacle_center, world_state.obstacle_halfsize, obstacle_stiffness)
    
    #predator-wall forces
    force_predator_wall = circle_box_static_forces(world_state.predator_pos, world_state.predator_radius, world_state.wall_center, world_state.wall_halfsize, wall_stiffness)
    
    #predator-obstacle forces
    force_predator_obstacle = circle_box_static_forces(world_state.predator_pos, world_state.predator_radius, world_state.obstacle_center, world_state.obstacle_halfsize, obstacle_stiffness)
    
    #predator-payload forces
    force_predator_from_payload, force_payload_from_predator = circle_box_dynamic_forces(world_state.predator_pos, world_state.predator_radius, world_state.payload_pos, world_state.payload_halfsize, payload_stiffness)
    
    #payload-wall forces
    force_payload_wall = box_box_forces(world_state.payload_pos, world_state.payload_halfsize, world_state.wall_center, world_state.wall_halfsize, wall_stiffness)
    
    #payload-obstacle forces
    force_payload_obstacle = box_box_forces(world_state.payload_pos, world_state.payload_halfsize, world_state.obstacle_center, world_state.obstacle_halfsize, obstacle_stiffness)
    
    #Need to refactor this 
    agent_total_force = agent_thrust + agent_drag + force_agent_agent + force_agent_from_predator + force_agent_from_payload + force_agent_wall + force_agent_obstacle
    predator_total_force = predator_thrust + predator_drag + force_predator_from_agent + force_predator_from_payload + force_predator_wall + force_predator_obstacle
    payload_total_force = payload_drag + force_payload_from_agent + force_payload_from_predator + force_payload_wall + force_payload_obstacle
    
    new_agent_pos, new_agent_vel = integrate(world_state.agent_pos, world_state.agent_vel, agent_total_force, world_state.agent_mass, dt)
    new_predator_pos, new_predator_vel = integrate(world_state.predator_pos, world_state.predator_vel, predator_total_force, world_state.predator_mass, dt)
    new_payload_pos, new_payload_vel = integrate(world_state.payload_pos, world_state.payload_vel, payload_total_force, world_state.payload_mass, dt)
    
    new_world_state = dataclasses.replace(world_state, agent_pos=new_agent_pos, agent_vel=new_agent_vel, predator_pos=new_predator_pos, predator_vel=new_predator_vel, payload_pos=new_payload_pos, payload_vel=new_payload_vel)
    return new_world_state