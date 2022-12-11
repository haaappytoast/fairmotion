import numpy as np 
import motionspan 

from fairmotion.utils import constants
from fairmotion.ops import conversions
from fairmotion.core.motion import Pose, Motion
from fairmotion.core.velocity import Velocity

# get linear velocity of a single joint in a single frame
def get_linear_joint_velocity(pose1, pose2, dt, joint_idx, is_local):
    local_vel, global_vel = Velocity.compute(pose1, pose2, dt)
    
    if is_local is True:
        joint_vel = local_vel[joint_idx]
    else: 
        joint_vel = global_vel[joint_idx]
    
    return np.array(joint_vel[3:6])
    
# compute density of a single joint (output: [1, ])
def extract_joint_density(motion, frame_start, frame_end, joint_idx, is_local):
    joint_span = motionspan.extract_joint_span(motion, frame_start, frame_end, joint_idx)
    joint_vels = 0
    
    for i in range(frame_start, frame_end):
        
        nof = abs(frame_end - frame_start)
        frame1 = max(0, i-1)
        frame2 = min(nof-1, i+1)
        
        pose1 = Motion.get_pose_by_frame(motion, frame1)
        pose2 = Motion.get_pose_by_frame(motion, frame2)
        dt = (frame2 - frame1) / float(motion.fps)
        
        vel = get_linear_joint_velocity(pose1, pose2, dt, joint_idx, is_local)
        joint_vels += vel
    
    joint_density = np.linalg.norm(joint_vels) / joint_span     # ! Check if the equation is correct 
    
    return np.array(joint_density) 

# compute density of selected joints (output: [n, ], n  = num of seleced joints)
def extract_selected_joints_density(motion, frame_start, frame_end, joint_idxes, is_local):
    joints_density = []
    skel = motion.poses[0].skel
    
    for joint_idx in joint_idxes:
        joint_density = extract_joint_density(motion, frame_start, frame_end, joint_idx, is_local)
        joints_density.append(joint_density)
    
    return np.array(joints_density)

# compute density of all joints (output: [noj, ])
def extract_motion_density(motion, frame_start, frame_end, is_local):
    motion_density = []
    skel = motion.poses[0].skel
    include_root = True
            
    for joint in skel.joints:
        
        joint_idx = skel.get_index_joint(joint)
        
        if include_root is False:
            if joint_idx == 0:
                continue
        
        joint_density = extract_joint_density(motion, frame_start, frame_end, joint_idx, is_local)
        motion_density.append(joint_density)
        
    return np.array(motion_density)
        
    
    


 

    

    
