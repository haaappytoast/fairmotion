import numpy as np 

from fairmotion.ops import conversions
from fairmotion.core.motion import Pose, Motion

# get all joint positions of all frames
def get_all_joint_positions(motion, frame_start, frame_end, joint_idx):
    joint_ps = []
    
    for frame in range(frame_start, frame_end):
        pose = Motion.get_pose_by_frame(motion, frame)
        global_p = get_joint_position(pose, joint_idx)
        joint_ps.append(np.array(global_p))
    
    return np.array(joint_ps)                               # [nof, 3]

# get global joint positions of a single joint 
def get_joint_position(pose, joint_idx):
    global_T = pose.get_transform(joint_idx, local=False)
    global_p = conversions.T2p(global_T)
    
    return global_p

# get c_i
def get_joint_centroid(joint_ps):
    joint_ps_sum = np.sum(joint_ps, axis=0)                 
    centroid = joint_ps_sum / joint_ps.shape[0]             # averaged by num of frames (output: [3,])

    return centroid

# compute span for a single joint (output: [1, ])
def extract_joint_span(motion, frame_start, frame_end, joint_idx):
    joint_span = 0    
    joint_ps = get_all_joint_positions(motion, frame_start, frame_end, joint_idx)
    centroid = get_joint_centroid(joint_ps)
    
    for frame in range(frame_start, frame_end):
        pose = Motion.get_pose_by_frame(motion, frame)
        joint_p = get_joint_position(pose, joint_idx)
        magnitude = np.linalg.norm(joint_p - centroid)     # distance between joint position & centroid 
        joint_span += magnitude
    
    return joint_span

# compute span for all joints (output: [noj, ])
def extract_motion_span(motion, frame_start, frame_end):
    motion_span = []
    skel = motion.poses[0].skel
    include_root = True
    
    for joint in skel.joints:
        
        joint_idx = skel.get_index_joint(joint)
        
        if include_root is False:
            if joint_idx == 0: 
                continue
        
        joint_span = extract_joint_span(motion, frame_start, frame_end, joint_idx)
        
        motion_span.append(joint_span)
    
    print("motion_span shape : ", np.array(motion_span).shape)
    
    return np.array(motion_span)

    
    