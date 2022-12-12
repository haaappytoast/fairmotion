import numpy as np 

from fairmotion.ops import conversions
from fairmotion.core.motion import Pose, Motion

# get global joint positions of a single joint 
def get_joint_position(pose, joint_idx):
    global_T = pose.get_transform(joint_idx, local=False)
    global_p = conversions.T2p(global_T)
    
    return global_p

# get all joint positions of selected frames 
def get_all_joint_positions(motion, frame_start, frame_end, joint_idx):
    joint_ps = []
    
    for frame in range(frame_start, frame_end):
        pose = Motion.get_pose_by_frame(motion, frame)
        global_p = get_joint_position(pose, joint_idx)
        joint_ps.append(global_p)
    
    return np.array(joint_ps)                               # [nof, 3]

# get c_i
def get_joint_centroid(joint_ps):
    joint_ps_sum = np.sum(joint_ps, axis=0)                 
    centroid = joint_ps_sum / joint_ps.shape[0]             # averaged by num of frames (output: [3,])

    return centroid

# get c_i for all joints 
def get_all_centroids(motion, frame_start, frame_end):
    centroids=[]
    skel = motion.poses[0].skel
    
    for joint in skel.joints:
        joint_idx = skel.get_index_joint(joint)
        joint_ps = get_all_joint_positions(motion, frame_start, frame_end, joint_idx)
        centroid = get_joint_centroid(joint_ps)
        centroids.append(centroid) 
    
    return centroids

# compute span of a single joint 
def extract_joint_span(motion, frame_start, frame_end, joint_idx, is_sum_all):
    joint_span = []    
    joint_ps = get_all_joint_positions(motion, frame_start, frame_end, joint_idx)
    centroid = get_joint_centroid(joint_ps)
    
    for frame in range(frame_start, frame_end):
        joint_p = joint_ps[frame]
        magnitude = np.linalg.norm(joint_p - centroid)     # ! distance between joint position & centroid 
        joint_span.append(magnitude)
    
    if is_sum_all is True:
        return np.sum(np.array(joint_span), axis=0)        # [1,]
    else:
        return np.array(joint_span)                        # [nof,]

# compute span of all joints 
def extract_motion_span(motion, frame_start, frame_end, is_sum_all):
    motion_span = []
    skel = motion.poses[0].skel
    include_root = True
    
    for joint in skel.joints:
        joint_idx = skel.get_index_joint(joint)

        if include_root is False:
            if joint_idx == 0: 
                continue
        
        joint_span = extract_joint_span(motion, frame_start, frame_end, joint_idx, is_sum_all=True)        
        motion_span.append(joint_span)                      # [noj,]
    
    if is_sum_all is True:
        return np.sum(np.array(motion_span), axis=0)        # [1,]
    else:
        return np.array(motion_span)                        # [noj,]

# compute span for a single pose in a single frame 
def extract_pose_span(pose, centroids):
    pose_span = 0
    skel = pose.skel
    
    for joint in skel.joints:
        joint_idx = skel.get_index_joint(joint)

        joint_p = get_joint_position(pose, joint_idx)
        centroid = centroids[joint_idx]
        
        magnitude = np.linalg.norm(joint_p - centroid)     # ! distance between joint position & centroid 
        pose_span += magnitude
    
    return np.array(pose_span)        #[1,]
    

# compute span of all joints with respect to time-axis (output: [nof,])
def extract_frame_wise_pose_span(motion, frame_start, frame_end):
    motion_span = []
    centroids = get_all_centroids(motion, frame_start, frame_end)
    
    for frame in range(frame_start, frame_end):
        pose = Motion.get_pose_by_frame(motion, frame)    
        pose_span = extract_pose_span(pose, centroids)
        motion_span.append(pose_span)
    
    return np.array(motion_span)
    
# comnpute span of selected joints 
# def extract_selected_joints_span(motion, frame_start, frame_end, joint_idxes):
#     joints_span = []
#     skel = motion.poses[0].skel
    
#     for joint_idx in joint_idxes:
#         joint_span = extract_joint_span(motion, frame_start, frame_end, joint_idx)
#         joints_span.append(joint_span)
    
#     return np.array(joints_span)                           # [n, ] (n = num of seleced joints)
