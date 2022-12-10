import numpy as np
import os, sys
import config
from fairmotion.data import bvh
from fairmotion.ops import motion as motion_class
from utils import get_motion, _get_magnitude

# * dot product of arr1 and arr2 in last axis
def _get_dot_of_3D_array_in_laxis(arr1, arr2):    
    assert(arr1.shape == arr2.shape)
    nframe, _, _ = arr1.shape

    dot_result = []

    # arr1[idx][:, :].shape : (27, 3)
    for idx in range(nframe):
        curr_m = arr1[idx][:, :]    # (jnt_no, pos_idx) = (27, 3)
        next_m = arr2[idx][:, :]    # (jnt_no, pos_idx) = (27, 3)
        
        next_m_T = np.transpose(next_m, axes=(1,0))

        dot_per_frame = np.dot(curr_m, next_m_T).diagonal()
        dot_result.append(dot_per_frame)

        if (idx % 1000) == 0:
            print("%d frames done"%idx)

    dot_array = np.array(dot_result)

    return dot_array

# * extract angle difference for each jnt for each frame
def extract_dtheta(motion, start_f=0, nof=1800):
    # save each joint directional change
    # positions: (seq_len(frame numbers), num_joints, 3)
    
    curr_m_trimmed = motion_class.cut(motion, start_f, start_f + nof)               # 0 ~ 1799
    next_m_trimmed = motion_class.cut(motion, start_f + 1, start_f + nof + 1)       # 1 ~ 1800

    numerator = _get_dot_of_3D_array_in_laxis(curr_m_trimmed.positions(local=False), next_m_trimmed.positions(local=False)) # (1800, 27)
    # get magnitude of each vector (okay)
    curr_mag = _get_magnitude(curr_m_trimmed.positions(local=False), 2) # (1800, 27)
    next_mag = _get_magnitude(next_m_trimmed.positions(local=False), 2) # (1800, 27)

    denominator = np.multiply(curr_mag, next_mag)   # element-wise multiply

    cangle = np.divide(numerator, denominator)         # element-wise division (1800, 27) : cos theta
    _angles = np.rad2deg(np.arccos(cangle))            # theta in rad

    return _angles                                      # (nframe, n_jnt)


# * extract directional change for one motion file
    #! we can do many jobs from HERE using extract_angle 
    #! since we got angle difference for each jnt for each frame!
def _extract_ddir(motion, start_f=0, nof=1800):
    # save each joint directional change
    # positions: (seq_len(frame numbers), num_joints, 3)
    
    _angles = extract_dtheta(motion, start_f=0, nof=1800)

    #! it's an example
    angle_per_jnt = np.sum(_angles, axis = 0)      # (27, ) :sum of angles for all frames in each jnt
    angle_per_frame = np.sum(_angles, axis = 1)        # (1800, ) : sum of angles for all jnts in each frame

    print("just an example of {angle_per_jnt}: \n", angle_per_jnt)
    return 

if __name__ == "__main__":

    bvh_f = "gBR_sFM_cAll_d04_mBR0_ch01.bvh"
    motion = get_motion(bvh_f)
    _extract_ddir(motion)

