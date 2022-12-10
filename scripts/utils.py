import numpy as np

import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from fairmotion.data import bvh
from fairmotion.ops import motion as motion_class
import config
from fairmotion.core import acceleration

# * get magnitude for each vector
# (x, y, z) -> srt(x^2 + y^2 + z^2)
def _get_magnitude(matrix, axs):
    return np.linalg.norm(matrix, axis = axs)


def _get_sum(matrix, axs):
    return np.sum(matrix, axis = axs)

def print_joint_names(motion):
    for jnt in motion.skel.joints :
        print(jnt.name, motion.skel.get_index_joint(jnt.name))

# * get motion from bvh file
def get_motion(bvh_file, load_vel=True, load_acc=False):
    bvh_path = config.AIST_PATH + bvh_file

        

    if(load_acc):
        load_vel = True # since acceleration needs velocity

    motion = bvh.load(bvh_path, load_motion=True, load_velocity= load_vel, load_acceleration=load_acc)

    # motion.accelerations
    return motion

# * get music id from bvh file name
def _get_music_id(name):
    mid = name.split('_')[-2]
    return mid

def load_npy_file(name, path):
    mid = _get_music_id(name)
    mname = path + mid + '.npy'

    m_feature = np.load(mname)

    return m_feature

def _get_frame_idx_of_mfeature(mfeature, fps, axs=0):

    dt = 1/fps
    frame_idx = np.round(mfeature[axs] / dt).astype(int)
    
    return np.vstack((frame_idx, mfeature[1])).astype(int)

def cal_msbeat_per_frame(beat, len, fps):

    filtered_beat = _get_frame_idx_of_mfeature(beat, fps)

    filled_np = np.zeros(len)

    size = np.size(filtered_beat[0], axis=0)
    for idx in range(size):
        fidx = filtered_beat[0][idx]
        filled_np[fidx] = filtered_beat[1][idx]

    return filled_np

#! this is example
if __name__ == "__main__":

    bvh_f = config.MOTION_FILE
    beat = load_npy_file(bvh_f, path=config.CURR_PATH + "/beat_feat/")
    beats = cal_msbeat_per_frame(beat, motion.num_frames())
    
    motion = get_motion(bvh_f, False, False)

    print(beat.shape)


    