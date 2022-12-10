import numpy as np

import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from fairmotion.data import bvh
from fairmotion.ops import motion as motion_class
TASK_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../"
DANCE_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../DanceData"
AIST_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../DanceData/AIST++_bvh/"


# * get motion from bvh file
def get_motion(bvh_file, load_vel, load_acc):
    bvh_path = AIST_PATH + bvh_file
    if(load_acc):
        load_vel = True # since acceleration needs velocity

    motion = bvh.load(bvh_path, load_motion=True, load_velocity= load_vel, load_acceleration=load_acc)
    return motion

#! this is example
if __name__ == "__main__":

    bvh_f = "gBR_sFM_cAll_d04_mBR0_ch01.bvh"
    motion = get_motion(bvh_f, True, True)
    print("Done")
    print(len(motion.accelerations))
    