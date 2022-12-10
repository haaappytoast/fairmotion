import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
import matplotlib.pyplot as plt

from fairmotion.data import bvh
from fairmotion.ops import motion as motion_class

import config
from utils import get_motion, _get_magnitude
from extract_dtheta import extract_dtheta

'''
extract angular velocity from motion with jnt_index
'''
def extract_ang_vel(motion, start_f=0, nof=1800, keys = None, local=False):
    new_motion = motion_class.cut(motion, start_f, start_f + nof)
    
    # if none, get all jnts info
    if not keys: 
        keys = new_motion.skel.joints
    
    ang_vels = []

    for acc in new_motion.vels:
        for key in keys:
            ang_vels.append(acc.get_angular(key, local))
        
    np_ang_vels = np.array(ang_vels)
    np_ang_vels = np_ang_vels.reshape((len(keys), -1, 3))
    np_ang_vels = np_ang_vels.swapaxes(0, 1)

    return np_ang_vels

def extract_lin_vel(motion, start_f=0, nof=1800, keys = None, local=False):
    new_motion = motion_class.cut(motion, start_f, start_f + nof)
    
    # if none, get all jnts info
    if not keys: 
        keys = new_motion.skel.joints
    
    lin_vels = []

    for acc in new_motion.vels:
        for key in keys:
            lin_vels.append(acc.get_linear(key, local))
        
    np_lin_vels = np.array(lin_vels)
    np_lin_vels = np_lin_vels.reshape((len(keys), -1, 3))
    np_lin_vels = np_lin_vels.swapaxes(0, 1)
    
    return np_lin_vels

def extract_ang_acc(motion, start_f=0, nof=1800, keys = None, local=False):
    new_motion = motion_class.cut(motion, start_f, start_f + nof)
    
    # if none, get all jnts info
    if not keys: 
        keys = new_motion.skel.joints
    
    ang_acc = []

    for acc in new_motion.accelerations:
        for key in keys:
            ang_acc.append(acc.get_angular(key, local))
        
    np_ang_acc = np.array(ang_acc)
    np_ang_acc = np_ang_acc.reshape((len(keys), -1, 3))
    np_ang_acc = np_ang_acc.swapaxes(0, 1)
    
    return np_ang_acc

def extract_lin_acc(motion, start_f=0, nof=1800, keys = None, local=False):
    new_motion = motion_class.cut(motion, start_f, start_f + nof)
    
    # if none, get all jnts info
    if not keys: 
        keys = new_motion.skel.joints
    
    lin_acc = []

    for acc in new_motion.accelerations:
        for key in keys:
            lin_acc.append(acc.get_linear(key, local))
        
    np_lin_acc = np.array(lin_acc)
    np_lin_acc = np_lin_acc.reshape((len(keys), -1, 3))
    np_lin_acc = np_lin_acc.swapaxes(0, 1)
    
    return np_lin_acc


if __name__ == "__main__":

    bvh_f = config.MOTION_FILE
    motion = get_motion(bvh_f, True, True)

    # example
    lin_acc = extract_lin_acc(motion, start_f=0, nof=1800, keys=config.LOWER_BODY, local=False)
    print(lin_acc.shape)
    lin_acc_mag = _get_magnitude(lin_acc, 2) # (1800, 27)

