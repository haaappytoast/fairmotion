import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
import matplotlib.pyplot as plt

from fairmotion.data import bvh
from fairmotion.ops import motion as motion_class

import config
from utils import get_motion, _get_magnitude, _get_sum, load_npy_file, cal_msbeat_per_frame
from extract_dtheta import extract_dtheta
import extract_vel as v

'''
to get sum_of_values from jnt_idx
@ param: jnt_dict = {"jnt_name": jnt_idx}
@ param: data = value which has size of {nframe, jnt_indices}
@ return: summed_data (nframe, )
'''
def _get_sum_of_values(data, jnt_dict, axs = 0):
    assert isinstance(data, np.ndarray)
    summed_data = np.zeros(np.size(data, 0))
    
    data = data.transpose()
    for (_, jnt_idx) in jnt_dict.items():
        summed_data += np.array(data[jnt_idx][:])
    return summed_data

    

if __name__ == "__main__":

    bvh_f = config.MOTION_FILE
    # bvh_f = "gBR_sFM_cAll_d04_mBR0_ch01.bvh"
    motion = get_motion(bvh_f, True, True)
    nof = 1800

    # LOWER_BODY example
    # 1. dacc for lower body
    key = config.LOWER_BODY
    low_lin_dacc = -v.extract_lin_acc(motion, start_f=0, nof = nof, keys=key, local=False)
    
    low_lin_dacc_mag = _get_magnitude(low_lin_dacc, 2)        # (nframes, njnt)
    low_avg_lin_dacc = _get_sum(low_lin_dacc_mag, axs=1) / float(nof)     # (nframes, )


    # 2. directional change
    low_dtheta = extract_dtheta(motion) # (nframes, njnts)
    low_avg_dtheta = _get_sum_of_values(low_dtheta, key, axs=0) / float(len(key))

    print("lower body avg_dtheta: ", np.max(low_avg_dtheta), np.min(low_avg_dtheta))
    print("lower body avg_lin_dacc: ", np.max(low_avg_lin_dacc), np.min(low_avg_lin_dacc))

    # UPPER_BODY example
    # 1. dacc for lower body
    key = config.UPPER_BODY
    up_lin_dacc = -v.extract_lin_acc(motion, start_f=0, nof = nof, keys=key, local=False)
    
    up_lin_dacc_mag = _get_magnitude(up_lin_dacc, 2)        # (nframes, njnt)
    up_avg_lin_dacc = _get_sum(up_lin_dacc_mag, axs=1) / float(nof)     # (nframes, )

    # 2. directional change
    up_dtheta = extract_dtheta(motion) # (nframes, njnts)
    up_avg_dtheta = _get_sum_of_values(up_dtheta, key, axs=0) / float(len(key))

    print("upper body avg_dtheta: ",np.max(up_avg_dtheta), np.min(up_avg_dtheta))
    print("upper body up_avg_lin_dacc: ",np.max(up_avg_lin_dacc), np.min(up_avg_lin_dacc))

    # music beat
    beat = load_npy_file(bvh_f, path=config.CURR_PATH + "/beat_feat/")
    beats = cal_msbeat_per_frame(beat, nof, motion.fps)


    # 3. plot
    x = np.arange(0, nof, 1)
    fig, (ax0, ax1) = plt.subplots(2, 1)

    ax0.plot(x, low_avg_lin_dacc, label='low_avg_lin_dacc'); ax0.plot(x, low_avg_dtheta, label='low_avg_dtheta'); ax0.plot(x, beats, label='music beats')
    ax1.plot(x, up_avg_lin_dacc, label='up_avg_lin_dacc'); ax1.plot(x, up_avg_dtheta, label='up_avg_dtheta'); ax1.plot(x, beats, label='music beats')
    ax0.legend(prop={'size': 20}); ax1.legend(prop={'size': 20})
    # ax1.plot(x, up_avg_lin_dacc, x, up_avg_dtheta,  x, beats)

    ax0.axis([0, 500, 0, 3.5])
    ax1.axis([0, 500, 0, 3.5])
    plt.show()
