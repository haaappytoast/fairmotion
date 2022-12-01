import numpy as np
import sys
import os

# include PATH
# realpath: /home/vml/music_ws/fairmotion/scripts
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

from fairmotion.data import bvh
from fairmotion.ops import conversions

TASK_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../"
DANCE_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../DanceData"
AIST_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../DanceData/AIST++_bvh/"

# * CHANGE ROOT position to (0, 0, 0) and save new bvh file
def change_save_bvh_files(path):
    file_list = os.listdir(path)
    bvh_list = [file for file in file_list if file.endswith('.bvh')] ## 파일명 끝이 .bvh
    print("# %d bvh files to change offset"%len(bvh_list))

    for bvh_file in bvh_list:
        motion = bvh.load(path + bvh_file, load_motion=True)

        # Change OFFSET of root joint
        motion.skel.root_joint.xform_from_parent_joint = get_new_root_offset_to(motion, np.array(0, 0, 0))
        NEW_PATH = path +"CHANGED/" 
        NEW_BVH_FILE = NEW_PATH + bvh_file

        # make directory if no
        if not os.path.exists(NEW_PATH):
            os.makedirs(NEW_PATH)
        bvh.save(motion, NEW_BVH_FILE)
    print("finished")
    return

# * change ROOT position offset to new offset
def get_new_root_offset_to(motion, offset = [0.0, 0.0, 0.0]):

    offset = np.array(offset, float)
    prev_rot, prev_pos = conversions.T2Rp(motion.skel.root_joint.xform_from_parent_joint)
    
    new_trf = conversions.Rp2T(prev_rot, offset)
    new_motion = motion
    new_motion.skel.root_joint.xform_from_parent_joint = new_trf
    return new_motion

if __name__ == "__main__":
    change_save_bvh_files(AIST_PATH)

    #! one example test 
    if False:
        BVH_FILE = AIST_PATH + "gBR_sBM_cAll_d04_mBR0_ch01.bvh"

        ## GET OFFSET from root joint
        motion = bvh.load(BVH_FILE, load_motion=True)
        
        new_motion = get_new_root_offset_to(motion, [0, 0, 0])