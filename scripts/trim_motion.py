import numpy as np
import sys
import os

# include PATH
# realpath: /home/vml/music_ws/fairmotion/scripts
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

from fairmotion.data import bvh
from fairmotion.ops import motion as motion_class

import json 

TASK_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../"
DANCE_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../DanceData"
AIST_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../DanceData/AIST++_bvh/"

## * This function checks 
## * 1. whether the motion file in $path exceed corresponding $sec
## * 2. trim motion it into that value
## * 3. save each bvh file for that motion
## * 4. and save json file that consists of list of motion file names that has more than corresponding sec 
# @params: 
#   path : where to search bvh file
#   _save_new_bvh: whether to save new bvh file with frame motion_trimmed
#   _save_json: whether to save json 
#   _sec : to check how many motions exceed this value
##
def check_sec_trim_save_bvh_files(path, _save_new_bvh = False, _save_json=False, _sec=30):
    file_list = os.listdir(path)
    bvh_list = [file for file in file_list if file.endswith('.bvh')] ## 파일명 끝이 .bvh
    print("%d bvh files to check"%len(bvh_list))

    motion_dicts = {}

    for idx in range(len(bvh_list)):
        bvh_file = bvh_list[idx]
        motion = bvh.load(path + bvh_file, load_motion=True)
        # make directory if there is none
        NEW_PATH = path +"/motion_trimmed/" 

        if(_save_new_bvh):
            if not os.path.exists(NEW_PATH):
                os.makedirs(NEW_PATH)

        # trim motion into sec
        if (motion.num_frames() >= int(_sec * motion.fps)):
            motion_dicts[bvh_file] = motion.num_frames()
            cut_motion = motion_class.cut(motion, 0, _sec * motion.fps)

            NEW_BVH_FILE = NEW_PATH + bvh_file
            bvh.save(cut_motion, NEW_BVH_FILE)

        # check progress    
        if(idx % 200) == 0:
            print("%dth file done"%idx)

    # save json file
    if(_save_json):
        with open(NEW_PATH + "motion_trimmed_list.json", "w") as json_file:
            json.dump(motion_dicts, json_file)
        print("list of motions over %d frames saved"%len(motion_dicts.keys()))

    print("\n%d bvh_files exceed %d frames"%(len(motion_dicts.keys()), _sec * motion.fps))       
    print("finished")

    return


if __name__ == "__main__":
    check_sec_trim_save_bvh_files(AIST_PATH, _save_new_bvh = False, _save_json=True, _sec=30)
