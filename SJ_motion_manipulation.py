import numpy as np
from fairmotion.data import bvh
from fairmotion.ops import motion as motion_ops 
from fairmotion.ops import conversions, math, quaternion

def main():
    # load data 
    BVH_FILE = "/home/soojin/Desktop/gct634-2022-main/TeamProject/dance_bvh_example_data.bvh"
    motion = bvh.load(BVH_FILE)
    
    # get root transform
    T_root = motion.poses[0].get_root_transform()
    R1, p1 = conversions.T2Rp(T_root)
    print("Root position: ", p1)
    
    # translate motion
    desired_coord = np.array([+1.479270, -16.935950, -0.017920 ])
    translated_motion = motion_ops.translate(motion, desired_coord, pivot=0, local=False)
    
    # get translated root transform
    new_T_root = translated_motion.poses[0].get_root_transform()
    new_R1, new_p1 = conversions.T2Rp(new_T_root)
    print("New root : ", new_p1)

    # save data 
    if(False):
        NEW_BVH_FILE = "/home/soojin/Desktop/gct634-2022-main/TeamProject/dance_bvh_example_data_translated.bvh"
        bvh.save(translated_motion, NEW_BVH_FILE)
     
if __name__ == "__main__":
    main()