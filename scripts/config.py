import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

CURR_PATH = os.path.dirname(os.path.realpath(__file__))
TASK_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../"
DANCE_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../DanceData"
AIST_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../DanceData/AIST++_bvh/"

UPPER_BODY = {"LeftShoulder": 17, "LeftArm": 18, "LeftForeArm": 19, "LeftHand": 20, "LeftHandIndex1": 21, 
            "RightShoulder": 22, "RightArm": 23, "RightForeArm": 24, "RightHand": 25, "RightHandIndex1": 26}

LOWER_BODY = {"Hips": 0, 
            "LHipJoint": 1, "LeftUpLeg": 2, "LeftLeg": 3, "LeftFoot": 4, "LeftToeBase": 5, 
            "RHipJoint": 6, "RightUpLeg": 7, "RightLeg": 8, "RightFoot": 9, "RightToeBase": 10} 

MOTION_FILE = "gJB_sFM_cAll_d09_mJB0_ch15.bvh"