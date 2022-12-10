'''
It's from Soojin's code
'''

import numpy as np 

from fairmotion.utils import constants
from fairmotion.ops import math
from fairmotion.core.motion import Motion
from fairmotion.core.velocity import Velocity

class Acceleration(object):
    def __init__(self, v_init=None, v_final=None, dt=None):
        self.data_local = None
        self.data_global = None

        if v_init:
            assert v_final and dt
            assert isinstance(v_init, Velocity) and isinstance(v_final, Velocity)
            self.skel = v_init.skel
            self.data_local, self.data_global = Acceleration.compute(v_init, v_final, dt)
    
    def set_skel(self, skel):
        self.skel = skel
    
    def set_data_local(self, data):
        self.data_local = data

    def set_data_global(self, data):
        self.data_global = data
    
    @classmethod
    def compute(cls, v_init, v_final, dt):
        assert v_init.skel == v_final.skel

        data_local = []
        data_global = []
        
        assert dt > constants.EPSILON
        
        data_local = v_final.data_local / dt - v_init.data_local / dt
        data_global = v_final.data_global / dt - v_init.data_global / dt

        return np.array(data_local), np.array(data_global)
    
    def get_all(self, key, local, R_ref=None):
        """Returns both linear and angular velocity stacked together"""
        return np.hstack(
            [
                self.get_angular(key, local, R_ref),
                self.get_linear(key, local, R_ref),
            ]
        )

    def get_angular(self, key, local, R_ref=None):
        data = self.data_local if local else self.data_global
        w = data[self.skel.get_index_joint(key), 0:3]
        if R_ref is not None:
            w = np.dot(R_ref, w)
        return w

    def get_linear(self, key, local, R_ref=None):
        data = self.data_local if local else self.data_global
        v = data[self.skel.get_index_joint(key), 3:6]
        if R_ref is not None:
            v = np.dot(R_ref, v)
        return v 
            
    @classmethod
    def interpolate(cls, a1, a2, alpha):
        data_local = math.lerp(a1.data_local, a2.data_local, alpha)
        data_global = math.lerp(a1.data_global, a2.data_global, alpha)
        a = cls()
        a.set_skel(a1.skel)
        a.set_data_local(data_local)
        a.set_data_global(data_global)
        return a
    
class MotionWithAcceleration(Motion):
    def __init__(self, name="motion", skel=None, fps=60):
        super().__init__(name, skel, fps)
        self.accelerations = []
        
    def compute_accelerations(self):
        self.accelerations = self._compute_accelerations()
        
    def _compute_accelerations(self, frame_start=None, frame_end=None):
        accelerations = []
        v_init = None
        v_final = None

        if frame_start is None:
            frame_start = 0
        if frame_end is None:
            frame_end = self.num_frames()               # num_frames() from motion.py
        
        for i in range(frame_start, frame_end):
            # interval: 2 frames 
            frame1 = max(0, (i - 1))
            frame2 = min(self.num_frames() - 1, (i + 1))

            dt = (frame2 - frame1) / float(self.fps)


            pose1 = self.get_pose_by_frame(frame1)
            pose2 = self.get_pose_by_frame(frame2)
            
            # compute velocity
            v_final = Velocity(pose1, pose2, dt)
            if v_init == None:
                v_init = v_final

            # compute accelerations
            accelerations.append(Acceleration(v_init, v_final, dt))

            # keep current velocity to use for the next frame 
            v_init = v_final
        
        return accelerations
    
    def get_acceleration_by_time(self, time):
        assert len(self.accelerations) > 0, (
            "Acceleration was not computed yet.", 
            "Please call self.compute_acceleration() first",
        )
        
        time = np.clip(time, 0, self.length())
        frame1 = self.time_to_frame(time)
        frame2 = min(frame1 + 1, self.num_frames() - 1)
        if frame1 == frame2:
            return self.accelerations[frame1]
        
        t1 = self.frame_to_time(frame1)
        t2 = self.frame_to_time(frame2)
        alpha = (time - t1) / (t2 - t1)
        alpha = np.clip((time - t1) / (t2 - t1), 0.0, 1.0)
        
        # ? Is this correct?
        a1 = self.get_acceleration_by_frame(frame1)
        a2 = self.get_acceleration_by_frame(frame2)
        return Acceleration.interpolate(a1, a2, alpha)
                
    def get_acceleration_by_frame(self, frame):
        assert len(self.accelerations) > 0, (
            "Acceleration was not computed yet.", 
            "Please call self.compute_acceleration() first",
        )
        assert frame < self.num_frames()
        return self.accelerations[frame]

    @classmethod
    def from_motion(cls, m):
        assert m.vels
        ma = cls(m.name, m.skel, m.fps)
        ma.poses = m.poses
        ma.vels = m.vels
        ma.info = m.info

        ma.compute_accelerations()
        return ma
