from PID import PID
import numpy as np


class DummyExpert:


    heightPID = PID(0.08, 0, 0.3, 0.1, 1)
    pitchPID = PID(10, 0, 1000, -10, 10)
    rollPID = PID(10, 0, 1000, -10, 10)
    XPID = PID(0.2, 0, 0.8, -10, 10)
    YPID = PID(0.2, 0, 0.8, -10, 10)

    def teach(self,obs):

        throttle = self.heightPID.control((obs[2]+1.0)/2.0*1000.0, obs[5]*100.0, 33)*2.0-1.0

        pitchTgt = self.XPID.control(obs[0]*400.0, obs[3]*100.0, 0)
        gimbalX = -self.pitchPID.control(obs[8]*180.0, obs[10]*2.0, pitchTgt)/10.0

        rollTgt = self.YPID.control(obs[1]*400.0, obs[4]*100.0, 0)
        gimbalY = -self.rollPID.control(obs[7]*180.0, obs[9]*2.0, -rollTgt)/10.0

        return np.array([throttle,gimbalX,gimbalY])