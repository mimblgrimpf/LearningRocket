# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 20:02:48 2020

@author: Tobias
"""


class RocketEngine:
    
    ev_SL = 200*9.806
    ev_Vac = 250*9.806
    F_max = 200
    F_min = 0.4
    
    def __init__(self,ev_SL,ev_Vac,F_max,F_min):
        self.ev_SL = ev_SL
        self.ev_Vac = ev_Vac
        self.F_max = F_max
        self.F_min = F_min
    
    def setThrottle(self, throttle):
        if 0 < throttle < self.F_min:
            throttle = self.F_min
        thrust = throttle*self.F_max
        mdot = thrust/self.ev_SL
        
        return [0,0,thrust],mdot
        