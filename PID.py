# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 12:29:46 2020

@author: Tobias
"""


class PID:
    Kp = 1
    Ki = 0.05
    Kd = 0.5
    
    minVal = 0
    maxVal = 1
    
    P = 0
    I = 0
    D = 0
    
    def __init__(self,Kp,Ki,Kd,minVal,maxVal):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.minVal = minVal
        self.maxVal = maxVal
        
    def control(self,offset,vel,target):
        self.P = self.Kp*(target-offset)
        self.I = self.I + self.Ki*(target-offset)
        self.D = -self.Kd*vel
        
        if self.I < self.minVal:
            self.I = self.minVal
        elif self.I > self.maxVal:
            self.I = self.maxVal
        
        out = self.P+self.I+self.D
        
        if out < self.minVal:
            out = self.minVal
        elif out > self.maxVal:
            out = self.maxVal
        
        return out
    
    def resetI(self):
        self.I = 0