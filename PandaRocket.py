import sys
import direct.directbase.DirectStart
import math
from RocketEngine import RocketEngine as RE
from PID import PID

from direct.showbase.ShowBase import ShowBase
from direct.showbase.DirectObject import DirectObject
from direct.showbase.InputStateGlobal import inputState
from direct.gui.OnscreenText import OnscreenText
from direct.directtools.DirectGeometry import LineNodePath

from panda3d.core import AmbientLight
from panda3d.core import DirectionalLight
from panda3d.core import Vec3
from panda3d.core import Vec4
from panda3d.core import Point3
from panda3d.core import TransformState
from panda3d.core import BitMask32
from panda3d.core import TextNode
from panda3d.core import LQuaternionf
from panda3d.core import LPoint3f
from panda3d.core import VBase4

from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletPlaneShape
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletCylinderShape
from panda3d.bullet import BulletConeShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletConeTwistConstraint
from panda3d.bullet import BulletSliderConstraint
from panda3d.bullet import BulletDebugNode
from panda3d.bullet import ZUp
from panda3d.bullet import XUp

from scipy.spatial.transform import Rotation as rot


class Simulation(ShowBase):
    scale = 10
    fuelPID = PID(10,0.5,10,-100,100)
    EMPTY = 0
    
    pitch = 0
    yaw = 0
      
    R = RE(200*9.806,250*9.806,80,0.4)
    throttle = 0.0
    fuelmass_full = 5
      
    
    def __init__(self):
        
        #ShowBase.__init__(self)
        base.setBackgroundColor(0.1, 0.1, 0.8, 1)
        base.setFrameRateMeter(True)
        
        
        self.fuelmass = self.fuelmass_full
    
        base.cam.setPos(-8*self.scale, -8*self.scale, 4*self.scale)
        base.cam.lookAt(0, 0, 1*self.scale)
    
        # Light
        alight = AmbientLight('ambientLight')
        alight.setColor(Vec4(0.5, 0.5, 0.5, 1))
        alightNP = render.attachNewNode(alight)
    
        dlight = DirectionalLight('directionalLight')
        dlight.setDirection(Vec3(1, 1, -1))
        dlight.setColor(Vec4(0.7, 0.7, 0.7, 1))
        dlightNP = render.attachNewNode(dlight)
    
        render.clearLight()
        render.setLight(alightNP)
        render.setLight(dlightNP)
    
        # Input
        self.accept('escape', self.doExit)
        self.accept('r', self.doReset)
        self.accept('f1', self.toggleWireframe)
        self.accept('f2', self.toggleTexture)
        self.accept('f3', self.toggleDebug)
        self.accept('f5', self.doScreenshot)
    
        inputState.watchWithModifiers('forward', 'w')
        inputState.watchWithModifiers('left', 'a')
        inputState.watchWithModifiers('reverse', 's')
        inputState.watchWithModifiers('right', 'd')
        inputState.watchWithModifiers('turnLeft', 'q')
        inputState.watchWithModifiers('turnRight', 'e')
    
        # Task
        taskMgr.add(self.update, 'updateWorld')
        
        self.ostData = OnscreenText(text='ready', pos=(-1.3, 0.9), scale=0.07, fg=Vec4(1,1,1,1),align=TextNode.ALeft)
        
        
    
        # Physics
        self.setup()

      # _____HANDLER_____

    def doExit(self):
        self.cleanup()
        sys.exit(1)
    
    def doReset(self):
        self.cleanup()
        self.setup()
    
    def toggleWireframe(self):
        base.toggleWireframe()
    
    def toggleTexture(self):
        base.toggleTexture()
    
    def toggleDebug(self):
        if self.debugNP.isHidden():
            self.debugNP.show()
        else:
            self.debugNP.hide()
    
    def doScreenshot(self):
        base.screenshot('Bullet')
    
      # ____TASK___
    
    def processInput(self, dt):
        throttleChange = 0.0
        
        if inputState.isSet('forward'): self.gimbalY = 20
        if inputState.isSet('reverse'): self.gimbalY = -20
        if inputState.isSet('left'):    self.gimbalX = 20
        if inputState.isSet('right'):   self.gimbalX = -20
        if inputState.isSet('turnLeft'):  throttleChange=-1.0
        if inputState.isSet('turnRight'): throttleChange=1.0
        
        self.throttle += throttleChange/100.0
        self.throttle = min(max(self.throttle,0),1)
        
        

    def update(self, task):
        dt = globalClock.getDt()
        
        pos = self.rocketNP.getPos()
        quat = self.rocketNP.getTransform().getQuat()
        
        self.gimbalX = 0
        self.gimbalY = 0
    
        self.processInput(dt)
        
        thrust,mdot =  self.R.setThrottle(self.throttle)
        self.updateRocket(mdot,dt)
        
        quat = self.rocketNP.getTransform().getQuat()
        quatGimbal = TransformState.makeHpr(Vec3(0,self.gimbalY,self.gimbalX)).getQuat()
        thrust = quat.xform(quatGimbal.xform(Vec3(thrust[0],thrust[1],thrust[2])))
        
        self.npThrustForce.reset()
        self.npThrustForce.drawArrow2d(Vec3(0,0,-1*self.scale),Vec3(0,0,-1*self.scale)-(thrust)/10, 45, 2)
        self.npThrustForce.create()
        
        self.rocketNP.node().applyForce(thrust,quat.xform(Vec3(0,0,-1*self.scale)))
        self.rocketNP.node().setActive(True)
        self.fuelNP.node().setActive(True)
        
        self.world.doPhysics(dt, 5, 1.0/180.0)
        
        telemetry = []
        
        telemetry.append('Throttle: {}%'.format(int(self.throttle*100)))
        telemetry.append('Fuel: {}%'.format(int(self.fuelmass/self.fuelmass_full*100.0)))
        telemetry.append('Gimbal: {}'.format(int(self.gimbalX))+',{}'.format(int(self.gimbalY)))
        #telemetry.append('[Debug]FuelPos: {}'.format(self.fuelSlider.getLinearPos()))
        
        
        
        self.ostData.setText('\n'.join(telemetry))
        
    
        return task.cont

    def cleanup(self):
        self.world.removeRigidBody(self.groundNP.node())
        self.world.removeRigidBody(self.rocketNP.node())
        self.world = None
    
        self.debugNP = None
        self.groundNP = None
        self.rocketNP = None
    
        self.worldNP.removeNode()

    def setup(self):
        self.worldNP = render.attachNewNode('World')
    
        # World
        self.debugNP = self.worldNP.attachNewNode(BulletDebugNode('Debug'))
        self.debugNP.show()
        self.debugNP.node().showWireframe(True)
        self.debugNP.node().showConstraints(True)
        self.debugNP.node().showBoundingBoxes(False)
        self.debugNP.node().showNormals(True)
    
        #self.debugNP.showTightBounds()
        #self.debugNP.showBounds()
    
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0,-9.81))
        self.world.setDebugNode(self.debugNP.node())
    
        # Ground (static)
        shape = BulletPlaneShape(Vec3(0, 0, 1), 1)
    
        self.groundNP = self.worldNP.attachNewNode(BulletRigidBodyNode('Ground'))
        self.groundNP.node().addShape(shape)
        self.groundNP.setPos(0, 0, -2)
        self.groundNP.setCollideMask(BitMask32.allOn())
    
        self.world.attachRigidBody(self.groundNP.node())
    
    
    
        #Rocket
        shape = BulletCylinderShape(0.2*self.scale, 2*self.scale, ZUp)
    
        self.rocketNP = self.worldNP.attachNewNode(BulletRigidBodyNode('Cylinder'))
        self.rocketNP.node().setMass(1)
        self.rocketNP.node().addShape(shape)
        self.rocketNP.setPos(0, 0, 2*self.scale)
        self.rocketNP.setCollideMask(BitMask32.allOn())
        #self.rocketNP.node().setCollisionResponse(0)
    
        self.world.attachRigidBody(self.rocketNP.node())
    
        for i in range(4):
            leg = BulletCylinderShape(0.02*self.scale, 1*self.scale, XUp)
            self.rocketNP.node().addShape(leg, TransformState.makePosHpr(Vec3(0.6*self.scale*math.cos(i*math.pi/2),0.6*self.scale*math.sin(i*math.pi/2),-1.2*self.scale),Vec3(i*90,0,30)))
            
        shape =  BulletConeShape(0.15*self.scale,0.3*self.scale, ZUp)
        self.rocketNP.node().addShape(shape, TransformState.makePosHpr(Vec3(0,0,-1*self.scale),Vec3(0,0,0)))
        
        
        #Fuel
        shape = BulletCylinderShape(0.15*self.scale, 0.1*self.scale, ZUp)
        self.fuelNP  = self.worldNP.attachNewNode(BulletRigidBodyNode('Cone'))
        self.fuelNP.node().setMass(self.fuelmass)
        self.fuelNP.node().addShape(shape)
        self.fuelNP.setPos(0,0,2*self.scale)
        self.fuelNP.setCollideMask(BitMask32.allOn())
        self.fuelNP.node().setCollisionResponse(0)
    
        self.world.attachRigidBody(self.fuelNP.node())
        
        frameA = TransformState.makePosHpr(Point3(0, 0, 0*self.scale), Vec3(0, 0, 90))
        frameB = TransformState.makePosHpr(Point3(0, 0, 0*self.scale), Vec3(0, 0, 90))
        
        self.fuelSlider = BulletSliderConstraint(self.rocketNP.node(), self.fuelNP.node(), frameA, frameB,1)
        self.fuelSlider.setPoweredLinearMotor(1)
        self.fuelSlider.setTargetLinearMotorVelocity(0)
        self.fuelSlider.setMaxLinearMotorForce(1000)
        #self.cone.setMaxMotorImpulse(2)
        #self.cone.setDamping(1000)
        self.fuelSlider.setDebugDrawSize(2.0)
        self.world.attachConstraint(self.fuelSlider)
        
        
        self.npThrustForce = LineNodePath(self.rocketNP, 'Thrust', thickness=4, colorVec=VBase4(1, 0.5, 0, 1))
        
    def updateRocket(self,mdot,dt):
        
        #Fuel Update
        self.fuelmass = self.fuelmass - dt*mdot
        if self.fuelmass <= 0:
            self.EMPTY = 1
        fuel_percent = self.fuelmass/self.fuelmass_full
        self.fuelNP.node().setMass(self.fuelmass)
        fuelHeight = 2*self.scale*fuel_percent
        I1 = 1/2*self.fuelmass*0.15*self.scale*0.15*self.scale
        I2 = 1/4*self.fuelmass*0.15*0.15*self.scale*self.scale+1/12*self.fuelmass*fuelHeight*fuelHeight
        self.fuelNP.node().setInertia(Vec3(I2,I2,I1))
        
        #Shift fuel along slider constraint
        fuelTargetPos = self.scale-fuelHeight/2
        fuelPos = self.fuelSlider.getLinearPos()
        fuelShift = self.fuelPID.control(fuelPos,0,fuelTargetPos)
        self.fuelSlider.setTargetLinearMotorVelocity(fuelShift)
        
        #print(I1,I2)
        
        

simulation = Simulation()
base.run()

