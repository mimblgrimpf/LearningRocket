import sys
import direct.directbase.DirectStart
import math
from RocketEngine import RocketEngine as RE

from direct.showbase.DirectObject import DirectObject
from direct.showbase.InputStateGlobal import inputState
from direct.gui.OnscreenText import OnscreenText

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

from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletPlaneShape
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletCylinderShape
from panda3d.bullet import BulletConeShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletConeTwistConstraint
from panda3d.bullet import BulletDebugNode
from panda3d.bullet import ZUp
from panda3d.bullet import XUp

from scipy.spatial.transform import Rotation as rot


class Simulation(DirectObject):
    scale = 10
      
    R = RE
    throttle = 0.0
     
      
    
    def __init__(self):
        base.setBackgroundColor(0.1, 0.1, 0.8, 1)
        base.setFrameRateMeter(True)
    
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
        force = Vec3(0, 0, 0)
        throttleChange = 0.0
        #torque = Vec3(0, 0, 0)
    
        if inputState.isSet('forward'): force.setX(1)
        if inputState.isSet('reverse'): force.setX(-1)
        if inputState.isSet('left'):    force.setY(-1)
        if inputState.isSet('right'):   force.setY( 1)
        if inputState.isSet('turnLeft'):  throttleChange=-1.0
        if inputState.isSet('turnRight'): throttleChange=1.0
    
        force *= -20.0/180.0*math.pi
        self.throttle += throttleChange/100.0
        self.throttle = min(max(self.throttle,0),1)
        thrust=Vec3(0,0,self.throttle*50)
        quatGimbal = self.rocketNozzle.getTransform(self.worldNP).getQuat()
        
        self.rocketNozzle.node().applyForce(quatGimbal.xform(thrust),LPoint3f(0,0,0))
        
        #torque *= 10.0
    
        #force = render.getRelativeVector(self.rocketNP, force)
        #torque = render.getRelativeVector(self.rocketNP, torque)
    
        self.rocketNozzle.node().setActive(True)
        self.rocketNP.node().setActive(True)
        #self.rocketNozzle.node().applyCentralForce(force)
        #self.rocketNozzle.node().applyTorque(torque)
        force = rot.from_euler('zyx',force).as_quat()
        
    
        self.cone.setMotorTarget(LQuaternionf(force[0],force[1],force[2],force[3]))

    def update(self, task):
        dt = globalClock.getDt()
    
        self.processInput(dt)
        #self.world.doPhysics(dt)
        self.world.doPhysics(dt, 5, 1.0/180.0)
        
        yawpitchroll = rot.from_quat(self.rocketNozzle.getTransform(self.rocketNP).getQuat())
        yawpitchroll = yawpitchroll.as_euler('zyx',degrees=True)
         
        #self.ostData.destroy()
        telemetry = []
        telemetry.append('Nozzle Position:\n Yaw: {}\n Pitch: {}\n Roll: {}'.format(
            int(yawpitchroll[0]),int(yawpitchroll[1]),int(yawpitchroll[2])))
        telemetry.append('\nThrottle: {}'.format(self.throttle))
        
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
        self.world.setGravity(Vec3(0, 0, -9.81))
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
        self.rocketNP.node().setMass(3.0)
        self.rocketNP.node().addShape(shape)
        self.rocketNP.setPos(0, 0, 2*self.scale)
        self.rocketNP.setCollideMask(BitMask32.allOn())
    
        self.world.attachRigidBody(self.rocketNP.node())
    
        for i in range(4):
            leg = BulletCylinderShape(0.02*self.scale, 1*self.scale, XUp)
            self.rocketNP.node().addShape(leg, TransformState.makePosHpr(Vec3(0.6*self.scale*math.cos(i*math.pi/2),0.6*self.scale*math.sin(i*math.pi/2),-1.2*self.scale),Vec3(i*90,0,30)))
            
            
        shape =  BulletConeShape(0.15*self.scale,0.3*self.scale, ZUp)
        
        self.rocketNozzle  = self.worldNP.attachNewNode(BulletRigidBodyNode('Cone'))
        self.rocketNozzle.node().setMass(1)
        self.rocketNozzle.node().addShape(shape)
        self.rocketNozzle.setPos(0,0,0.8*self.scale)
        self.rocketNozzle.setCollideMask(BitMask32.allOn())
    
        self.world.attachRigidBody(self.rocketNozzle.node())
        
        frameA = TransformState.makePosHpr(Point3(0, 0, -1*self.scale), Vec3(0, 0, 90))
        frameB = TransformState.makePosHpr(Point3(0, 0, 0.2*self.scale), Vec3(0, 0, 90))
    
        self.cone = BulletConeTwistConstraint(self.rocketNP.node(), self.rocketNozzle.node(), frameA, frameB)
        self.cone.enableMotor(1)
        #self.cone.setMaxMotorImpulse(2)
        #self.cone.setDamping(1000)
        self.cone.setDebugDrawSize(2.0)
        self.cone.setLimit(20, 20, 0, softness=1.0, bias=1.0, relaxation=10.0)
        self.world.attachConstraint(self.cone)
        
        """# Box (dynamic)
        shape = BulletBoxShape(Vec3(0.5, 0.5, 0.5))
    
        self.boxNP = self.worldNP.attachNewNode(BulletRigidBodyNode('Box'))
        self.boxNP.node().setMass(1.0)
        self.boxNP.node().addShape(shape)
        self.boxNP.setPos(0, 0, 2)
        #self.boxNP.setScale(2, 1, 0.5)
        self.boxNP.setCollideMask(BitMask32.allOn())
        #self.boxNP.node().setDeactivationEnabled(False)
    
        self.world.attachRigidBody(self.boxNP.node())"""
        """
        visualNP = loader.loadModel('bullet-samples/models/box.egg')
        visualNP.clearModelNodes()
        visualNP.reparentTo(self.rocketNP)
        """
        # Bullet nodes should survive a flatten operation!
        #self.worldNP.flattenStrong()
        #render.ls()

simulation = Simulation()
base.run()

