import math
import random as r
import sys
import numpy as np

from direct.directtools.DirectGeometry import LineNodePath
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.InputStateGlobal import inputState
from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock
from panda3d.bullet import BulletConeShape
from panda3d.bullet import BulletCylinderShape
from panda3d.bullet import BulletDebugNode
from panda3d.bullet import BulletPlaneShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletSliderConstraint
from panda3d.bullet import BulletWorld
from panda3d.bullet import XUp
from panda3d.bullet import ZUp
from panda3d.core import AmbientLight, NodePath, PandaNode
from panda3d.core import BitMask32
from panda3d.core import DirectionalLight
from panda3d.core import Point3
from panda3d.core import TextNode
from panda3d.core import TransformState
from panda3d.core import Vec3
from panda3d.core import Vec4

from Environ_dependancies import air_dens
from PID import PID
from RocketEngine import RocketEngine as RE
from LandingRockets.NeuralNetwork import NeuralNetwork

def dot(a, b):
    return a.x * b.x + a.y * b.y + a.z * b.z


def length(a):
    return math.sqrt(a.x ** 2 + a.y ** 2 + a.z ** 2)


def norm(a):
    anorm = Vec3(a)
    anorm.normalize()
    return anorm


class Simulation(ShowBase):
    scale = 1
    height = 500
    lateralError = 200
    dt = 0.1
    steps = 0
    #Test Controllers
    fuelPID = PID(10, 0.5, 10, 0, 100)
    heightPID = PID(0.08, 0, 0.3, 0.1, 1)
    pitchPID = PID(10, 0, 1000, -10, 10)
    rollPID = PID(10, 0, 1000, -10, 10)
    XPID = PID(0.2, 0, 0.8, -10, 10)
    YPID = PID(0.2, 0, 0.8, -10, 10)
    vulcain = NeuralNetwork()
    tau = 0.5
    Valves=[]

    CONTROL = False

    EMPTY = False
    LANDED = False
    DONE = False

    gimbalX = 0
    gimbalY = 0
    targetAlt = 150

    R = RE(200 * 9.806, 250 * 9.806, 7607000 / 9 * scale, 0.4)
    throttle = 0.0
    fuelMass_full = 417000 * scale
    fuelMass_init = 0.10

    radius = 1.8542 * scale
    length = 47 * scale
    Cd = 1.5


    def __init__(self, VISUALIZE=False):

        self.VISUALIZE = VISUALIZE


        if VISUALIZE is True:
            ShowBase.__init__(self)
            self.setBackgroundColor(0.2, 0.2, 0.2, 1)
            self.setFrameRateMeter(True)
            self.render.setShaderAuto()
            self.cam.setPos(-120 * self.scale, -120 * self.scale, 120 * self.scale)
            self.cam.lookAt(0, 0, 10 * self.scale)
            alight = AmbientLight('ambientLight')
            alight.setColor(Vec4(0.5, 0.5, 0.5, 1))
            alightNP = self.render.attachNewNode(alight)

            dlight = DirectionalLight('directionalLight')
            dlight.setDirection(Vec3(1, -1, -1))
            dlight.setColor(Vec4(0.7, 0.7, 0.7, 1))
            dlightNP = self.render.attachNewNode(dlight)

            self.render.clearLight()
            self.render.setLight(alightNP)
            self.render.setLight(dlightNP)

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
            self.ostData = OnscreenText(text='ready', pos=(-1.3, 0.9), scale=0.07, fg=Vec4(1, 1, 1, 1),
            align=TextNode.ALeft)

        self.fuelMass = self.fuelMass_full * self.fuelMass_init
        self.vulcain.load_existing_model(path="LandingRockets/",model_name="140k_samples_1024neurons_3layers_l2-0.000001")



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
        ...#self.toggleWireframe()

    def toggleTexture(self):
        ...#self.toggleTexture()

    def toggleDebug(self):
        if self.debugNP.isHidden():
            self.debugNP.show()
        else:
            self.debugNP.hide()

    def doScreenshot(self):
        self.screenshot('Bullet')

    # ____TASK___

    def processInput(self):
        throttleChange = 0.0

        if inputState.isSet('forward'): self.gimbalY = 10
        if inputState.isSet('reverse'): self.gimbalY = -10
        if inputState.isSet('left'):    self.gimbalX = 10
        if inputState.isSet('right'):   self.gimbalX = -10
        if inputState.isSet('turnLeft'):  throttleChange = -1.0
        if inputState.isSet('turnRight'): throttleChange = 1.0

        self.throttle += throttleChange / 100.0
        self.throttle = min(max(self.throttle, 0), 1)

    def processContacts(self):
        result = self.world.contactTestPair(self.rocketNP.node(), self.groundNP.node())
        #print(result.getNumContacts())
        if result.getNumContacts() != 0:
            self.LANDED = True
            #self.DONE = True

    def update(self,task):

        #self.control(0.1,0.1,0.1)
        #self.processInput()
        #self.processContacts()
        # self.terrain.update()
        return task.cont

    def cleanup(self):
        self.world.removeRigidBody(self.groundNP.node())
        self.world.removeRigidBody(self.rocketNP.node())
        self.world = None

        self.debugNP = None
        self.groundNP = None
        self.rocketNP = None

        self.worldNP.removeNode()

        self.LANDED = False
        self.EMPTY = False
        self.DONE = False
        self.steps = 0
        self.fuelMass = self.fuelMass_full*self.fuelMass_init

    def setup(self):

        self.targetAlt = r.randrange(100,300)
        self.Valves = np.array([0.15,0.2,0.15])
        self.EngObs = self.vulcain.predict_data_point(np.array(self.Valves).reshape(1,-1))

        if self.VISUALIZE is True:
            self.worldNP = self.render.attachNewNode('World')

        else:
            self.root = NodePath(PandaNode("world root"))
            self.worldNP = self.root.attachNewNode('World')






        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.80665))

        # World
        self.debugNP = self.worldNP.attachNewNode(BulletDebugNode('Debug'))
        self.debugNP.node().showWireframe(True)
        self.debugNP.node().showConstraints(True)
        self.debugNP.node().showBoundingBoxes(False)
        self.debugNP.node().showNormals(True)
        self.debugNP.show()
        self.world.setDebugNode(self.debugNP.node())

        # self.debugNP.showTightBounds()
        # self.debugNP.showBounds()


        # Ground (static)
        shape = BulletPlaneShape(Vec3(0, 0, 1), 1)

        self.groundNP = self.worldNP.attachNewNode(BulletRigidBodyNode('Ground'))
        self.groundNP.node().addShape(shape)
        self.groundNP.setPos(0, 0, 0)
        self.groundNP.setCollideMask(BitMask32.allOn())

        self.world.attachRigidBody(self.groundNP.node())

        # Rocket
        shape = BulletCylinderShape(self.radius, self.length, ZUp)

        self.rocketNP = self.worldNP.attachNewNode(BulletRigidBodyNode('Cylinder'))
        self.rocketNP.node().setMass(27200 * self.scale)
        self.rocketNP.node().addShape(shape)
        #self.rocketNP.setPos(20,20,250)
        self.rocketNP.setPos(20, 20, r.randrange(100, 300)+200)
        #self.rocketNP.setPos(r.randrange(-self.lateralError, self.lateralError, 1), r.randrange(-self.lateralError, self.lateralError, 1), self.height)
        # self.rocketNP.setPos(0, 0, self.length*10)
        self.rocketNP.setCollideMask(BitMask32.allOn())
        # self.rocketNP.node().setCollisionResponse(0)
        self.rocketNP.node().notifyCollisions(True)

        self.world.attachRigidBody(self.rocketNP.node())

        for i in range(4):
            leg = BulletCylinderShape(0.1 * self.radius, 0.5 * self.length, XUp)
            self.rocketNP.node().addShape(leg, TransformState.makePosHpr(
                Vec3(6 * self.radius * math.cos(i * math.pi / 2), 6 * self.radius * math.sin(i * math.pi / 2),
                     -0.6 * self.length), Vec3(i * 90, 0, 30)))

        shape = BulletConeShape(0.75 * self.radius, 0.5 * self.radius, ZUp)
        self.rocketNP.node().addShape(shape, TransformState.makePosHpr(Vec3(0, 0, -1 / 2 * self.length), Vec3(0, 0, 0)))

        # Fuel
        self.fuelRadius = 0.9 * self.radius
        shape = BulletCylinderShape(self.fuelRadius, 0.01 * self.length, ZUp)
        self.fuelNP = self.worldNP.attachNewNode(BulletRigidBodyNode('Cone'))
        self.fuelNP.node().setMass(self.fuelMass_full * self.fuelMass_init)
        self.fuelNP.node().addShape(shape)
        self.fuelNP.setPos(0, 0, self.rocketNP.getPos().getZ() - self.length * 0.5 * (1 - self.fuelMass_init))
        self.fuelNP.setCollideMask(BitMask32.allOn())
        self.fuelNP.node().setCollisionResponse(0)

        self.world.attachRigidBody(self.fuelNP.node())

        frameA = TransformState.makePosHpr(Point3(0, 0, 0), Vec3(0, 0, 90))
        frameB = TransformState.makePosHpr(Point3(0, 0, 0), Vec3(0, 0, 90))

        self.fuelSlider = BulletSliderConstraint(self.rocketNP.node(), self.fuelNP.node(), frameA, frameB, 1)
        self.fuelSlider.setTargetLinearMotorVelocity(0)
        self.fuelSlider.setDebugDrawSize(2.0)
        self.fuelSlider.set_lower_linear_limit(0)
        self.fuelSlider.set_upper_linear_limit(0)
        self.world.attachConstraint(self.fuelSlider)

        self.npThrustForce = LineNodePath(self.rocketNP, 'Thrust', thickness=4, colorVec=Vec4(1, 0.5, 0, 1))
        self.npDragForce = LineNodePath(self.rocketNP, 'Drag', thickness=4, colorVec=Vec4(1, 0, 0, 1))
        self.npLiftForce = LineNodePath(self.rocketNP, 'Lift', thickness=4, colorVec=Vec4(0, 0, 1, 1))
        self.npFuelState = LineNodePath(self.fuelNP, 'Fuel', thickness=20, colorVec=Vec4(0, 1, 0, 1))

        self.rocketCSLon = self.radius ** 2 * math.pi
        self.rocketCSLat = self.length * 2 * self.radius

        if self.VISUALIZE is True:
            self.terrain = loader.loadModel("LZGrid2.egg")
            self.terrain.setScale(10)
            self.terrain.reparentTo(self.render)
            self.terrain.setColor(Vec4(0.1, 0.2, 0.1, 1))
            self.toggleTexture()

        #self.fuelNP.setPos(0, 0, self.rocketNP.getPos().getZ() - self.length * 0.4 * (1 - self.fuelMass_init))

        for i in range(5):
            self.world.doPhysics(self.dt, 5, 1.0 / 180.0)

        self.fuelSlider.set_lower_linear_limit(-self.length * 0.5 * (1 - self.fuelMass_init))
        self.fuelSlider.set_upper_linear_limit(self.length * 0.5 * (1 - self.fuelMass_init))

        for i in range(100):
            self.world.doPhysics(self.dt, 5, 1.0 / 180.0)
            self.rocketNP.node().applyForce(Vec3(0,0,300000), Vec3(0, 0, 0))








    def updateRocket(self, mdot, dt):

        # Fuel Update
        self.fuelMass = self.fuelNP.node().getMass() - dt * mdot
        if self.fuelMass <= 0:
            self.EMPTY is True
        fuel_percent = self.fuelMass / self.fuelMass_full
        self.fuelNP.node().setMass(self.fuelMass)
        fuelHeight = self.length * fuel_percent
        I1 = 1 / 2 * self.fuelMass * self.fuelRadius ** 2
        I2 = 1 / 4 * self.fuelMass * self.fuelRadius ** 2 + 1 / 12 * self.fuelMass * fuelHeight * fuelHeight
        self.fuelNP.node().setInertia(Vec3(I2, I2, I1))

        # Shift fuel along slider constraint
        fuelTargetPos = 0.5 * (self.length - fuelHeight)
        fuelPos = self.fuelSlider.getLinearPos()
        self.fuelSlider.set_upper_linear_limit(fuelTargetPos)
        self.fuelSlider.set_lower_linear_limit(-fuelTargetPos)

        self.npFuelState.reset()
        self.npFuelState.drawArrow2d(Vec3(0, 0, -0.5 * fuelHeight), Vec3(0, 0, 0.5 * fuelHeight), 45, 2)
        self.npFuelState.create()

    def observe(self):
        pos = self.rocketNP.getPos()
        vel = self.rocketNP.node().getLinearVelocity()
        quat = self.rocketNP.getTransform().getQuat()
        Roll, Pitch, Yaw = quat.getHpr()
        rotVel = self.rocketNP.node().getAngularVelocity()
        offset = self.targetAlt-pos.getZ()

        return pos, vel, Roll, Pitch, Yaw, rotVel, self.fuelMass / self.fuelMass_full, self.EMPTY, self.DONE, self.LANDED, offset, self.EngObs[0], self.Valves

    def control(self,ValveCommands):

        self.gimbalX = 0
        self.gimbalY = 0

        self.Valves = ValveCommands-(ValveCommands-self.Valves)*np.exp(-self.dt/self.tau)

        self.EngObs = self.vulcain.predict_data_point(np.array(self.Valves).reshape(1,-1 ))
        #Brennkammerdruck, Gaskammertemp, H2Massenstrom, LOX MAssentrom, Schub
        #Bar,K,Kg/s,Kg/s,kN
        #self.dt = globalClock.getDt()

        pos = self.rocketNP.getPos()
        vel = self.rocketNP.node().getLinearVelocity()
        quat = self.rocketNP.getTransform().getQuat()
        Roll, Pitch, Yaw = quat.getHpr()
        rotVel = self.rocketNP.node().getAngularVelocity()

        # CHECK STATE
        if self.fuelMass <= 0:
            self.EMPTY = True
        #if pos.getZ() <= 36:
        #    self.LANDED = True
        self.LANDED = False
        self.processContacts()

        P, T, rho = air_dens(pos[2])
        rocketZWorld = quat.xform(Vec3(0, 0, 1))

        AoA = math.acos(min(max(dot(norm(vel), norm(-rocketZWorld)), -1), 1))

        dynPress = 0.5 * dot(vel, vel) * rho

        dragArea = self.rocketCSLon + (self.rocketCSLat - self.rocketCSLon) * math.sin(AoA)

        drag = norm(-vel) * dynPress * self.Cd * dragArea

        time = globalClock.getFrameTime()

        liftVec = norm(vel.project(rocketZWorld) - vel)
        if AoA > 0.5 * math.pi:
            liftVec = -liftVec
        lift = liftVec * (math.sin(AoA * 2) * self.rocketCSLat * dynPress)

        if self.CONTROL:
            self.throttle = self.heightPID.control(pos.getZ(), vel.getZ(), 33)

            pitchTgt = self.XPID.control(pos.getX(), vel.getX(), 0)
            self.gimbalX = -self.pitchPID.control(Yaw, rotVel.getY(), pitchTgt)

            rollTgt = self.YPID.control(pos.getY(), vel.getY(), 0)
            self.gimbalY = -self.rollPID.control(Pitch, rotVel.getX(), -rollTgt)



        self.thrust = self.EngObs[0][4]*1000
        #print(self.EngObs)
        quat = self.rocketNP.getTransform().getQuat()
        quatGimbal = TransformState.makeHpr(Vec3(0, self.gimbalY, self.gimbalX)).getQuat()
        thrust = quatGimbal.xform(Vec3(0, 0, self.thrust))
        thrustWorld = quat.xform(thrust)

        #print(thrustWorld)

        self.npDragForce.reset()
        self.npDragForce.drawArrow2d(Vec3(0, 0, 0), quat.conjugate().xform(drag) / 1000, 45, 2)
        self.npDragForce.create()

        self.npLiftForce.reset()
        self.npLiftForce.drawArrow2d(Vec3(0, 0, 0), quat.conjugate().xform(lift) / 1000, 45, 2)
        self.npLiftForce.create()

        self.npThrustForce.reset()
        if self.EMPTY is False & self.LANDED is False:
            self.npThrustForce.drawArrow2d(Vec3(0, 0, -0.5 * self.length),
                                           Vec3(0, 0, -0.5 * self.length) - thrust / 30000, 45, 2)
            self.npThrustForce.create()

        self.rocketNP.node().applyForce(drag, Vec3(0, 0, 0))
        self.rocketNP.node().applyForce(lift, Vec3(0, 0, 0))
        #print(self.EMPTY,self.LANDED)
        if self.EMPTY is False & self.LANDED is False:
            self.rocketNP.node().applyForce(thrustWorld, quat.xform(Vec3(0, 0, -1 / 2 * self.length)))
            self.updateRocket(self.EngObs[0][2]+self.EngObs[0][3], self.dt)
        self.rocketNP.node().setActive(True)
        self.fuelNP.node().setActive(True)

        self.processInput()

        self.world.doPhysics(self.dt)
        self.steps+=1


        if self.steps > 1000:
            self.DONE = True

        telemetry = []

        telemetry.append('Thrust: {}'.format(int(self.EngObs[0][4])))
        telemetry.append('Fuel: {}%'.format(int(self.fuelMass / self.fuelMass_full * 100.0)))
        telemetry.append('Gimbal: {}'.format(int(self.gimbalX)) + ',{}'.format(int(self.gimbalY)))
        telemetry.append('AoA: {}'.format(int(AoA / math.pi * 180.0)))
        telemetry.append('\nPos: {},{}'.format(int(pos.getX()), int(pos.getY())))
        telemetry.append('Height: {}'.format(int(pos.getZ())))
        telemetry.append('RPY: {},{},{}'.format(int(Roll), int(Pitch), int(Yaw)))
        telemetry.append('Speed: {}'.format(int(np.linalg.norm(vel))))
        telemetry.append('Vel: {},{},{}'.format(int(vel.getX()), int(vel.getY()), int(vel.getZ())))
        telemetry.append('Rot: {},{},{}'.format(int(rotVel.getX()), int(rotVel.getY()), int(rotVel.getZ())))
        telemetry.append('LANDED: {}'.format(self.LANDED))
        telemetry.append('Time: {}'.format(self.steps*self.dt))
        telemetry.append('TARGET: {}'.format(self.targetAlt))
        #print(pos)
        if self.VISUALIZE is True:
            self.ostData.setText('\n'.join(telemetry))
            self.cam.setPos(pos[0] - 120 * self.scale, pos[1] - 120 * self.scale, pos[2] + 80 * self.scale)
            self.cam.lookAt(pos[0], pos[1], pos[2])
            self.taskMgr.step()


    """
    def setVisualization(self,visualize):
        if visualize is True:
            self.VISUALIZE=True
        else:
            self.VISUALIZE=False"""




if __name__ == "__main__":
    #simulation = Simulation(VISUALIZE=False)
    #simulation.CONTROL=True
    simulation2 = Simulation(VISUALIZE=True)
    simulation2.CONTROL = False
    #simulation.taskMgr.add(simulation.update, 'update')

    #print(simulation.taskMgr.getAllTasks())

    #simulation.run()
    for i in range(10000):
        #simulation.control([0.1,0.1,0.1])
        simulation2.control(np.asarray([0.15,0.15,0.15]).reshape(1,-1))
        #pos,_,_,_,_,_,_,_,_=simulation.observe()
        #print(pos)