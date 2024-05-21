
# %%
import time
import roboticstoolbox as rtb
import numpy as np
import spatialmath as sm
import spatialgeometry as sg
from pydrake.solvers import MathematicalProgram, Solve
# conda install -c conda-forge libstdcxx-ng=12
import swift
from robotsAPI import Lite6API

def jacobian_i_k_optimisation(robot, v, v_max=1):
    # jacobian inverse kinematics with optimisation
    J = robot.jacobe(robot.q)
    prog = MathematicalProgram()
    v_opt = prog.NewContinuousVariables(6, "v_opt")
    # Define the error term for the cost function
    error = J @ v_opt - v
    prog.AddCost(error.dot(error))
    # Add bounding box constraint for joint velocities
    lower_bounds = [-v_max] * 6  # Lower bounds for each joint velocity
    upper_bounds = [v_max] * 6   # Upper bounds for each joint velocity
    prog.AddBoundingBoxConstraint(lower_bounds, upper_bounds, v_opt)
    # Solve the optimization problem
    result = Solve(prog)
    return result.is_success(), result.GetSolution(v_opt)



class OXOPlayer:
    def __init__(self, robot, api=None, simulation=None, scene=None):
        self.robot = robot
        self.api = api
        self.simulation = simulation
        self.scene = scene
        if self.simulation:
            self.simulation.launch(realtime=True)
            self.simulation.add(self.robot)
            for ob in scene:
                self.simulation.add(ob)
        
    def move_to(self, dest, dt=0.05, gain=2, treshold=0.001): 
        arrived = False
        while not arrived:
            if self.api:
                q = self.api.get_joint_positions(is_radian=True)
                self.robot.q = q
            else:
                q = self.robot.q
            v, arrived = rtb.p_servo(self.robot.fkine(q), dest, gain=gain, threshold=treshold)
            qd = jacobian_i_k_optimisation(self.robot, v, v_max=1)[1]
            self.robot.qd = qd
            if self.api:
                self.api.set_joint_velocity(qd, is_radian=True)
                if not self.simulation:
                    time.sleep(dt)
                else:
                    self.simulation.step(0)
            else: 
                self.simulation.step(dt)
        if self.api:
            self.api.set_joint_velocity([0, 0, 0, 0, 0, 0], is_radian=True)
        return arrived, self.robot.q
    
    def draw_grid(self, grid_center, grid_size, Trest=None, lift_height=0.01):
        for i in [-1, 1]:
            self.move_to(grid_center * sm.SE3(grid_size/6 * i, grid_size/2 * i, -lift_height))
            self.move_to(grid_center * sm.SE3(grid_size/6 * i, grid_size/2 * i, 0))
            self.move_to(grid_center * sm.SE3(grid_size/6 * i, grid_size/2 * -i, 0))
            self.move_to(grid_center * sm.SE3(grid_size/6 * i, grid_size/2 * -i, -lift_height))
        for i in [-1, 1]: 
            self.move_to(grid_center * sm.SE3(grid_size/2 * -i, grid_size/6 * i, -lift_height))
            self.move_to(grid_center * sm.SE3(grid_size/2 * -i, grid_size/6 * i, 0))
            self.move_to(grid_center * sm.SE3(grid_size/2 * i, grid_size/6 * i, 0))
            self.move_to(grid_center * sm.SE3(grid_size/2 * i, grid_size/6 * i, -lift_height))
        if Trest:
            #probably better to implement qrest
            self.move_to(Trest)

    def draw_x(self, center: sm.SE3, length, Trest=None, lift_height=0.01):
        half_length = length / 2
        self.move_to(center * sm.SE3(-half_length, -half_length, -lift_height))
        self.move_to(center * sm.SE3(-half_length, -half_length, 0))
        self.move_to(center * sm.SE3(half_length, half_length, 0))
        self.move_to(center * sm.SE3(half_length, half_length, -lift_height))
        
        self.move_to(center * sm.SE3(-half_length, half_length, -lift_height))
        self.move_to(center * sm.SE3(-half_length, half_length, 0))
        self.move_to(center * sm.SE3(half_length, -half_length, 0))
        self.move_to(center * sm.SE3(half_length, -half_length, -lift_height))
        if Trest:
            self.move_to(Trest)

    def draw_o(self, center: sm.SE3, radius, Trest=None, lift_height=0.01):
        for i in range(50):
            theta = 2 * np.pi * i / 50
            T = center * sm.SE3(radius * np.cos(theta), radius * np.sin(theta), 0)
            self.move_to(T)
        if Trest:
            self.move_to(Trest)


    def execute_joint_trajectory(self, q_traj, duration):
        n_samples = len(q_traj)
        resolution = duration/n_samples
        for q in q_traj:
            if self.simulation:
                self.robot.q = q
                self.simulation.step(resolution)


    def move_to_linear(self, Tdest):
        q0 = self.robot.q
        T0 = self.robot.fkine(q0)
        ctraj1 = rtb.ctraj(T0, Tdest, 10)
        print(T0, Tdest)
        qtraj1 = self.robot.ikine_LM(ctraj1, mask=[1, 1, 1, 0.1, 0.1, 0.1], q0 = q0)
        self.execute_joint_trajectory(qtraj1.q, duration=1)

    
    

table = sg.Mesh(
    filename=str("/home/mrcyme/Documents/FARI/repositories/demo-fari-robotic-arm/robotic_arm/stand.dae"),
    scale=(1.0,) * 3,
    color=[240, 103, 103],
)

#api = Lite6API(ip="192.168.1.159")
table.T = table.T * sm.SE3.Tz(0.7)
sim = swift.Swift()
scene = [table]
lite6 = rtb.models.URDF.Lite6()
lite6.q = lite6.qr
lite6.base *=  sm.SE3.Rz(180, 'deg')* sm.SE3.Tz(0.7)
oxo_player = OXOPlayer(lite6, api=None, simulation=sim, scene=scene)
#screen_center = table.T*sm.SE3.Tx(-0.25)*sm.SE3.Tz(0.05)*sm.SE3.RPY([-180, -180, 0], order='xyz', unit='deg')

screen_center = table.T*sm.SE3.Tx(-0.25)*sm.SE3.Tz(0.2)*sm.SE3.RPY([-180, -180, 0], order='xyz', unit='deg')


# %%
time.sleep(5)
Trest = oxo_player.robot.fkine(oxo_player.robot.qr)
oxo_player.draw_grid(screen_center, 0.1, Trest=Trest)
time.sleep(1)
oxo_player.draw_x(screen_center, 0.02, Trest=Trest)
time.sleep(1)
oxo_player.draw_o(screen_center, 0.02, Trest=Trest)
# %%
