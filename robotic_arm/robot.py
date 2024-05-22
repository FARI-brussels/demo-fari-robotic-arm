
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
from typing import Union
ArrayLike = Union[list, np.ndarray, tuple, set]

def jacobian_i_k_optimisation(robot, v, qd_max=1):
    # jacobian inverse kinematics with optimisation
    J = robot.jacobe(robot.q)
    prog = MathematicalProgram()
    qd_opt = prog.NewContinuousVariables(6, "v_opt")
    # Define the error term for the cost function
    error = J @ qd_opt - v
    prog.AddCost(error.dot(error))
    # Add bounding box constraint for joint velocities
    lower_bounds = [-qd_max] * 6  # Lower bounds for each joint velocity
    upper_bounds = [qd_max] * 6   # Upper bounds for each joint velocity
    prog.AddBoundingBoxConstraint(lower_bounds, upper_bounds, qd_opt)
    # Solve the optimization problem
    result = Solve(prog)
    return result.is_success(), result.GetSolution(qd_opt)

def joint_servo(q, q_dest, gain: Union[float, ArrayLike] = 1.0, threshold=0.1):
    """
    Joint-based servoing.
    Returns the joint velocities which will cause the robot to approach the desired joint positions.
    
    :param q: The current joint positions of the robot.
    :type q: ndarray
    :param qd: The desired joint positions of the robot.
    :type qd: ndarray
    :param gain: The gain for the controller. Can be a scalar or a vector corresponding to each joint.
    :type gain: float, or array-like
    :param threshold: The threshold or tolerance of the final error between the robot's joint positions and desired joint positions.
    :type threshold: float
    :returns qdd: The joint velocities which will cause the robot to approach qd.
    :rtype qdd: ndarray(n)
    :returns arrived: True if the robot is within the threshold of the final joint positions.
    :rtype arrived: bool
    """
    # Joint position error
    e = q_dest - q
    
    if sm.base.isscalar(gain):
        k = gain * np.eye(len(q))
    else:
        k = np.diag(gain)
    
    # Joint velocities
    qdd = k @ e
    
    arrived = True if np.sum(np.abs(e)) < threshold else False
    
    return qdd, arrived

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
        if self.api:
            robot.q = self.api.get_joint_positions(is_radian=True)
        
    def move_to(self, dest, dt=0.05, gain=2, treshold=0.001, qd_max=0.2): 
        arrived = False
        while not arrived:
            if self.api:
                q = self.api.get_joint_positions(is_radian=True)
                self.robot.q = q
            else:
                q = self.robot.q
            v, arrived = rtb.p_servo(self.robot.fkine(q), dest, gain=gain, threshold=treshold)
            qd = jacobian_i_k_optimisation(self.robot, v, qd_max=qd_max)[1]
            self.robot.qd = qd
            if self.api:
                self.api.set_joint_velocities(qd, is_radian=True)
                if not self.simulation:
                    time.sleep(dt)
                else:
                    self.simulation.step(0)
            else: 
                self.simulation.step(dt)
        if self.api:
            self.api.set_joint_velocities([0, 0, 0, 0, 0, 0], is_radian=True)
        return arrived, self.robot.q
    
    def move_to_q(self, q_dest, dt=0.05, gain=2, treshold=0.001): 
        arrived = False
        while not arrived:
            if self.api:
                q = self.api.get_joint_positions(is_radian=True)
                self.robot.q = q
            else:
                q = self.robot.q
            qd, arrived = joint_servo(q, q_dest, gain=gain, threshold=treshold)
            self.robot.qd = qd
            if self.api:
                self.api.set_joint_velocities(qd, is_radian=True)
                if not self.simulation:
                    time.sleep(dt)
                else:
                    self.simulation.step(0)
            else: 
                self.simulation.step(dt)
        if self.api:
            self.api.set_joint_velocities([0, 0, 0, 0, 0, 0], is_radian=True)
        return arrived, self.robot.q
            
    
    def draw_grid(self, grid_center, grid_size, q_rest=None, lift_height=0.01, qd_max=1):
        for i in [-1, 1]:
            self.move_to(grid_center * sm.SE3(grid_size/6 * i, grid_size/2 * i, -lift_height),  qd_max=qd_max)
            self.move_to(grid_center * sm.SE3(grid_size/6 * i, grid_size/2 * i, 0), qd_max=qd_max)
            self.move_to(grid_center * sm.SE3(grid_size/6 * i, grid_size/2 * -i, 0), qd_max=qd_max)
            self.move_to(grid_center * sm.SE3(grid_size/6 * i, grid_size/2 * -i, -lift_height), qd_max=qd_max)
        for i in [-1, 1]: 
            self.move_to(grid_center * sm.SE3(grid_size/2 * -i, grid_size/6 * i, -lift_height),  qd_max=qd_max)
            self.move_to(grid_center * sm.SE3(grid_size/2 * -i, grid_size/6 * i, 0),  qd_max=qd_max)
            self.move_to(grid_center * sm.SE3(grid_size/2 * i, grid_size/6 * i, 0),  qd_max=qd_max)
            self.move_to(grid_center * sm.SE3(grid_size/2 * i, grid_size/6 * i, -lift_height),  qd_max=qd_max)
        if q_rest.any():
            #probably better to implement qrest
            self.move_to_q(q_rest)

    def draw_x(self, center: sm.SE3, length, q_rest=None, lift_height=0.01, qd_max=1):
        half_length = length / 2
        self.move_to(center * sm.SE3(-half_length, -half_length, -lift_height), qd_max=qd_max)
        self.move_to(center * sm.SE3(-half_length, -half_length, 0), qd_max=qd_max)
        self.move_to(center * sm.SE3(half_length, half_length, 0), qd_max=qd_max)
        self.move_to(center * sm.SE3(half_length, half_length, -lift_height), qd_max=qd_max)
        
        self.move_to(center * sm.SE3(-half_length, half_length, -lift_height), qd_max=qd_max)
        self.move_to(center * sm.SE3(-half_length, half_length, 0), qd_max=qd_max)
        self.move_to(center * sm.SE3(half_length, -half_length, 0), qd_max=qd_max)
        self.move_to(center * sm.SE3(half_length, -half_length, -lift_height), qd_max=qd_max)
        if q_rest.any():
            #probably better to implement qrest
            self.move_to_q(q_rest)

    def draw_o(self, center: sm.SE3, radius, q_rest=None, lift_height=0.01):
        for i in range(50):
            theta = 2 * np.pi * i / 50
            T = center * sm.SE3(radius * np.cos(theta), radius * np.sin(theta), 0)
            self.move_to(T)
        if q_rest.any():
            #probably better to implement qrest
            self.move_to_q(q_rest)


    def move_to_qb(self, q_dest, duration, n_samples=100):
        if self.api:
            q = self.api.get_joint_positions(is_radian=True)
            print(q)
            self.robot.q = q
        else:
            q = self.robot.q
        j_traj = rtb.jtraj(q, q_dest, n_samples)
        dt = duration/n_samples
        for qd in j_traj.qd:
            self.robot.qd = qd
            if self.api:
                self.api.set_joint_velocities(qd, is_radian=True)
                time.sleep(dt)
                if self.simulation:
                    self.simulation.step(dt)
            else: 
                self.simulation.step(dt)

  

table = sg.Mesh(
    filename=str("/home/fari/Documents/demo-fari-robotic-arm/robotic_arm/stand.dae"),
    scale=(1.0,) * 3,
    color=[240, 103, 103],
)

api = Lite6API(ip="192.168.1.159")
#api = None 
table.T = table.T * sm.SE3.Tz(0.7)
sim = swift.Swift()
scene = [table]
lite6 = rtb.models.URDF.Lite6()
lite6.base *=  sm.SE3.Rz(90, 'deg')* sm.SE3.Tz(0.7)
q_rest = [100, -9, 50, -170 ,-40,- 190]
q_rest = np.radians(q_rest)
lite6.q = q_rest
oxo_player = OXOPlayer(lite6, api=api, simulation=sim, scene=scene)
screen_center = table.T*sm.SE3.Tx(-0.25)*sm.SE3.Tz(0.098)*sm.SE3.RPY([0, 180, 0], order='xyz', unit='deg')
#screen_center = table.T*sm.SE3.Tx(-0.25)*sm.SE3.Tz(0.2)*sm.SE3.RPY([-180, 180, 0], order='xyz', unit='deg')

# %% 

q = [1.6057,-0.575,0.8727,0,1.0996,-0.0524]
q = [100, -9, 50, -180 ,-40,- 190]
q_rest = np.radians(q)
print(q_rest)
print(oxo_player.robot.q)
oxo_player.move_to_q(q_rest, )

#oxo_player.move_to(screen_center, qdmax = 0.2)

# %%
time.sleep(1)
oxo_player.draw_grid(screen_center, 0.1, q_rest=q_rest, qd_max=1)
time.sleep(1)
#oxo_player.draw_x(screen_center, 0.02, q_rest=q_rest)
time.sleep(1)
#oxo_player.draw_o(screen_center, 0.02, q_rest=q_rest)
# %%
