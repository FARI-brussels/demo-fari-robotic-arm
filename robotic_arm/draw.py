# %%
import time
import traceback
import roboticstoolbox as rtb
import numpy as np
import spatialmath as sm
import spatialgeometry as sg
from scipy.spatial.transform import Rotation as R
from pydrake.solvers import MathematicalProgram, Solve
# conda install -c conda-forge libstdcxx-ng=12
import time
import math
import swift

def jacobian_i_k_optimisation(robot, v, v_max=1.2):
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

def end_effector_base_position_from_tip(T_tip ,offset):
    """
    Calculate the end effector base position given the tip position and an arbitrary offset.
    
    :param tip_position: A tuple of (x, y, z) coordinates for the end effector tip.
    :param tip_orientation: A tuple of (roll, pitch, yaw) angles in degrees for the end effector tip orientation.
    :param offset: A tuple of (x_e, y_e, z_e, roll_e, pitch_e, yaw_e) representing the offset of the base from the tip.
    :return: The position of the end effector base as a tuple (x, y, z).
    """
    x_e, y_e, z_e, roll_e, pitch_e, yaw_e = offset
    T_offset = sm.SE3.RPY([roll_e, pitch_e, yaw_e], unit='deg') * sm.SE3(x_e, y_e, z_e)
    
    # Calculate the transformation matrix for the base
    T_base = T_tip * T_offset.inv()
    return T_base

def generate_circular_trajectory(T0, r, n_points=100):
    """
    Generates a circular trajectory in 3D space.
    
    :param T0: Initial pose as an SE3 object.
    :param r: Radius of the circle.
    :param n_points: Number of points to generate along the circle.
    :return: A list of SE3 objects representing poses along the circular trajectory.
    """
    
    # Angle increments for generating points on the circle
    t = np.linspace(0, 1,  n_points, endpoint=False)
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    
    # Circle parameterization in its local XY plane
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros_like(x)  # Assuming the circle lies in the XY plane
    
    # Generate poses along the circular trajectory
    trajectory = []
    for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
        # Construct the translation vector in the local frame of T0
        local_translation = sm.SE3(xi, yi, zi)
        
        # Transform to the global frame using T0
        global_translation = T0 * local_translation

        # Append the global pose to the trajectory
        trajectory.append(global_translation)
    #return rtb.tools.trajectory.Trajectory("circle", trajectory, t)
    return trajectory



class RobotMain(object):
    """Robot Main Class"""
    def __init__(self, robot, **kwargs):
        self.alive = True
        self._arm = robot
        self._tcp_speed = 100
        self._tcp_acc = 2000
        self._angle_speed = 20
        self._angle_acc = 500
        self._vars = {}
        self._funcs = {}
        self._robot_init()
        self.q = self._arm.angles

    # Robot init
    def _robot_init(self):
        self._arm.clean_warn()
        self._arm.clean_error()
        self._arm.motion_enable(True)
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(1)
        self._arm.register_error_warn_changed_callback(self._error_warn_changed_callback)
        self._arm.register_state_changed_callback(self._state_changed_callback)
        if hasattr(self._arm, 'register_count_changed_callback'):
            self._arm.register_count_changed_callback(self._count_changed_callback)
        self._arm.reset(wait=True)

    def _reset(self):
        self._arm.reset(wait=True)


    # Register error/warn changed callback
    def _error_warn_changed_callback(self, data):
        if data and data['error_code'] != 0:
            self.alive = False
            self.pprint('err={}, quit'.format(data['error_code']))
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)

    # Register state changed callback
    def _state_changed_callback(self, data):
        if data and data['state'] == 4:
            self.alive = False
            self.pprint('state=4, quit')
            self._arm.release_state_changed_callback(self._state_changed_callback)

    # Register count changed callback
    def _count_changed_callback(self, data):
        if self.is_alive:
            self.pprint('counter val: {}'.format(data['count']))

    def _check_code(self, code, label):
        if not self.is_alive or code != 0:
            self.alive = False
            ret1 = self._arm.get_state()
            ret2 = self._arm.get_err_warn_code()
            self.pprint('{}, code={}, connected={}, state={}, error={}, ret1={}. ret2={}'.format(label, code, self._arm.connected, self._arm.state, self._arm.error_code, ret1, ret2))
        return self.is_alive

    @staticmethod
    def pprint(*args, **kwargs):
        try:
            stack_tuple = traceback.extract_stack(limit=2)[0]
            print('[{}][{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), stack_tuple[1], ' '.join(map(str, args))))
        except:
            print(*args, **kwargs)

    @property
    def arm(self):
        return self._arm

    @property
    def VARS(self):
        return self._vars

    @property
    def FUNCS(self):
        return self._funcs

    @property
    def is_alive(self):
        if self.alive and self._arm.connected and self._arm.error_code == 0:
            if self._arm.state == 5:
                cnt = 0
                while self._arm.state == 5 and cnt < 5:
                    cnt += 1
                    time.sleep(0.1)
            return self._arm.state < 4
        else:
            return False

    

    
class Lite6:
    def __init__(self, simulation, robot_ip = None, tcp_offset = None) -> None:
        self.virtual_robot = rtb.models.URDF.Lite6()
        self.simulation = simulation
        self.tcp_offset = tcp_offset
        self.real_robot = None
        if robot_ip:
            from xarm.wrapper import XArmAPI
            self.real_robot = RobotMain(XArmAPI(robot_ip, baud_checkset=False))
            print(self.real_robot)

    

    def move_to_cartesian_position(self, dest, dt=0.05, gain=2, treshold=0.001, offset=True):
        if self.tcp_offset and offset:
            dest = end_effector_base_position_from_tip(dest, self.tcp_offset)
        if self.simulation:
            axes = sg.Axes(length=0.1, pose=dest)
            self.simulation.add(axes)
        if self.real_robot:
            self.real_robot._arm.set_mode(4)
            self.real_robot._arm.set_state(0)
        arrived = False
        while not arrived:
            if self.real_robot:
                q = self.real_robot._arm.get_servo_angle(is_radian=True)[1][:6]
                self.virtual_robot.q = q
            else:
                q = self.virtual_robot.q
            v, arrived = rtb.p_servo(self.virtual_robot.fkine(q), dest, gain=gain, threshold=treshold)
            qd = jacobian_i_k_optimisation(self.virtual_robot, v, v_max=1)[1]
            self.virtual_robot.qd = qd
            if self.real_robot:
                self.real_robot._arm.vc_set_joint_velocity(qd, is_radian=True)
            if self.simulation:
                self.simulation.step(dt)
        if self.real_robot:
            self.real_robot._arm.vc_set_joint_velocity([0, 0, 0, 0, 0, 0], is_radian=True)
        return arrived, self.virtual_robot.q


    def move_to_joint_position(self, q_dest, q0):
        pass


    def execute_joint_trajectory(self, q_traj, duration):
        n_samples = len(q_traj)
        resolution = duration/n_samples
        if self.real_robot:
            self.real_robot._arm.set_mode(1)
            self.real_robot._arm.set_state(0)
        for q in q_traj:
            if self.simulation:
                self.virtual_robot.q = q
                self.simulation.step(resolution)
            if self.real_robot:
                self.real_robot._arm.set_servo_angle_j(q, is_radian=True)
                time.sleep(resolution)
    
    def get_pose(self):
        return self.virtual_robot.fkine(self.virtual_robot.q)
    
    def reset(self):
        if self.simulation:
            self.virtual_robot.q = self.virtual_robot.qz
            self.simulation.step(0.1)
        if self.real_robot:
            self.real_robot._reset()
            
    def ctraj(self, T0, T1, n_samples):
        if self.tcp_offset:
            T0 = end_effector_base_position_from_tip(T0, self.tcp_offset)
            T1 = end_effector_base_position_from_tip(T1, self.tcp_offset)
        return rtb.ctraj(T0, T1, n_samples)
    
    def circ_ctraj(self, Tcenter, radius, n_samples):
        if self.tcp_offset:
            Tcenter= end_effector_base_position_from_tip(Tcenter, self.tcp_offset)
        return generate_circular_trajectory(Tcenter, radius, n_samples)
        
    def draw_x(self, center: sm.SE3, length, Trest=None, lift_height=0.01):
        half_length = length / 2
        # End points of the first line
        T1 = center * sm.SE3(-half_length, -half_length, 0)
        T2 = center * sm.SE3(half_length, half_length, 0)   # Translated from center by half_length in X and Y
        T3_high = center * sm.SE3(-half_length, half_length, - lift_height) 
        print(T3_high)
        # End points of the second line
        T3 = center * sm.SE3(-half_length, half_length, 0)  # Translated from center by -half_length in X and half_length in Y
        T4 = center * sm.SE3(half_length, -half_length, 0)  # Translated from center by half_length in X and -half_length in 

        _, q0 = self.move_to_cartesian_position(T1)

        ctraj1 = self.ctraj(T1, T2, 100)
        qtraj1 = self.virtual_robot.ikine_LM(ctraj1, mask=[1, 1, 1, 0.1, 0.1, 0.1], q0 = q0)
        self.execute_joint_trajectory(qtraj1.q, duration=1)
        self.move_to_cartesian_position(T3_high)
        _, q0 = self.move_to_cartesian_position(T3)
        ctraj2 = self.ctraj(T3, T4, 100)
        qtraj2 = self.virtual_robot.ikine_LM(ctraj2, mask=[1, 1, 1, 0.1
                                                           , 0.1, 0.1], q0=q0)
        self.execute_joint_trajectory(qtraj2.q, duration=1)
        self.move_to_cartesian_position(Trest) 

    def draw_o(self, center: sm.SE3, radius, Trest=None, lift_height=0.01):
        ctraj = self.circ_ctraj(center, radius, 100)
        _, q0 = self.move_to_cartesian_position(ctraj[0], offset=False)
        qtraj = []
        for T in ctraj:
            q0 = self.virtual_robot.ikine_LM(T, mask=[1, 1, 1, 0.1, 0.1, 0.1], q0=q0).q
            qtraj.append(q0)
        self.execute_joint_trajectory(qtraj, duration=1)
        self.move_to_cartesian_position(Trest)     

    
    
        

tcp_offset = [0, 0, 0.2, 0, 0, 0]


sim = swift.Swift()
sim.launch(realtime=True)
lite6 = Lite6(simulation=sim, tcp_offset=tcp_offset)
lite6.virtual_robot.base *=  sm.SE3.Rz(180, 'deg')* sm.SE3.Tz(0.7)
#lite6 = Lite6(simulation=sim, robot_ip="192.168.1.159",  tcp_offset=tcp_offset)
table = sg.Mesh(
    filename=str("/home/mrcyme/Documents/FARI/repositories/demo-fari-robotic-arm/robotic_arm/stand.dae"),
    scale=(1.0,) * 3,
    color=[240, 103, 103],
)
table.T = table.T * sm.SE3.Tz(0.7)
sim.add(table)
sim.add(lite6)



T0 = sm.SE3(0.1, 0.1, 0)*sm.SE3.RPY([-180, -180, 0], order='xyz', unit='deg')
T1 = sm.SE3(0.1, 0, 0)*sm.SE3.RPY([-180, -180, 0], order='xyz', unit='deg')
lite6.grab_pen(T0, Trest=lite6.virtual_robot.fkine(lite6.virtual_robot.qz))
#lite6.draw_o(T0, 0.005, Trest=lite6.virtual_robot.fkine(lite6.virtual_robot.qz))
lite6.draw_x(T1, 0.005, Trest=lite6.virtual_robot.fkine(lite6.virtual_robot.qz))
#lite6.store_pen(T0, Trest=lite6.virtual_robot.fkine(lite6.virtual_robot.qz))