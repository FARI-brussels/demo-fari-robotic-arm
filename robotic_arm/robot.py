import time
import traceback
import roboticstoolbox as rtb
import numpy as np
import spatialmath as sm
import spatialgeometry as sg
from scipy.spatial.transform import Rotation as R
from pydrake.solvers import MathematicalProgram, Solve
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
        LITE6.q = LITE6.qz
        env.step(0.1)


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

    def draw_x(self, x, y, z, length, rest_position=(0,0,20), lift_height=10.0, tcp_speed=30, tcp_acc=1000):
        try:
            self._arm.close_lite6_gripper()
            self._tcp_speed = tcp_speed
            self._tcp_acc = tcp_acc

            # Calculate end points of the first line
            x1, y1 = x - length / 2, y - length / 2
            x2, y2 = x + length / 2, y + length / 2

            # Calculate end points of the second line
            x3, y3 = x - length / 2, y + length / 2
            x4, y4 = x + length / 2, y - length / 2

            # Z coordinate while drawing
            z_lifting = z + lift_height  # Z coordinate while lifting

            # Draw the first line
            self.move_to(x1, y1, z_lifting, 200.0, 0.0, 0.0, modes=["simulation"] )
            self._arm.set_position(x1, y1, z, 200.0, 0.0, 0.0, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=20.0, wait=True)
            self._arm.set_position(x2, y2, z, 200.0, 0.0, 0.0, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=20.0, wait=True)

            # Lift, move to the start of the second line, and lower
            self._arm.set_position(x2, y2, z_lifting, 200.0, 0.0, 0.0, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=20.0, wait=True)
            self._arm.set_position(x3, y3, z_lifting, 200.0, 0.0, 0.0, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=20.0, wait=True)
            self._arm.set_position(x3, y3, z, 200.0, 0.0, 0.0, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=20.0, wait=True)

            # Draw the second line
            self._arm.set_position(x4, y4, z, 200.0, 0.0, 0.0, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=20.0, wait=True)
            self._arm.set_position(x4, y4, z_lifting, 200.0, 0.0, 0.0, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=20.0, wait=True)
        
        except Exception as e:
            self.pprint('MainException: {}'.format(e))
        finally:
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)

    def grab_pen(self, pen_position, rest_position=(116,192,53), lift_height=50.0, tcp_speed=30, tcp_acc=1000):
        try:
            self._tcp_speed = tcp_speed
            self._tcp_acc = tcp_acc
            self._arm.open_lite6_gripper()
            self._arm.set_position(pen_position[0], pen_position[1], pen_position[2]+lift_height, -180, 0.0, 90, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=50.0, wait=True)
            self._arm.set_position(pen_position[0], pen_position[1], pen_position[2], -180, 0.0, 90, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=50.0, wait=True)
            self._arm.close_lite6_gripper()
            time.sleep(1)
            self._arm.set_position(pen_position[0], pen_position[1], pen_position[2]+lift_height, -180, 0.0, 50, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=50.0, wait=True)
            self._arm.set_position(*rest_position, 200, 0.0, 0.0, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=50.0, wait=False)
        except Exception as e:
            self.pprint('MainException: {}'.format(e))
        finally:
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)

    def store_pen(self, pen_position, rest_position=(116,192,53), lift_height=50.0, tcp_speed=30, tcp_acc=1000):
        try:
            self._tcp_speed = tcp_speed
            self._tcp_acc = tcp_acc
            self._arm.set_position(pen_position[0], pen_position[1], pen_position[2]+lift_height, -180, 0.0, 90, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=20.0, wait=True)
            self._arm.set_position(pen_position[0], pen_position[1], pen_position[2]+lift_height/4, -180, 0.0, 90, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=20.0, wait=True)
            self._arm.open_lite6_gripper()
            self._arm.set_position(pen_position[0], pen_position[1], pen_position[2]+lift_height, -180, 0.0, 90, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=20.0, wait=True)
            self._arm.set_position(*rest_position,-180, 0.0, 90, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=20.0, wait=False)
            self._arm.stop_lite6_gripper()
        except Exception as e:
            self.pprint('MainException: {}'.format(e))
        finally:
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)



class Lite6:
    def __init__(self, simulation, robot_ip = None, tcp_offset = None) -> None:
        self.virtual_robot = rtb.models.URDF.Lite6()
        self.simulation = simulation
        if self.simulation:
            self.simulation.add(self.virtual_robot, robot_alpha=True, collision_alpha=False)
        self.tcp_offset = tcp_offset
        self.real_robot = None
        if robot_ip:
            from xarm.wrapper import XArmAPI
            self.real_robot = RobotMain(XArmAPI(robot_ip, baud_checkset=False))

    def move_to(self, dest, dt=0.05, gain=1, treshold=0.01):
        if self.tcp_offset:
            dest = end_effector_base_position_from_tip(dest, self.tcp_offset)
        if self.simulation:
            axes = sg.Axes(length=0.1, pose=dest)
            self.simulation.add(axes)
        if self.real_robot:
            self.real_robot.set_mode(4)
            self.real_robot.set_state(0)
        arrived = False
        while not arrived:
            v, arrived = rtb.p_servo(self.virtual_robot.fkine(self.virtual_robot.q), dest, gain=gain, threshold=treshold)
            qd = jacobian_i_k_optimisation(self.virtual_robot, v, v_max=1)[1]
            self.virtual_robot.qd = qd
            if self.real_robot:
                self._arm.vc_set_joint_velocity(qd, is_radian=True)
            if self.simulation:
                self.simulation.step(dt)
        if self.real_robot:
            self._arm.vc_set_joint_velocity([0, 0, 0, 0, 0, 0], is_radian=True)
        return arrived
    
    def get_pose(self):
        return self.virtual_robot.fkine(self.virtual_robot.q)
    
    def reset(self):
        self.virtual_robot.q = self.virtual_robot.qz
        self.simulation.step(0.1)