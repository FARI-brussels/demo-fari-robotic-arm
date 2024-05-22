from abc import ABC, abstractmethod
import time
class RoboticArmAPI(ABC):
    @abstractmethod
    def __init__(self):
        """
        Initialize the robotic arm.
        """
        pass
    
    @abstractmethod
    def reset_robot(self):
        """
        Reset the robotic arm to its initial state.
        """
        pass
    
    @abstractmethod
    def is_alive(self):
        """
        Check if the robotic arm is still operational.

        Returns:
            bool: True if the robotic arm is operational, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_joint_position(self, joint_id):
        """
        Get the current position of a specific joint.

        Args:
            joint_id (int): The ID of the joint.

        Returns:
            float: The position of the joint.
        """
        pass

    @abstractmethod
    def get_joint_positions(self):
        """
        Get the current positions of all joints.

        Returns:
            list: A list of joint positions.
        """
        pass

    @abstractmethod
    def set_joint_position(self, joint_id, position):
        """
        Set the position of a specific joint.

        Args:
            joint_id (int): The ID of the joint.
            position (float): The desired position of the joint.
        """
        pass

    @abstractmethod
    def set_joint_positions(self, positions):
        """
        Set the positions of all joints.

        Args:
            positions (list): A list of desired joint positions.
        """
        pass

    @abstractmethod
    def get_joint_velocity(self, joint_id):
        """
        Get the current velocity of a specific joint.

        Args:
            joint_id (int): The ID of the joint.

        Returns:
            float: The velocity of the joint.
        """
        pass

    @abstractmethod
    def get_joint_velocities(self):
        """
        Get the current velocities of all joints.

        Returns:
            list: A list of joint velocities.
        """
        pass

    @abstractmethod
    def set_joint_velocity(self, joint_id, velocity):
        """
        Set the velocity of a specific joint.

        Args:
            joint_id (int): The ID of the joint.
            velocity (float): The desired velocity of the joint.
        """
        pass

    @abstractmethod
    def set_joint_velocities(self, velocities):
        """
        Set the velocities of all joints.

        Args:
            velocities (list): A list of desired joint velocities.
        """
        pass

    @abstractmethod
    def get_joint_acceleration(self, joint_id):
        """
        Get the current acceleration of a specific joint.

        Args:
            joint_id (int): The ID of the joint.

        Returns:
            float: The acceleration of the joint.
        """
        pass

    @abstractmethod
    def get_joint_accelerations(self):
        """
        Get the current accelerations of all joints.

        Returns:
            list: A list of joint accelerations.
        """
        pass

    @abstractmethod
    def set_joint_acceleration(self, joint_id, acceleration):
        """
        Set the acceleration of a specific joint.

        Args:
            joint_id (int): The ID of the joint.
            acceleration (float): The desired acceleration of the joint.
        """
        pass

    @abstractmethod
    def set_joint_accelerations(self, accelerations):
        """
        Set the accelerations of all joints.

        Args:
            accelerations (list): A list of desired joint accelerations.
        """
        pass



class Lite6API(RoboticArmAPI):
    def __init__(self, ip, port=None):
        from xarm.wrapper import XArmAPI
        self._api = XArmAPI(f"{ip}", baud_checkset=False)
        self.q = self._api.angles
        self._api.clean_warn()
        self._api.clean_error()
        self._api.motion_enable(True)
        self._api.set_mode(0)
        self._api.set_state(0)
        self._api.register_error_warn_changed_callback(self._error_warn_changed_callback)
        self._api.register_state_changed_callback(self._state_changed_callback)
        if hasattr(self._api, 'register_count_changed_callback'):
            self._api.register_count_changed_callback(self._count_changed_callback)


    def _reset(self):
        self._api.reset(wait=True)


    # Register error/warn changed callback
    def _error_warn_changed_callback(self, data):
        if data and data['error_code'] != 0:
            self.alive = False
            self.pprint('err={}, quit'.format(data['error_code']))
            self._api.release_error_warn_changed_callback(self._error_warn_changed_callback)

    # Register state changed callback
    def _state_changed_callback(self, data):
        if data and data['state'] == 4:
            self.alive = False
            self.pprint('state=4, quit')
            self._api.release_state_changed_callback(self._state_changed_callback)

    # Register count changed callback
    def _count_changed_callback(self, data):
        if self.is_alive:
            self.pprint('counter val: {}'.format(data['count']))

    def _check_code(self, code, label):
        if not self.is_alive or code != 0:
            self.alive = False
            ret1 = self._api.get_state()
            ret2 = self._api.get_err_warn_code()
            self.pprint('{}, code={}, connected={}, state={}, error={}, ret1={}. ret2={}'.format(label, code, self._api.connected, self._api.state, self._api.error_code, ret1, ret2))
        return self.is_alive

    def get_joint_position(self, joint_id, is_radian=True):
        return self._api.get_servo_angle(is_radian=is_radian)[joint_id]

    def get_joint_positions(self, is_radian=True):
        return self._api.get_servo_angle(is_radian=is_radian)[1][:6]

    def set_joint_position(self, joint_id, a):
        # Not optimal as this condition is checked on every call
        if not self._api.get_mode() == 1:
            self._api.set_mode(1)
        return self._api.set_servo_angle(joint_id, a, is_radian=True)


    def set_joint_positions(self, q,  is_radian=True):
        # Not optimal as this condition is checked on every call
        if not self._api.mode == 1:
            self._api.set_mode(1)
        return self._api.set_servo_angle_j(q, is_radian=is_radian)


    def get_joint_velocity(self, joint_id):
        pass

    def get_joint_velocities(self):
        pass

    def set_joint_velocity(self, joint_id, velocity):
        pass

    def set_joint_velocities(self, qd, is_radian=True):
        # Not optimal as this condition is checked on every call
        if not self._api.mode == 4:
            self._api.set_mode(4)
            time.sleep(1)
        if not self._api.state == 0:
            self._api.set_state(0)
        return self._api.vc_set_joint_velocity(qd, is_radian=is_radian)

    def get_joint_acceleration(self, joint_id):
        pass

    def get_joint_accelerations(self):
        pass

    def set_joint_acceleration(self, joint_id, acceleration):
        pass

    def set_joint_accelerations(self, accelerations):
        pass

    def reset_robot(self):
        pass

    def is_alive(self):
        pass

    def open_gripper(self):
        pass


    def open_gripper(self):
        self.api.open_lite6_gripper()
        time.sleep(1)
        self.real_robot._api.stop_lite6_gripper() #stop the gripper motor after opening


    def close_gripper(self, wait=None):
        self.api.close_lite6_gripper()
        if wait:
            time.sleep(wait)
