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
        time.sleep(1)
        self._api.reset(wait=True)

    def get_joint_position(self, joint_id, is_radian=True):
        return self._api.get_servo_angle(is_radian=is_radian)[joint_id]

    def get_joint_positions(self, is_radian=True):
        return self._api.get_servo_angle(is_radian=is_radian)

    def set_joint_position(self, joint_id, a):
        # Not optimal as this condition is checked on every call
        if not self._api.get_mode() == 1:
            self._api.set_mode(1)
        return self._api.set_servo_angle(joint_id, a, is_radian=True)


    def set_joint_positions(self, q,  is_radian=True):
        # Not optimal as this condition is checked on every call
        if not self._api.get_mode() == 1:
            self._api.set_mode(1)
        return self._api.set_servo_angle_j(q, is_radian=True)


    def get_joint_velocity(self, joint_id):
        pass

    def get_joint_velocities(self):
        pass

    def set_joint_velocity(self, joint_id, velocity):
        pass

    def set_joint_velocities(self, velocities):
        # Not optimal as this condition is checked on every call
        if not self._api.get_mode() == 4:
            self._api.set_mode(4)
        return self._api.vc_set_joint_velocity(q, is_radian=True)

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
