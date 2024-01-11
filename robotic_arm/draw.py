#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2022, UFACTORY, Inc.
# All rights reserved.
#
# Author: Vinman <vinman.wen@ufactory.cc> <vinman.cub@gmail.com>

"""
# Notice
#   1. Changes to this file on Studio will not be preserved
#   2. The next conversion will overwrite the file with the same name
# 
# xArm-Python-SDK: https://github.com/xArm-Developer/xArm-Python-SDK
#   1. git clone git@github.com:xArm-Developer/xArm-Python-SDK.git
#   2. cd xArm-Python-SDK
#   3. python setup.py install
"""

import sys
import math
import time
import queue
import datetime
import random
import traceback
import threading
from xarm import version
from xarm.wrapper import XArmAPI


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

    # Robot Main Run
    def draw_o(self, x, y, z, d, rest_position=(0,0,20), lift_height=10.0, tcp_speed=10, tcp_acc=1000):
        try:
            r=d/2
            code = self._arm.close_lite6_gripper()
            print(code)
            # Draw Cricle
            self._tcp_speed = tcp_speed
            self._tcp_acc = tcp_acc
            z_lifting = z + lift_height 
            code = self._arm.set_position(*[x-r, y, z_lifting, 200, 0.0, 0.0], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            code = self._arm.set_position(*[x-r, y, z, 200, 0.0, 0.0], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            code = self._arm.move_circle(*[x, y+r, z, 200, 0.0, 0.0], [x+r, y, z, 200, 0.0, 0.0], float(90), speed=self._tcp_speed, mvacc=self._tcp_acc, wait=True)
            code = self._arm.set_position(*[x-r, y, z_lifting, 200, 0.0, 0.0], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            code = self._arm.set_position(*[x-r, y, z_lifting, 200, 0.0, 0.0], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'move_circle'):
                return
            code = self._arm.set_position(*rest_position, 200, 0.0, 0.0, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        except Exception as e:
            self.pprint('MainException: {}'.format(e))
        self.alive = False
        self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
        self._arm.release_state_changed_callback(self._state_changed_callback)
        if hasattr(self._arm, 'release_count_changed_callback'):
            self._arm.release_count_changed_callback(self._count_changed_callback)

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
            self._arm.set_position(x1, y1, z_lifting, 200.0, 0.0, 0.0, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=20.0, wait=True)
            self._arm.set_position(x1, y1, z, 200.0, 0.0, 0.0, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=20.0, wait=True)
            self._arm.set_position(x2, y2, z, 200.0, 0.0, 0.0, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=20.0, wait=True)

            # Lift, move to the start of the second line, and lower
            self._arm.set_position(x2, y2, z_lifting, 200.0, 0.0, 0.0, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=20.0, wait=True)
            self._arm.set_position(x3, y3, z_lifting, 200.0, 0.0, 0.0, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=20.0, wait=True)
            self._arm.set_position(x3, y3, z, 200.0, 0.0, 0.0, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=20.0, wait=True)

            # Draw the second line
            self._arm.set_position(x4, y4, z, 200.0, 0.0, 0.0, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=20.0, wait=True)
            self._arm.set_position(x4, y4, z_lifting, 200.0, 0.0, 0.0, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=20.0, wait=True)
            self._arm.set_position(*rest_position, 200, 0.0, 0.0, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=20.0, wait=False)

        except Exception as e:
            self.pprint('MainException: {}'.format(e))
        finally:
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)




if __name__ == '__main__':
    RobotMain.pprint('xArm-Python-SDK Version:{}'.format(version.__version__))
    arm = XArmAPI('192.168.1.159', baud_checkset=False)
    robot_main = RobotMain(arm)
    #robot_main.draw_x(100, 294,  0, 20)
    robot_main.draw_o(100, 294,  0, 10)

