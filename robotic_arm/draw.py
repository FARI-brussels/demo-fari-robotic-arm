import sys
import math
import time
import queue
import datetime
import random
import traceback
import threading
import swift
import roboticstoolbox as rtb
import numpy as np
import spatialmath as sm
import spatialgeometry as sg
from scipy.spatial.transform import Rotation as R
from pydrake.solvers import MathematicalProgram, Solve
import time
import math


MODES = ["simulation"]
LITE6 = rtb.models.URDF.Lite6()
LITE6.q = LITE6.qz


if "real" in MODES:
    from xarm.wrapper import XArmAPI
    from robot import RobotMain
    arm = XArmAPI('192.168.1.159', baud_checkset=False)







