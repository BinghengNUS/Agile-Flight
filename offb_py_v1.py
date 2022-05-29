#!/usr/bin/env python3
import rospy
from enum import Enum
from std_msgs.msg import Int64, Header, Byte
from std_srvs.srv import SetBool
import math
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import Altitude, ExtendedState, HomePosition, State, \
                            WaypointList, PositionTarget, RCOut
from mavros_msgs.srv import CommandBool, ParamGet, SetMode, WaypointClear, \
                            WaypointPush
from pymavlink import mavutil
from sensor_msgs.msg import NavSatFix, Imu, BatteryState
from six.moves import xrange
from threading import Thread
"""----Packages needed for Bingheng's code---"""
import UavEnv
import Robust_Flight
from casadi import *
import time
import numpy as np
import matplotlib.pyplot as plt
import uavNN
import torch
from numpy import linalg as LA
import math
# from filterpy.kalman import MerweScaledSigmaPoints
# from filterpy.kalman import UnscentedKalmanFilter as UKF

class uavTaskType(Enum):
    Idle = 0
    TakeOff = 1
    Mission = 2
    Land = 3

class TaskManager:
    def __init__(self):
        self.altitude = Altitude()
        self.extened_state = ExtendedState()
        self.global_position = NavSatFix()
        self.battery_state = BatteryState()
        self.imu_data = Imu()
        self.home_position = HomePosition()
        self.local_position = PoseStamped()
        self.state = State()
        self.local_velocity = TwistStamped()  # local_velocity initialize
        self.local_velocity_body = TwistStamped()

        self.pos = PoseStamped()
        self.position = PositionTarget() # thrust control commands
        self.pwm_out = RCOut()

        self.task_state = uavTaskType.Idle

        # ROS publisher
        self.pos_control_pub = rospy.Publisher('mavros/setpoint_raw/local', PositionTarget, queue_size = 10)
        self.position_pub = rospy.Publisher('mavros/setpoint_position/local', PoseStamped, queue_size = 1)

        # ROS subscribers

        self.local_pos_sub = rospy.Subscriber('mavros/local_position/pose', PoseStamped, self.local_position_callback)
        self.local_vel_sub = rospy.Subscriber('mavros/local_position/velocity_local', TwistStamped, self.local_velocity_callback) #vehicle velocity in ENU world frame
        self.local_vel_sub = rospy.Subscriber('mavros/local_position/velocity_body', TwistStamped, self.local_velocity_body_callback) # velocity in body frame
        self.state_sub = rospy.Subscriber('mavros/state', State, self.state_callback)
        self.cmd_sub = rospy.Subscriber('user/cmd', Byte, self.cmd_callback)
        self.pwm_sub = rospy.Subscriber('mavros/rc/out', RCOut, self.pwm_callback)
        self.battery_sub = rospy.Subscriber('mavros/battery', BatteryState, self.battery_callback)

        # send setpoints in seperate thread to better prevent failsafe
        self.pos_thread = Thread(target=self.send_pos_ctrl, args=())
        self.pos_thread.daemon = True
        self.pos_thread.start()

        # ROS services
        service_timeout = 30
        rospy.loginfo("Waiting for ROS services")
        try:
            rospy.wait_for_service('mavros/param/get',service_timeout)
            rospy.wait_for_service('mavros/cmd/arming',service_timeout)
            rospy.wait_for_service('mavros/mission/push',service_timeout)
            rospy.wait_for_service('mavros/mission/clear',service_timeout)
            rospy.wait_for_service('mavros/set_mode',service_timeout)
            rospy.loginfo("ROS services are up")
        except rospy.ROSException:
            rospy.logerr("failed to connect to services")
        self.get_param_srv = rospy.ServiceProxy('mavros/param/get', ParamGet)
        self.set_arming_srv = rospy.ServiceProxy('mavros/cmd/arming', CommandBool)
        self.set_mode_srv = rospy.ServiceProxy('mavros/set_mode', SetMode)

    def send_pos(self):
        rate = rospy.Rate(10)
        self.pos.header = Header()
        self.pos.header.frame_id = "base_footprint"

        while not rospy.is_shutdown():
            self.pos.header.stamp = rospy.Time.now()
            self.position_pub.publish(self.pos)
            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def send_pos_ctrl(self):
        rate = rospy.Rate(100)
        self.pos.header = Header()
        self.pos.header.frame_id = "base_footprint"

        while not rospy.is_shutdown():
            self.pos.header.stamp = rospy.Time.now()
            self.pos_control_pub.publish(self.position)
            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def pwm_callback(self, data):
        self.pwm_out = data

    def battery_callback(self, data):
        self.battery_state = data
        #rospy.loginfo("voltage is {0}".format(self.battery_state.voltage))


    def cmd_callback(self, data):
        #self.task_state = data
        cmd = data.data
        rospy.loginfo("Command received: {0}".format(self.task_state))
        rospy.loginfo("Command received: {0}".format(data))
        if cmd == 1:
            rospy.loginfo("Taks state changed to {0}".format(self.task_state))
            self.task_state = uavTaskType.TakeOff
        elif cmd == 2:
            rospy.loginfo("Taks state changed to {0}".format(self.task_state))
            self.task_state = uavTaskType.Mission
        elif cmd == 3:
            rospy.loginfo("Taks state changed to {0}".format(self.task_state))
            self.task_state = uavTaskType.Land

    def local_velocity_callback(self, data): # local_velocity callback
        self.local_velocity = data

    def local_velocity_body_callback(self, data): # local_velocity callback
        self.local_velocity_body = data

    def local_position_callback(self, data):
        self.local_position = data

    def state_callback(self, data):
        if self.state.armed != data.armed:
            rospy.loginfo("armed state changed from {0} to {1}".format(self.state.armed, data.armed))

        if self.state.connected != data.connected:
            rospy.loginfo("connected changed from {0} to {1}".format(self.state.connected, data.connected))

        if self.state.mode != data.mode:
            rospy.loginfo("mode changed from {0} to {1}".format(self.state.mode, data.mode))

        if self.state.system_status != data.system_status:
            rospy.loginfo("system_status changed from {0} to {1}".format(
                mavutil.mavlink.enums['MAV_STATE'][
                    self.state.system_status].name, mavutil.mavlink.enums[
                        'MAV_STATE'][data.system_status].name))

        self.state = data

    #
    # Helper methods
    #

    def set_arm(self, arm, timeout):
        """arm: True to arm or False to disarm, timeout(int): seconds"""
        rospy.loginfo("setting FCU arm: {0}".format(arm))
        old_arm = self.state.armed
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        arm_set = False
        for i in xrange(timeout * loop_freq):
            if self.state.armed == arm:
                arm_set = True
                rospy.loginfo("set arm success | seconds: {0} of {1}".format(
                    i / loop_freq, timeout))
                break
            else:
                try:
                    res = self.set_arming_srv(arm)
                    if not res.success:
                        rospy.logerr("failed to send arm command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                rospy.logerr("fail to arm")

    def set_mode(self, mode, timeout):
        """mode: PX4 mode string, timeout(int): seconds"""
        rospy.loginfo("setting FCU mode: {0}".format(mode))
        old_mode = self.state.mode
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        mode_set = False
        for i in xrange(timeout * loop_freq):
            if self.state.mode == mode:
                mode_set = True
                rospy.loginfo("set mode success | seconds: {0} of {1}".format(
                    i / loop_freq, timeout))
                break
            else:
                try:
                    res = self.set_mode_srv(0, mode)  # 0 is custom mode
                    if not res.mode_sent:
                        rospy.logerr("failed to send mode command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                rospy.logerr("fail to set mode")

"""------------Load system parameters for the proposed controller-------------"""
Sys_para = np.array([1.8, 0.011, 0.015, 0.021, 0.21, 8.55e-6, 4.7e-6]) #8.54858e-06  1.3e-5
IGE_para = np.array([0.44, 1.8, 0.1])
# Time interval of taking-off
t_end = 5
# Sampling time-step
dt_sample = 1/30
uav = UavEnv.quadrotor(Sys_para, IGE_para, dt_sample)
uav.Model()

"""------------Define parameterization model----------------------------------"""
# Define neural network for process noise
D_in, D_h, D_out = 6, 50, 20
#model_QR = uavNN.Net(D_in, D_h, D_out)

def para(para_bar):
    P_min = 0
    P_max = 2e-1
    gammar_min = 0
    gammar_max = 1
    gammaq_min = 0
    gammaq_max = 1
    R_min = 0
    R_max = 1e2
    Q_min = 0
    Q_max = 1e2
    tunable = np.zeros((1, 20))
    for i in range(20):
        tunable[0, i] = para_bar[i, 0]  # convert tensor to array
    tunable_para = np.zeros((1, 20))
    for i in range(9):
        tunable_para[0, i] = P_min + (P_max - P_min) * tunable[0, i]
    for i in range(6):
        tunable_para[0, i + 10] = R_min + (R_max - R_min) * tunable[0, i + 10]
    for i in range(3):
        tunable_para[0, i + 17] = Q_min + (Q_max - Q_min) * tunable[0, i + 17]
    tunable_para[0, 9] = gammar_min + (gammar_max - gammar_min) * tunable[0, 10]
    tunable_para[0, 16] = gammaq_min + (gammaq_max - gammaq_min) * tunable[0, 16]
    return tunable_para

# Parameterize control gain
D_in, D_h, D_out = 6, 50, 6
#model_gain = uavNN.NetCtrl(D_in, D_h, D_out)
gp_min = 1e-1  # lower bound of position control gain
gp_max = 5  # upper bound of position control gain
gv_min = 1e-1  # lower bound of velocity control gain, previous value was 1e-3
gv_max = 5   # upper bound of velocity control gain
def control_gain(nn_gain):
    ctrl_gain = np.zeros((1, 6))
    for i in range(6):
        if i<=2:
            ctrl_gain[0, i] = gp_min + (gp_max-gp_min)*nn_gain[i, 0]
        else:
            ctrl_gain[0, i] = gv_min + (gv_max-gv_min)*nn_gain[i, 0]
    return ctrl_gain


"""------------Define reference trajectory------------------------------------"""
# Target point for take-off
x_t, y_t, z_t = 0, 0, 1.5
dx_t, dy_t, dz_t = 0, 0, 0
ddx_t, ddy_t, ddz_t = 0, 0, 0
target = np.hstack((x_t, dx_t, ddx_t, y_t, dy_t, ddy_t, z_t, dz_t, ddz_t))

def polynomial_ref(t_end, target, initial_state):
    x_0, dx_0, ddx_0 = initial_state[0], 0, 0
    y_0, dy_0, ddy_0 = initial_state[1], 0, 0
    z_0, dz_0, ddz_0 = initial_state[2], 0, 0
    x_t, dx_t, ddx_t = target[0], target[1], target[2]
    y_t, dy_t, ddy_t = target[3], target[4], target[5]
    z_t, dz_t, ddz_t = target[6], target[7], target[8]
    x_con = np.vstack((x_0, dx_0, ddx_0, x_t, dx_t, ddx_t))
    y_con = np.vstack((y_0, dy_0, ddy_0, y_t, dy_t, ddy_t))
    z_con = np.vstack((z_0, dz_0, ddz_0, z_t, dz_t, ddz_t))
    A_p   = np.array([[0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 2, 0, 0],
                      [t_end**5, t_end**4, t_end**3, t_end**2, t_end, 1],
                      [5*t_end**4, 4*t_end**3, 3*t_end**2, 2*t_end, 1, 0],
                      [20*t_end**3, 12*t_end**2, 6*t_end, 2, 0, 0]])
    Coeff_x = np.matmul(LA.inv(A_p), x_con)
    Coeff_y = np.matmul(LA.inv(A_p), y_con)
    Coeff_z = np.matmul(LA.inv(A_p), z_con)
    return Coeff_x, Coeff_y, Coeff_z

def Reference(Coeff_x, Coeff_y, Coeff_z, time, t_end, target):
    # Position reference as a polynomial of time
    p_t  = np.array([[time**5, time**4, time**3, time**2, time, 1]])
    v_t  = np.array([[5*time**4, 4*time**3, 3*time**2, 2*time, 1, 0]])
    a_t  = np.array([[20*time**3, 12*time**2, 6*time, 2, 0, 0]])
    if time <= t_end:
        ref_p = np.array([[np.matmul(p_t, Coeff_x)[0, 0], np.matmul(p_t, Coeff_y)[0, 0], np.matmul(p_t, Coeff_z)[0, 0]]]).T
        ref_v = np.array([[np.matmul(v_t, Coeff_x)[0, 0], np.matmul(v_t, Coeff_y)[0, 0], np.matmul(v_t, Coeff_z)[0, 0]]]).T
        ref_a = np.array([[np.matmul(a_t, Coeff_x)[0, 0], np.matmul(a_t, Coeff_y)[0, 0], np.matmul(a_t, Coeff_z)[0, 0]]]).T
    else:
        ref_p = np.array([[target[0], target[3], target[6]]]).T
        ref_v = np.array([[target[1], target[4], target[7]]]).T
        ref_a = np.array([[target[2], target[5], target[8]]]).T

    b1_d = np.array([[1, 0, 0]]).T
    reference = {"ref_p": ref_p,
                 "ref_v": ref_v,
                 "ref_a": ref_a,
                 "b1_d": b1_d}
    return reference

"""------------Quaternion to Rotation Matrix---------------------------------"""
def Quaternion2Rotation(quaternion):
    # convert a point from body frame to inertial frame
    q0, q1, q2, q3 = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    R_b = np.array([[r00, r01, r02],
                    [r10, r11, r12],
                    [r20, r21, r22]])
    return R_b

"""-----------------Actual total thrust-------------------------------------"""
def Actual_thrust(pwm, voltage, R_b, Sys_para):
    # rpm1 = (-0.0057*pwm[0]**2+28.271*pwm[0]-22359)*2*math.pi/60
    # #print("rpm1 is {0}".format(rpm1))
    # rpm2 = (-0.0057*pwm[1]**2+28.271*pwm[1]-22359)*2*math.pi/60
    # rpm3 = (-0.0057*pwm[2]**2+28.271*pwm[2]-22359)*2*math.pi/60
    # rpm4 = (-0.0057*pwm[3]**2+28.271*pwm[3]-22359)*2*math.pi/60
    # th1  = Sys_para[5]*rpm1**2
    # #print("th1 is {0}".format(th1))

    # th2  = Sys_para[5]*rpm2**2
    # th3  = Sys_para[5]*rpm3**2
    # th4  = Sys_para[5]*rpm4**2

    # th1 = 1e-5*pwm[0]**2-0.0147*pwm[0]+4.7981
    # th2 = 1e-5*pwm[1]**2-0.0147*pwm[1]+4.7981
    # th3 = 1e-5*pwm[2]**2-0.0147*pwm[2]+4.7981
    # th4 = 1e-5*pwm[3]**2-0.0147*pwm[3]+4.7981

    # compute rpm with pwm and voltage old version
    # rpm1 = -2.606e4 + 23.81*pwm[0] + 297.6*voltage - 0.004972*pwm[0]**2 + 0.1419*pwm[0]*voltage
    # rpm2 = -2.606e4 + 23.81*pwm[1] + 297.6*voltage - 0.004972*pwm[1]**2 + 0.1419*pwm[1]*voltage
    # rpm3 = -2.606e4 + 23.81*pwm[2] + 297.6*voltage - 0.004972*pwm[2]**2 + 0.1419*pwm[2]*voltage
    # rpm4 = -2.606e4 + 23.81*pwm[3] + 297.6*voltage - 0.004972*pwm[3]**2 + 0.1419*pwm[3]*voltage
    # compute rpm with pwm and voltage new version
    rpm1 = -1.261e4 + 14.94*pwm[0] - 311.2 * voltage - 0.003654 * pwm[0]**2 + 0.4248*pwm[0]*voltage
    rpm2 = -1.261e4 + 14.94*pwm[1] - 311.2 * voltage - 0.003654 * pwm[1]**2 + 0.4248*pwm[1]*voltage
    rpm3 = -1.261e4 + 14.94*pwm[2] - 311.2 * voltage - 0.003654 * pwm[2]**2 + 0.4248*pwm[2]*voltage
    rpm4 = -1.261e4 + 14.94*pwm[3] - 311.2 * voltage - 0.003654 * pwm[3]**2 + 0.4248*pwm[3]*voltage

    #rospy.loginfo("rpm is {0}".format(rpm1))
# compute thrust with rpm
    th1  = 9.14e-8*rpm1**2 - 4.872e-5*rpm1 + 0.1091
    th2  = 9.14e-8*rpm2**2 - 4.872e-5*rpm2 + 0.1091
    th3  = 9.14e-8*rpm3**2 - 4.872e-5*rpm3 + 0.1091
    th4  = 9.14e-8*rpm4**2 - 4.872e-5*rpm4 + 0.1091
    #rospy.loginfo("th1 is {0}".format(th1))
    total_th = th1+th2+th3+th4
    z = np.array([[0, 0, 1]]).T
    f_ad = np.matmul(R_b, total_th*z)
    return f_ad

"""-----------------Define controller---------------------------------------"""
# Controller
GeoCtrl = Robust_Flight.Controller(Sys_para, uav.X, uav.Xmhe)

"""-------------------Define MHE----------------------------------------------"""
horizon = 10 # previous 30
uavMHE = Robust_Flight.MHE(horizon, dt_sample)
uavMHE.SetStateVariable(uav.Xmhe_p)
uavMHE.SetOutputVariable(uav.output_p)
uavMHE.SetControlVariable(uav.f_d)
uavMHE.SetRotationVariable(uav.R_B, uav.quater)
uavMHE.SetNoiseVariable(uav.wf)
uavMHE.SetModelDyn(uav.pdyn_mhe, uav.pdyn_ukf)
uavMHE.SetCostDyn()

"""------------------Initialization-------------------------------------------"""
# Initial weighting matrix of the arrival cost
# P0 = 100 * np.identity(uav.Xmhe_p.numel())
# Control force and torque list
m = Sys_para[0]
g = 9.8
ctrl_f = []
ctrl_f += [np.array([[0,0,m*g]]).T]
# Rotation list
R_B = []
# Initial estimated force
df_I0 = np.array([[0, 0, 0]]).T

time = 0
k_time = 0
# initialization flag
flagi = 0
# Tracking error list
track_e = []
track_e += [np.zeros((6, 1))]
# Measurement list
Y_p = []


# Load the neural network model
PATH1 = 'Trained_model_QR.pt'
model_QR = torch.load(PATH1)
PATH2 = 'Trained_model_gain.pt'
model_gain = torch.load(PATH2)


"""---------------------------------Define model for UKF---------------------------"""
# def DynUKF(x, dt, U):
#     u = np.array([[U[0], U[1], U[2]]]).T
#     quater = np.array([U[3], U[4], U[5],
#                        U[6]])
#     x_next = uavMHE.MDyn_ukf_fn(s=x, c=u, qt=quater)['MDyn_ukff']
#     x_next = np.reshape(x_next, (9))
#     return x_next

# def OutputUKF(x):
#     H = uavMHE.H_fn(s=x)['Hf']
#     output = np.matmul(H, x)
#     y = np.reshape(output, (6))
#     return y

# UKF settings
# sigmas = MerweScaledSigmaPoints(9, alpha=.1, beta=2., kappa=1)
# ukf    = UKF(dim_x=9, dim_z=6, fx=DynUKF, hx=OutputUKF, dt=dt_sample, points=sigmas)

# Covariance matrices for UKF
# ukf.R = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]) # measurement noise
# ukf.Q = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01,10,10, 10]) # process noise

# Load the trajectory data
filename = 'waypoints'
traj_waypoints = np.loadtxt(filename)

# Choose the trajectory
takeoff_only = False
nn_param_list = []
# Choose estimator
mhe = True

if __name__ == '__main__':
    rospy.init_node('Moving Horizon Estimator')
    uavTask = TaskManager()

    uavTask.pos.pose.position.x = 0
    uavTask.pos.pose.position.y = 0
    uavTask.pos.pose.position.z = 0

    uavTask.set_mode("OFFBOARD", 5)
    uavTask.set_arm(True, 5)

    while not rospy.is_shutdown():
        rate = rospy.Rate(100) # the operating frequency at which the code runs
        print(uavTask.task_state)
        #uavTask.position_pub.publish(uavTask.pos)
        if uavTask.task_state == uavTaskType.TakeOff:
            #rospy.loginfo("Doing Takeoff")
            pwm = np.zeros(4)
            pwm[0] = uavTask.pwm_out.channels[0]
            pwm[1] = uavTask.pwm_out.channels[1]
            pwm[2] = uavTask.pwm_out.channels[2]
            pwm[3] = uavTask.pwm_out.channels[3]
            # rospy.loginfo("pwm 1 is {0}".format(pwm[0]))
            # rospy.loginfo("pwm 2 is {0}".format(pwm[1]))
            # rospy.loginfo("pwm 3 is {0}".format(pwm[2]))
            # rospy.loginfo("pwm 4 is {0}".format(pwm[3]))R_b
            uavTask.position.position.x = 0
            uavTask.position.position.y = 0
            uavTask.position.position.z = 1.5
            uavTask.position.type_mask = 3064 # flag for pid
            uavTask.pos_control_pub.publish(uavTask.position)

        elif uavTask.task_state == uavTaskType.Mission:
            #rospy.loginfo("Doing Mission")
            ## Controller will be used here ###
            # Receive feedback from ROS topic

            # Quaternion (body frame:NWU, intertia frame: ENU)
            q0 = uavTask.local_position.pose.orientation.x
            q1 = uavTask.local_position.pose.orientation.y
            q2 = uavTask.local_position.pose.orientation.z
            q3 = uavTask.local_position.pose.orientation.w
            quaternion = np.array([q3, q0, q1, q2])
            R_b = Quaternion2Rotation(quaternion) #transformation matrix from NWU body to ENU world
            x = uavTask.local_position.pose.position.x
            y = uavTask.local_position.pose.position.y
            z = uavTask.local_position.pose.position.z # ENU used in ROS
            vx_enu = uavTask.local_velocity.twist.linear.x # NWU body frame
            vy_enu = uavTask.local_velocity.twist.linear.y
            vz_enu = uavTask.local_velocity.twist.linear.z
            wx = uavTask.local_velocity.twist.angular.x
            # inilialization of x_hatmh
            if flagi ==0:
                r_I0 = np.array([[x, y, z]]).T
                v_I0 = np.array([[vx_enu, vy_enu, vz_enu]]).T
                x_hatmh = np.vstack((r_I0, v_I0, df_I0))
                xmhe_traj = x_hatmh
                P_xy0 = np.array([x, y, z])
                flagi = 1

            # ###########
            # vx_body = uavTask.local_velocity_body.twist.linear.x
            # vy_body = uavTask.local_velocity_body.twist.linear.y
            # vz_body = uavTask.local_velocity_body.twist.linear.z
            # v_body = np.array([[vx_body, vy_body, vz_body]]).T
            # v_enu = np.matmul(R_b, v_body)
            # ##########
            # v_enu_actual = np.array([[vx_enu, vy_enu, vz_enu]]).T

            # rospy.loginfo("converted enu velocity is {0}".format(v_enu - v_enu_actual))

            y_p = np.array([[x, y, z, vx_enu, vy_enu, vz_enu]]).T
            Y_p += [y_p]
            # pwm
            pwm = np.zeros(4)
            pwm[0] = uavTask.pwm_out.channels[0]
            pwm[1] = uavTask.pwm_out.channels[1]
            pwm[2] = uavTask.pwm_out.channels[2]
            pwm[3] = uavTask.pwm_out.channels[3]
            voltage = uavTask.battery_state.voltage

            # reference

            if takeoff_only:
                Coeff_x, Coeff_y, Coeff_z = polynomial_ref(t_end, target, P_xy0)
                ref = Reference(Coeff_x, Coeff_y, Coeff_z, time, t_end, target)
                ref_p, ref_v, ref_dv, b1_d = ref['ref_p'], ref['ref_v'], ref['ref_a'], ref['b1_d']
            else:
                if k_time<=(np.size(traj_waypoints, 0)-1):
                    ref_p = np.array([[traj_waypoints[k_time, 1], traj_waypoints[k_time, 0], traj_waypoints[k_time, 2]]]).T
                    ref_v = np.array([[traj_waypoints[k_time, 4], traj_waypoints[k_time, 3], traj_waypoints[k_time, 5]]]).T
                    ref_dv = np.array([[traj_waypoints[k_time, 7], traj_waypoints[k_time, 6], traj_waypoints[k_time, 8]]]).T
                    b1_d = np.array([[1, 0, 0]]).T
                else:
                    ref_p = np.array([[traj_waypoints[-1, 1], traj_waypoints[-1, 0], traj_waypoints[-1, 2]]]).T
                    ref_v = np.array([[traj_waypoints[-1, 4], traj_waypoints[-1, 3], traj_waypoints[-1, 5]]]).T
                    ref_dv = np.array([[traj_waypoints[-1, 7], traj_waypoints[-1, 6], traj_waypoints[-1, 8]]]).T
                    b1_d = np.array([[1, 0, 0]]).T

            ref_pva = np.vstack((ref_p, ref_v))
            if mhe:
                #Solve MHE to obtain the estimated state trajectory within a horizon
                rospy.loginfo("current z feedback is {0}".format(Y_p[-1][2,0]))
                opt_sol = uavMHE.MHEsolver(Y_p, x_hatmh, xmhe_traj, ctrl_f, model_QR, k_time, R_B, ref_pva)
                xmhe_traj = opt_sol['state_traj_opt']
                costate_traj = opt_sol['costate_traj_opt']
                # Establish the auxiliary MHE system
                auxSys = uavMHE.GetAuxSys(xmhe_traj, costate_traj, model_QR, Y_p, ref_pva)
                matA, matB, matH = auxSys['matA'], auxSys['matB'], auxSys['matH']
                matD, matE, matF = auxSys['matD'], auxSys['matE'], auxSys['matF']
                if time > (horizon * dt_sample):
                    # Update x_hatmh based on xmhe_traj
                    for ix in range(len(x_hatmh)):
                        x_hatmh[ix] = xmhe_traj[1, ix]
                else:
                    for ix in range(len(x_hatmh)):
                        x_hatmh[ix] = xmhe_traj[0, ix]
                df_I = np.transpose(xmhe_traj[-1, 6:9])
            else:
                # Implement UKF
                Qt = quaternion
                # Qt = np.reshape(Qt, (1, 4))
                # ctrlf = np.reshape(ctrl_f[-1], (1, 3))
                # U1 = np.hstack((ctrlf, Qt))
                # U1 = np.reshape(U1, (7))
                # ukf.predict(U=U1)
                # y = np.reshape(Y_p[-1], (6))
                # ukf.update(z=y)
                # Xukf = ukf.x.copy()
                # df_I = Xukf[6:9]

            # Position control
            scale_factor = voltage/17.1*2
            #df_Iukf = np.reshape(df_Iukf,(3,1))
            #df_Iukf = df_Iukf - np.array([[0, 0, scale_factor]]).T
            #rospy.loginfo("estimator force z is {0}".format(df_Imh))
            #rospy.loginfo("estimator position z is {0}".format(xmhe_traj[-1, 2]))
            df_I = np.reshape(df_I, (3, 1))
            df_I = df_I - np.array([[0, 0, scale_factor+1]]).T
            #rospy.loginfo("estimator force is {0}".format(df_Iukf))

            # ref_p = np.array([[0,0,-1]]).T
            # ref_v = np.zeros((3,1))
            # ref_dv = np.zeros((3,1))
            input_gain = np.vstack((Y_p[-1]))
            nn_ctrl = model_gain(input_gain)
            nn_gain = model_QR(input_gain)
            nn_param= para(nn_gain)
            nn_param_list += [nn_param]
           # np.save('nn_param_list_8s',nn_param_list)
            ctrl_gain = control_gain(nn_ctrl)
            # ctrl_gain = np.array([[4,4,4,2,2,2]]) #manual gain
            # receive feedback of position and attitude from ROS topic

            #R_B is the transformation matrix from NED body to NED world frame
            R_B += [R_b]
            if time<1:
                factor = 0
                rospy.loginfo("using factor 0")
            # elif time<4:
            #     factor = 1
            #     rospy.loginfo("using factor 1")
            else:
                factor = 1
                rospy.loginfo("using factor 0")
            R_Bd_next, Tf, e_x, e_v, fd = GeoCtrl.position_ctrl(ctrl_gain, Y_p[-1], R_B[-1], ref_p, ref_v, ref_dv, b1_d, factor*df_I)
            # Store tracking error and control forces
            error_track = np.vstack((e_x, e_v))
            track_e += [error_track]
            np.save('track_error_list_noDO_8s',track_e)
            f_ad = Actual_thrust(pwm, voltage, R_b, Sys_para)
            ctrl_f += [f_ad]
            #rospy.loginfo("z force is {0}".format(f_ad[2, 0]))
            # Send desired control force vector in inertial frame to ROS topic
            m = Sys_para[0]
            # force in inertial NED
            fx = fd[0, 0]/m/(g*scale_factor)
            fy = fd[1, 0]/m/(g*scale_factor)
            fz = fd[2, 0]/m/(g*scale_factor)

            #thr_sp = -(np.matmul(np.diag([0.3, 0.3, 0.19]), e_x) + np.matmul(np.diag([0.5, 0.5, 0.19]), e_v) + np.array([[0, 0, 0.5]]).T)
            f_diff = fd[2,0]-f_ad[2,0]
            uavTask.position.position.x = fx
            uavTask.position.position.y = fy
            uavTask.position.position.z = fz
            uavTask.position.velocity.x = e_x[-1,0]
            uavTask.position.velocity.y = df_I[2,0]
            uavTask.position.velocity.z = e_x[1,0]
            uavTask.position.acceleration_or_force.x = ref_p[1, 0]
            uavTask.position.acceleration_or_force.y = ref_p[0, 0]
            uavTask.position.acceleration_or_force.z = ref_p[-1,0]
            #rospy.loginfo("estimator z is {0}".format(df_Imh[2,0]))
            #rospy.loginfo("f diff is {0}".format(f_diff))
            #rospy.loginfo("feedback position z is {0}".format(z))
            #rospy.loginfo("feedback velocity vz is {0}".format(vz))


            uavTask.position.yaw = 0

            time += dt_sample
            k_time += 1
            uavTask.position.type_mask = 32768 # flag for proposed control mode
            uavTask.pos_control_pub.publish(uavTask.position)

        elif uavTask.task_state == uavTaskType.Land:
            rospy.loginfo("Doing Land")
            uavTask.position.position.x = 0
            uavTask.position.position.y = 0
            uavTask.position.position.z = 0

            uavTask.position.yaw = 0

            uavTask.position.type_mask = 3064 # flag for pid
            uavTask.pos_control_pub.publish(uavTask.position)


        rate.sleep()
    rospy.spin()
