#! /usr/bin/python
import sensor_msgs.msg
import rospy
import threading
import numpy as np

from actionlib import SimpleActionServer

from control_msgs.msg import (FollowJointTrajectoryAction,
                              FollowJointTrajectoryFeedback,
                              FollowJointTrajectoryResult)

from barrett_tactile_msgs.msg import TactileInfo
from bhand_controller.srv import SetControlMode
from bhand_controller.msg import State, TactileArray, Service

from collections import namedtuple

DOF_STATE = namedtuple("DOF", ["spread", "f1", "f2", "f3"])


def waypoint_to_np(wp):
    return np.array([wp.positions[0],  # spread
                     wp.positions[3],  # f3
                     wp.positions[1],  # f1
                     wp.positions[2],  # f2
                 ])


def dof_state_to_np(dof):
    return np.array([dof.spread, dof.f3, dof.f1, dof.f2])


def dof_state_from_np(dof_np):
    return DOF_STATE(
        spread=dof_np[0], f1=dof_np[2], f2=dof_np[3], f3=dof_np[1])


#INIT NODE code

# try:
#     self._service_bhand_actions = rospy.ServiceProxy(self._actions_service_name, Actions)
# except ValueError, e:
#     rospy.logerr('BHandGUI: Error connecting service (%s)'%e)

# def send_bhand_action(self, action):    
#     '''
#         Calls the service to set the control mode of the hand
#         @param action: Action number (defined in msgs/Service.msg)
#         @type action: int
#     '''         
#     try:
#         ret = self._service_bhand_actions(action)               
#     except ValueError, e:
#         rospy.logerr('BHandGUI::send_bhand_action: (%s)'%e)
#     except rospy.ServiceException, e:
#         rospy.logerr('BHandGUI::send_bhand_action: (%s)'%e)
#         QMessageBox.warning(self._widget, "Warning", "Service is not available: send_bhand_action")

# self.send_bhand_action(Service.INIT_HAND)

#Get Tactile Info Code
# self._tact_topic = '/%s/tact_array'%self.bhand_node_name 
# try:
#     self._tact_subscriber = rospy.Subscriber(self._tact_topic, TactileArray, self._receive_tact_data)
# except ValueError, e:
#     rospy.logerr('BHandGUI: Error connecting topic (%s)'%e)


class JointTracjectoryActionServer(object):
    def __init__(self):
        self.action_server = SimpleActionServer(
            "/barrett/follow_joint_trajectory",
            FollowJointTrajectoryAction,
            execute_cb=self._follow_joint_trajectory_cb,
            auto_start=False)

        self._tactile_sub = rospy.Subscriber("/bhand_node/tactile_info",
                                             TactileInfo,
                                             self.update_tactile_state_cb)
        self._joint_sub = rospy.Subscriber("/joint_states",
                                           sensor_msgs.msg.JointState,
                                           self.update_joint_state_cb)

        self.hand_cmd_pub = rospy.Publisher("/bhand_node/command", sensor_msgs.msg.JointState, queue_size=10)
        self.service_set_mode = rospy.ServiceProxy('/bhand_node/set_control_mode', SetControlMode)
        self.current_joint_and_dof_state_mutex = threading.Lock()
        self.current_tactile_state_mutex = threading.Lock()

        self.current_joint_state = None
        self.current_dof = None
        self.current_tactile_state = None

        self.current_goal = None
        
        self.feedback = FollowJointTrajectoryFeedback()
        self.feedback.joint_names = ['bh_j11_joint', 'bh_j32_joint', 'bh_j12_joint', 'bh_j22_joint']

        self.result = FollowJointTrajectoryResult()
    
        self.EXECUTION_WAYPOINT_THRESHOLD = 0.5
        self.START_POINT_THRESHOLD = 0.1
        self.control_rate = 10
        self.action_server.start()

    def _follow_joint_trajectory_cb(self, goal):
        self.current_goal = goal

        self.service_set_mode('VELOCITY')

        start_waypoint = goal.trajectory.points[0]
        end_waypoint = goal.trajectory.points[-1]

        start_waypoint_np = waypoint_to_np(start_waypoint)
        end_waypoint_np = waypoint_to_np(end_waypoint)

        current_dof_np = dof_state_to_np(self.current_dof)

        # first check that we are close to the starting point
        if not np.allclose(start_waypoint_np, current_dof_np, atol=self.START_POINT_THRESHOLD):
            rospy.logerr(
                "CANNOT EXECUTE TRAJECTORY: our current dof values: %s, are far from the trajectory starting dof values: "
                % (current_dof_np, start_waypoint_np))

        else:
            rospy.logdebug("Beginning Trajectory Execution")
              
        start_time = rospy.Time.now()
        rate = rospy.Rate(self.control_rate)
        success = True
        
        rospy.logdebug("Trajectory has: {} ".format(len(goal.trajectory.points)))
        for i, waypoint in enumerate(goal.trajectory.points):
            
            rospy.logdebug("Working on waypoint: " + str(waypoint))
            
            # first make sure we have not deviated to far from trajectory
            # abort if so
            current_dof_np = dof_state_to_np(self.current_dof)
            waypoint_np = waypoint_to_np(waypoint)
            
            gain = np.array([0.6, 0.6, 0.6, 0.6])
            current_position_error = current_dof_np - waypoint_np
            if not np.allclose(waypoint_np, current_dof_np,
                               atol=self.EXECUTION_WAYPOINT_THRESHOLD):
                rospy.logerr(
                    "STOPPING EXECUTE TRAJECTORY: our current dof values: %s, are far from the expected current dof values: %s"
                    % (str(current_dof_np), str(waypoint_np)))
                self.action_server.set_aborted()
                return 

            velocity = np.clip(-gain * current_position_error, -0.1, 0.1) 
            # second still within the time allocated to this waypoint
            # continue to send waypoint velocity command
            cmd_msg = sensor_msgs.msg.JointState()
            cmd_msg.name = ['bh_j11_joint', 'bh_j32_joint', 'bh_j12_joint', 'bh_j22_joint']
            cmd_msg.position = [0, 0, 0, 0]  # [self.base_spread, self.finger3_spread, self.finger1_spread, self.finger2_spread]                                                                            
            cmd_msg.velocity = velocity
            cmd_msg.effort = [0,0,0,0]
            rospy.logdebug("waypoint velocity: "  + str(waypoint.velocities))
            rospy.logdebug("cmd_msg velocity: " + str(cmd_msg.velocity))
            self.feedback.desired = waypoint
            self.feedback.actual.positions = self.current_joint_state
            self.feedback.actual.velocities = self.current_joint_state
            self.feedback.actual.accelerations = self.current_joint_state
            self.feedback.actual.time_from_start = waypoint.time_from_start

            while waypoint.time_from_start >= rospy.Time.now() - start_time:
            
                current_dof_np = dof_state_to_np(self.current_dof)
                waypoint_np = waypoint_to_np(waypoint)
                current_position_error = current_dof_np - waypoint_np
                velocity = np.clip(-gain * current_position_error, -0.1, 0.1)
                # for index in range(len(velocity)):
                #     if np.abs(velocity[index]) < 0.02 and waypoint.positions[index] != 0:
                #         velocity[index] = np.sign(velocity[index]) * 0.02


                cmd_msg.velocity = velocity
                rospy.logdebug("Position Error: " + str(current_position_error))
                rospy.logdebug("Velocity: " + str(velocity))
                tolerance = 0.03 if i == len(goal.trajectory.points) - 1 else 0.15
                rospy.logdebug("Tolerance {}, I: {}".format(tolerance, i))
                if np.allclose(current_position_error, np.zeros_like(current_position_error), atol=tolerance):
                    rospy.logdebug("Jumping to next trajectory point, since we have reached current waypoint")
                    break
                # check if something external has told us to stop (Ctrl C) or a prempt
                if self.action_server.is_preempt_requested() or rospy.is_shutdown():
                    self.action_server.set_preempted()
                    rospy.logerr("PREEMPTING HAND FOLLOW TRAJECTORY")
                    return
                
                # Everything is going smoothly, send velocity cmd to hand
                self.hand_cmd_pub.publish(cmd_msg)
                #rospy.logdebug("Published Command: ")
                #rospy.logdebug(cmd_msg)
    
                rate.sleep()

        '''
        # Now do proportional control to get hand to the final desired state
        self.service_set_mode('POSITION')

        import IPython
        IPython.embed()
        cmd_msg = sensor_msgs.msg.JointState()
        cmd_msg.name = ['bh_j11_joint', 'bh_j32_joint', 'bh_j12_joint', 'bh_j22_joint']
        cmd_msg.position = end_waypoint.position  # [self.base_spread, self.finger3_spread, self.finger1_spread, self.finger2_spread]                                                                       
        cmd_msg.velocity = [0, 0, 0, 0]
        cmd_msg.effort = [0, 0, 0, 0]

        rospy.logdebug("Published Command: ")
        rospy.logdebug(cmd_msg)

        
        #self.hand_cmd_pub.publish(cmd_msg)
        '''
        if success:
            rospy.loginfo('Barret Hand Follow Joint Trajectory Succeeded')
            self.result.error_code = 0 # 0 means Success 
            self.action_server.set_succeeded(self.result)

    def update_joint_state_cb(self, msg):
        # hand message not arm message
        if msg.name[0] == 'bh_j23_joint':
            self.current_joint_and_dof_state_mutex.acquire()
            self.current_joint_state = msg
            self.current_dof = DOF_STATE(
                spread=msg.position[6],
                f1=msg.position[1],
                f2=msg.position[2],
                f3=msg.position[3])
            self.current_joint_and_dof_state_mutex.release()

    def update_tactile_state_cb(self, msg):
        pass


if __name__ == "__main__":
    rospy.init_node("Barrett_Trajectory_Follower", log_level=rospy.DEBUG)
    joint_trajectory_follower =   JointTracjectoryActionServer()
    rospy.spin()
