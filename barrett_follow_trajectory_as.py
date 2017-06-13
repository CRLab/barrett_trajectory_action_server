#! /usr/bin/python
import sensor_msgs.msg
import rospy
import threading
import numpy as np
import std_srvs.srv
import barrett_trajectory_action_server.srv 
from actionlib import SimpleActionServer

from control_msgs.msg import (FollowJointTrajectoryAction,
                              FollowJointTrajectoryFeedback,
                              FollowJointTrajectoryResult)

from barrett_tactile_msgs.msg import TactileInfo
from bhand_controller.srv import SetControlMode

from collections import namedtuple, defaultdict

DOF_STATE = namedtuple("DOF", ["spread", "f1", "f2", "f3"])

# INDICES FOR DOFS TO BE SENT TO THE HAND TO EXECUTE
SPREAD_DOF_INDEX = 0
F1_DOF_INDEX = 2
F2_DOF_INDEX = 3
F3_DOF_INDEX = 1

# INDICES for DOFS from waypoints provided by MOVEIt!
SPREAD_WAYPOINT_INDEX = 0 
F1_WAYPOINT_INDEX = 1
F2_WAYPOINT_INDEX = 2
F3_WAYPOINT_INDEX = 3

def waypoint_to_np(wp):
    return np.array([wp.positions[SPREAD_WAYPOINT_INDEX],  # spread
                     wp.positions[F3_WAYPOINT_INDEX],  # f3
                     wp.positions[F1_WAYPOINT_INDEX],  # f1
                     wp.positions[F2_WAYPOINT_INDEX],  # f2
                 ])


def dof_state_to_np(dof):
    return np.array([dof.spread, dof.f3, dof.f1, dof.f2])


def dof_state_from_np(dof_np):
    return DOF_STATE(
        spread=dof_np[SPREAD_DOF_INDEX], f1=dof_np[F1_DOF_INDEX], f2=dof_np[F2_DOF_INDEX], f3=dof_np[F3_DOF_INDEX])


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

        self.service_reset_tactile_state = rospy.Service('/barrett/reset_tactile_state', std_srvs.srv.Empty, self.reset_tactile_state)
        self.service_set_ignore_tactile_state = rospy.Service('/barrett/set_ignore_tactile_state',  std_srvs.srv.SetBool, self.set_ignore_tactile_state)
        self.service_get_tactile_info = rospy.Service('/barrett/get_tactile_info',  barrett_trajectory_action_server.srv.GetTactileContacts, self.get_tactile_info)

        self.current_joint_and_dof_state_mutex = threading.Lock()
        self.current_tactile_state_mutex = threading.Lock()

        self.current_joint_state = None
        self.current_dof = None
        self.current_tactile_state = None

        self.current_goal = None
        
        self.feedback = FollowJointTrajectoryFeedback()
        self.feedback.joint_names = ['bh_j11_joint', 'bh_j32_joint', 'bh_j12_joint', 'bh_j22_joint']

        self.ignore_tactile_state = False
        self.activated_dofs = np.ones(4)
        self.tactile_info = set()

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
            return
              
        start_time = rospy.Time.now()
        rate = rospy.Rate(self.control_rate)
        success = True
        
        for i, waypoint in enumerate(goal.trajectory.points):
                        
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
                velocity *= self.activated_dofs

                cmd_msg.velocity = velocity
                tolerance = 0.03 if i == len(goal.trajectory.points) - 1 else 0.15
                rospy.logdebug("Velocity {}, tolerance {}".format(velocity, tolerance))

                if np.allclose(current_position_error, np.zeros_like(current_position_error), atol=tolerance):
                    break

                # check if something external has told us to stop (Ctrl C) or a prempt
                if self.action_server.is_preempt_requested() or rospy.is_shutdown():
                    self.action_server.set_preempted()
                    rospy.logerr("PREEMPTING HAND FOLLOW TRAJECTORY")
                    return
                
                # Everything is going smoothly, send velocity cmd to hand
                self.hand_cmd_pub.publish(cmd_msg)
    
                rate.sleep()

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
        if self.ignore_tactile_state:
            rospy.loginfo("Ignoring tactile info")
            return

        for position in msg.tactile_info:
            dof_name = position.header.frame_id.split("_")[1]

            if dof_name == "link1":
                self.activated_dofs[F1_DOF_INDEX] = 0
            elif dof_name == "link2":
                self.activated_dofs[F2_DOF_INDEX] = 0
            elif dof_name == "link3":
                self.activated_dofs[F3_DOF_INDEX] = 0
                
            self.tactile_info.add(position.header.frame_id)

    def reset_tactile_state(self, req):
        rospy.loginfo("Resetting tactile state")

        self.activated_dofs = np.ones(4)
        self.tactile_info = set()
        return std_srvs.srv.EmptyResponse()

    def set_ignore_tactile_state(self, ignore_flag):
        rospy.loginfo("Setting ignore tactile to {}".format(ignore_flag))
        self.ignore_tactile_state = ignore_flag
        return std_srvs.srv.SetBoolResponse(success=True, message="Flag set correctly")

    def get_tactile_info(self, req):
        return barrett_trajectory_action_server.srv.GetTactileContacts(active_sensors=list(self.tactile_info))


if __name__ == "__main__":
    rospy.init_node("Barrett_Trajectory_Follower", log_level=rospy.DEBUG)
    joint_trajectory_follower = JointTracjectoryActionServer()
    rospy.spin()
