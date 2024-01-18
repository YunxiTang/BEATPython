import rospy
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import tf
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class Visualizer:
    def __init__(self, world_map, path_solution) -> None:
        self._world_map = world_map
        self._path_solution = path_solution

        try:
            rospy.init_node('result_visualizer', anonymous=False)
        except rospy.exceptions.ROSException as e:
            pass

        self.map_pub = rospy.Publisher('city_map', MarkerArray, queue_size=1)
        self.path_pub = rospy.Publisher('optimal_path', Path, queue_size=1)

        self.rate = rospy.Rate(20)


        # plot planned path
        self.path_lenth = len(path_solution)
        self._planned_path = Path()

        self._planned_path.header.stamp = rospy.Time.now()
        self._planned_path.header.frame_id = f"obstacle_frame"

        for k in range(self.path_lenth):    
            pose_stamped = PoseStamped()
            pose_stamped.pose.position.x = path_solution[k][0]
            pose_stamped.pose.position.y = path_solution[k][1]
            pose_stamped.pose.position.z = path_solution[k][2]

            pose_stamped.pose.orientation.x = 0
            pose_stamped.pose.orientation.y = 0
            pose_stamped.pose.orientation.z = 0
            pose_stamped.pose.orientation.w = 1

            self._planned_path.poses.append(pose_stamped)

        # plot the city map
        self.num_blocks = len(world_map._obstacle)
        shape = Marker.CUBE
        self.marker_array = MarkerArray()

        for i in range(self.num_blocks):
            marker = Marker()
            marker.header.frame_id = f"obstacle_frame"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "basic_shapes"
            marker.id = i
            marker.type = shape
            marker.action = Marker.ADD

            marker.pose.position.x = world_map._obstacle[i]._pos_x
            marker.pose.position.y = world_map._obstacle[i]._pos_y
            marker.pose.position.z = world_map._obstacle[i]._pos_z
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
     
            # Set the scale of the marker -- 1x1x1 here means 1m on a side
            marker.scale.x = world_map._obstacle[i]._size_x
            marker.scale.y = world_map._obstacle[i]._size_y
            marker.scale.z = world_map._obstacle[i]._size_z
        
            # Set the color -- be sure to set alpha to something non-zero!
            marker.color.r = 0.5
            marker.color.g = 0.5
            marker.color.b = 0.5
            marker.color.a = 1.0
     
            marker.lifetime = rospy.Duration()

            self.marker_array.markers.append(marker)

        # plot the initial and goal position
        start_marker = Marker()
        start_marker.header.frame_id = f"obstacle_frame"
        start_marker.header.stamp = rospy.Time.now()
        start_marker.ns = "basic_shapes"
        start_marker.id = 1000
        start_marker.type = Marker.SPHERE
        start_marker.action = Marker.ADD

        start_marker.pose.position.x = world_map._start[0]
        start_marker.pose.position.y = world_map._start[1]
        start_marker.pose.position.z = world_map._start[2]
        start_marker.pose.orientation.x = 0.0
        start_marker.pose.orientation.y = 0.0
        start_marker.pose.orientation.z = 0.0
        start_marker.pose.orientation.w = 1.0
    
        start_marker.scale.x = 3.0
        start_marker.scale.y = 3.0
        start_marker.scale.z = 3.0
    
        # Set the color -- be sure to set alpha to something non-zero!
        start_marker.color.r = 0.0
        start_marker.color.g = 0.0
        start_marker.color.b = 1.0
        start_marker.color.a = 1.0
    
        start_marker.lifetime = rospy.Duration()

        self.marker_array.markers.append(start_marker)

        goal_marker = Marker()
        goal_marker.header.frame_id = f"obstacle_frame"
        goal_marker.header.stamp = rospy.Time.now()
        goal_marker.ns = "basic_shapes"
        goal_marker.id = 2000
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD

        goal_marker.pose.position.x = world_map._goal[0]
        goal_marker.pose.position.y = world_map._goal[1]
        goal_marker.pose.position.z = world_map._goal[2]
        goal_marker.pose.orientation.x = 0.0
        goal_marker.pose.orientation.y = 0.0
        goal_marker.pose.orientation.z = 0.0
        goal_marker.pose.orientation.w = 1.0
    
        goal_marker.scale.x = 3.0
        goal_marker.scale.y = 3.0
        goal_marker.scale.z = 3.0
    
        # Set the color -- be sure to set alpha to something non-zero!
        goal_marker.color.r = 1.0
        goal_marker.color.g = 0.0
        goal_marker.color.b = 0.0
        goal_marker.color.a = 1.0
    
        goal_marker.lifetime = rospy.Duration()

        self.marker_array.markers.append(goal_marker)

        self.br = tf.TransformBroadcaster()
        

    def visualize(self):
        while not rospy.is_shutdown():
            self.map_pub.publish(self.marker_array)
            self.path_pub.publish(self._planned_path)

            for i in range(self.num_blocks):
                self.br.sendTransform(
                                (self.marker_array.markers[i].pose.position.x, 
                                 self.marker_array.markers[i].pose.position.y, 
                                 self.marker_array.markers[i].pose.position.z),
                                (0.0, 0.0, 0.0, 1.0),
                                rospy.Time.now(),
                                self.marker_array.markers[i].header.frame_id,
                                "world")
            self.rate.sleep()
            
