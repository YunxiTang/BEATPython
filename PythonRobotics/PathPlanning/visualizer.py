import rospy
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import tf
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PolygonStamped, Polygon, Point32


class Visualizer:
    def __init__(self, world_map, path_solution=None, path_list=None) -> None:
        self._world_map = world_map
        self._path_solution = path_solution
        
        try:
            rospy.init_node('result_visualizer', anonymous=False)
        except rospy.exceptions.ROSException as e:
            pass
        self.map_pub = rospy.Publisher('city_map', MarkerArray, queue_size=1)
        
        if path_solution is not None:
            self._list = False
            self.path_pub = rospy.Publisher('optimal_path', Path, queue_size=1)
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

        else:
            self._list = True
            self.path_pub1 = rospy.Publisher('optimal_path1', Path, queue_size=1)
            self.path_pub2 = rospy.Publisher('optimal_path2', Path, queue_size=1)
            self.path_pub3 = rospy.Publisher('optimal_path3', Path, queue_size=1)
            self.path_pub4 = rospy.Publisher('optimal_path4', Path, queue_size=1)
            self.path_pub5 = rospy.Publisher('optimal_path5', Path, queue_size=1)

            self.path_lenth = len(path_list[0])
            self._planned_path1 = Path()
            self._planned_path2 = Path()
            self._planned_path3 = Path()
            self._planned_path4 = Path()
            self._planned_path5 = Path()

            
            self._planned_path1.header.stamp = rospy.Time.now()
            self._planned_path1.header.frame_id = f"obstacle_frame"
            for k in range(self.path_lenth):    
                pose_stamped = PoseStamped()
                pose_stamped.pose.position.x = path_list[0][k][0]
                pose_stamped.pose.position.y = path_list[0][k][1]
                pose_stamped.pose.position.z = path_list[0][k][2]

                pose_stamped.pose.orientation.x = 0
                pose_stamped.pose.orientation.y = 0
                pose_stamped.pose.orientation.z = 0
                pose_stamped.pose.orientation.w = 1
                self._planned_path1.poses.append(pose_stamped)

            self._planned_path2.header.stamp = rospy.Time.now()
            self._planned_path2.header.frame_id = f"obstacle_frame"
            for k in range(self.path_lenth):    
                pose_stamped = PoseStamped()
                pose_stamped.pose.position.x = path_list[1][k][0]
                pose_stamped.pose.position.y = path_list[1][k][1]
                pose_stamped.pose.position.z = path_list[1][k][2]

                pose_stamped.pose.orientation.x = 0
                pose_stamped.pose.orientation.y = 0
                pose_stamped.pose.orientation.z = 0
                pose_stamped.pose.orientation.w = 1
                self._planned_path2.poses.append(pose_stamped)

            self._planned_path3.header.stamp = rospy.Time.now()
            self._planned_path3.header.frame_id = f"obstacle_frame"
            for k in range(self.path_lenth):    
                pose_stamped = PoseStamped()
                pose_stamped.pose.position.x = path_list[2][k][0]
                pose_stamped.pose.position.y = path_list[2][k][1]
                pose_stamped.pose.position.z = path_list[2][k][2]

                pose_stamped.pose.orientation.x = 0
                pose_stamped.pose.orientation.y = 0
                pose_stamped.pose.orientation.z = 0
                pose_stamped.pose.orientation.w = 1
                self._planned_path3.poses.append(pose_stamped)

            self._planned_path4.header.stamp = rospy.Time.now()
            self._planned_path4.header.frame_id = f"obstacle_frame"
            for k in range(self.path_lenth):    
                pose_stamped = PoseStamped()
                pose_stamped.pose.position.x = path_list[3][k][0]
                pose_stamped.pose.position.y = path_list[3][k][1]
                pose_stamped.pose.position.z = path_list[3][k][2]

                pose_stamped.pose.orientation.x = 0
                pose_stamped.pose.orientation.y = 0
                pose_stamped.pose.orientation.z = 0
                pose_stamped.pose.orientation.w = 1
                self._planned_path4.poses.append(pose_stamped)

            self._planned_path5.header.stamp = rospy.Time.now()
            self._planned_path5.header.frame_id = f"obstacle_frame"
            for k in range(self.path_lenth):    
                pose_stamped = PoseStamped()
                pose_stamped.pose.position.x = path_list[4][k][0]
                pose_stamped.pose.position.y = path_list[4][k][1]
                pose_stamped.pose.position.z = path_list[4][k][2]

                pose_stamped.pose.orientation.x = 0
                pose_stamped.pose.orientation.y = 0
                pose_stamped.pose.orientation.z = 0
                pose_stamped.pose.orientation.w = 1
                self._planned_path5.poses.append(pose_stamped)

        self.rate = rospy.Rate(20)


        # plot planned path
        

        # plot the city map
        self.num_blocks = len(world_map._obstacle)
        shape = Marker.CUBE
        self.marker_array = MarkerArray()

        marker = Marker()
        marker.header.frame_id = f"obstacle_frame"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "basic_shapes"
        marker.id = 987
        marker.type = shape
        marker.action = Marker.ADD

        marker.pose.position.x = 100.
        marker.pose.position.y = 100.
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
    
        # Set the scale of the marker -- 1x1x1 here means 1m on a side
        marker.scale.x = 200.
        marker.scale.y = 200.
        marker.scale.z = 0.5
    
        # Set the color -- be sure to set alpha to something non-zero!
        marker.color.r = 0.8
        marker.color.g = 0.8
        marker.color.b = 0.8
        marker.color.a = 1.0

        marker.lifetime = rospy.Duration()
        self.marker_array.markers.append(marker)

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
            marker.color.r = 0.7
            marker.color.g = 0.7
            marker.color.b = 0.7
            marker.color.a = 1.0

            if world_map._obstacle[i]._wall:
                marker.color.a = 0.001
     
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

            if False:
                self.path_pub.publish(self._planned_path)
            else:
                self.path_pub1.publish(self._planned_path1)
                self.path_pub2.publish(self._planned_path2)
                self.path_pub3.publish(self._planned_path3)
                self.path_pub4.publish(self._planned_path4)
                self.path_pub5.publish(self._planned_path5)


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


    def visualize_with_plane(self):
        boundaryPolygon  = rospy.Publisher('Passage_Polygon', PolygonStamped, queue_size=1)
        
        boundaryList = PolygonStamped()
        boundaryPoints = Polygon()
        
        p1 = Point32()
        p1.x = 0.
        p1.y = 0.
        p1.z = 100.

        p2 = Point32()
        p2.x = 100.
        p2.y = 0.
        p2.z = 0.

        p3 = Point32()
        p3.x = 0.
        p3.y = 100.
        p3.z = 0.

       
        boundaryList.polygon.points = [p1, p2, p3]
       
        boundaryList.header.frame_id = f"obstacle_frame"
        boundaryList.header.stamp = rospy.Time.now()
        
        while not rospy.is_shutdown():
            self.map_pub.publish(self.marker_array)
            boundaryPolygon.publish(boundaryList)
            self.rate.sleep()
            
