#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import Joy
from geometry_msgs.msg import TransformStamped as transform
from std_msgs.msg import Float32


class JoyStickInterface:
    """
    Simple XBOX Interface
    """

    def __init__(self):
        # ======================= Joystick ==========================================
        # translation cmd
        self.Tran_cmd = np.zeros(2)

        # orientation cmd
        self.Orien_cmd: float
        self.Orien_cmd = 0.0

        self.joy_subscriber = rospy.Subscriber("/joy", Joy, self.JoyCallback)

        self.trans_publisher = rospy.Publisher(
            "/Tele/DeltaTranslation", transform, queue_size=1
        )
        self.orien_publisher = rospy.Publisher(
            "/Tele/DeltaWristJoint", Float32, queue_size=1
        )

        self.rate = rospy.Rate(20)

    def JoyCallback(self, msg: Joy):
        # =============== msg.transform.translation.x y z
        self.Tran_cmd[0] = msg.axes[0]
        self.Tran_cmd[1] = msg.axes[1]

        self.Orien_cmd = msg.axes[3]

        # publish the post-processed tele-cmds
        t_cmd = transform()
        t_cmd.transform.translation.x = self.Tran_cmd[0]
        t_cmd.transform.translation.y = self.Tran_cmd[1]
        t_cmd.transform.translation.z = 0.0
        t_cmd.transform.rotation.x = 0.0
        t_cmd.transform.rotation.y = 0.0
        t_cmd.transform.rotation.z = 0.0
        t_cmd.transform.rotation.w = 1.0

        orien_cmd = Float32()
        orien_cmd.data = self.Orien_cmd

        self.trans_publisher.publish(t_cmd)
        self.orien_publisher.publish(orien_cmd)


if __name__ == "__main__":
    rospy.init_node("Teleoperation_Interface", anonymous=False)
    js_interface = JoyStickInterface()
    js_interface.rate.sleep()
    rospy.spin()
