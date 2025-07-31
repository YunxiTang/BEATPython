"""
Advanced mode
"""

import pyrealsense2 as rs
import numpy as np
import einops
import cv2


def get_connected_devices_serial():
    serials = list()
    for d in rs.context().devices:
        if d.get_info(rs.camera_info.name).lower() != "platform camera":
            serial = d.get_info(rs.camera_info.serial_number)
            product_line = d.get_info(rs.camera_info.product_line)
            if product_line == "D400":
                # only works with D400 series
                serials.append(serial)
    serials = sorted(serials)
    return serials


if __name__ == "__main__":
    print(get_connected_devices_serial())
