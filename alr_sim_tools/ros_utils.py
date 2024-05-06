from typing import List, Tuple, Any
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

import rospy
import ros_numpy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image, PointCloud2

sys.path.append(str((Path(__file__).parent / "ros_msg_srv_definitions").absolute()))
from grasping_benchmarks_ros.srv import GraspPlannerRequest

from alr_sim_tools.typing_utils import NpArray


def to_record_array(arr: NpArray, dtype: List[Tuple[str, Any]]) -> NpArray:
    """Convert a numpy array to a record array. Each column of the array is converted to
    a field of the record array.

    Args:
        arr: Numpy array
        dtype: List of tuples with the name and type of each field

    Returns:
        Numpy record array
    """

    if arr.dtype.names is not None:
        raise ValueError("Array already has names")

    if len(arr.shape) != 2:
        raise ValueError("Array must be 2D")

    if len(dtype) != arr.shape[1]:
        raise ValueError(
            "Number of fields in dtype must match number of columns in array"
        )

    rec_arr = np.empty(arr.shape[0], dtype=dtype)
    for i, (name, _) in enumerate(dtype):
        rec_arr[name] = arr[:, i]

    return rec_arr


def create_grasp_planner_request(
    rgb_image: NpArray["H,W,3", float],
    depth_image: NpArray["H,W", float],
    segmentation_image: NpArray["H,W", int],
    pointcloud_points: NpArray["N,3", float],
    pointcloud_colors: NpArray["N,3", int],
    camera_position: NpArray["3", float],
    camera_quaternion: NpArray["4", float],
    camera_intrinsics: NpArray["3, 3", float],
    camera_height: float,
    camera_width: float,
    number_of_candidates: int = 1,
) -> GraspPlannerRequest:
    """Create a GraspPlannerRequest ROS message from all necessary data"""

    assert GraspPlannerRequest is not None, "Could not import grasping_benchmarks_ros"

    planner_req = GraspPlannerRequest()

    # TODO add headers to the messages
    # header = Header()
    # header.stamp = rospy.Time.now()
    # header.frame_id = "rgbd_cam"

    planner_req.color_image = ros_numpy.msgify(Image, rgb_image, "rgb8")

    planner_req.depth_image = ros_numpy.msgify(
        Image, (depth_image * 1000).astype("uint16"), "16UC1"
    )

    planner_req.seg_image = ros_numpy.msgify(
        Image, segmentation_image.astype(np.uint8), "8UC1"
    )

    planner_req.camera_info = ros_numpy.msgify(
        CameraInfo, camera_intrinsics, height=camera_height, width=camera_width
    )

    # TODO rgb info is not included yet
    planner_req.cloud = ros_numpy.msgify(
        PointCloud2,
        to_record_array(
            pointcloud_points, [("x", np.float32), ("y", np.float32), ("z", np.float32)]
        ),
    )

    view_point = np.eye(4)
    view_point[:3, :3] = Rotation.from_quat(camera_quaternion[[1, 2, 3, 0]]).as_matrix()
    view_point[:3, 3] = camera_position
    planner_req.view_point = ros_numpy.msgify(PoseStamped, view_point)

    planner_req.n_of_candidates = number_of_candidates

    return planner_req
