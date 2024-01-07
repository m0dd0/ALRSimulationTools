from typing import Optional, Tuple
from pathlib import Path

import numpy as np

import ros_numpy
import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import CameraInfo, Image, PointCloud2


from alr_sim_tools.typing_utils import NpArray

from alr_sim.utils.point_clouds import rgb_array_to_uint32, rgb_float_to_int

# import tf2_ros
# from rospy import Publisher

# from alr_sim.core.Camera import Camera
# from alr_sim.utils.geometric_transformation import posRotMat2TFMat
# from alr_sim.utils.point_clouds import rgb_array_to_uint32, rgb_float_to_int
import sys

try:
    sys.path.append(str((Path(__file__).parent / "ros_msg_srv_definitions").absolute()))
    from grasping_benchmarks_ros.srv import (
        GraspPlannerRequest,
        GraspPlannerResponse,
        GraspPlanner,
    )
except ImportError:
    print("Could not import grasping_benchmarks_ros")
    GraspPlannerRequest = None
    GraspPlannerResponse = None
    GraspPlanner = None


def pos_quat_to_ros_msg(
    pos: NpArray["3", float],
    quat: NpArray["4", float],
    header: Optional[Header] = None,
) -> PoseStamped:
    """Convert a position and quaternion to a ROS PoseStamped message

    Args:
        pos: Position in x,y,z
        quat: Quaternion in w,x,y,z
        header: ROS header

    Returns:
        ROS PoseStamped message
    """
    pos_msg = PoseStamped()

    if header is not None:
        pos_msg.header = header

    pos_msg.pose.orientation.w = quat[0]
    pos_msg.pose.orientation.x = quat[1]
    pos_msg.pose.orientation.y = quat[2]
    pos_msg.pose.orientation.z = quat[3]
    pos_msg.pose.position.x = pos[0]
    pos_msg.pose.position.y = pos[1]
    pos_msg.pose.position.z = pos[2]

    return pos_msg


def pc_to_ros_msg(
    pc_points: NpArray["N, 3", float],
    pc_colors: NpArray["N, 3", int],
    header: Optional[Header] = None,
) -> PointCloud2:
    """
    Convert a point cloud to a ROS PointCloud2 message
    :param pc_points: Point cloud points in x,y,z
    :param pc_colors: Point cloud colors in r,g,b
    :param header: ROS header
    :return: ROS PointCloud2 message
    """
    pc_data = np.zeros(
        pc_points.shape[0],
        dtype=[
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("rgb", np.uint32),
        ],
    )
    pc_data["x"] = pc_points[:, 0]
    pc_data["y"] = pc_points[:, 1]
    pc_data["z"] = pc_points[:, 2]

    rgb = rgb_float_to_int(pc_colors)
    pc_data["rgb"] = rgb_array_to_uint32(rgb)

    pc_msg = ros_numpy.msgify(PointCloud2, pc_data)

    if header is not None:
        pc_msg.header = header

    return pc_msg


def depth_img_to_ros_msg(
    depth_img: NpArray["H,W", float], header: Optional[Header] = None
) -> Image:
    """
    Convert a depth image to a ROS Image message
    :param depth_img: Depth image in width,height,meters
    :param header: ROS header
    :return: ROS 16UC1 Image message
    """
    depth_img_ui16 = (depth_img * 1000).astype("uint16")
    depth_msg = ros_numpy.msgify(Image, depth_img_ui16, "16UC1")

    if header is not None:
        depth_msg.header = header

    return depth_msg


def rgb_img_to_ros_msg(
    rgb_img: NpArray["H,W,3", int], header: Optional[Header] = None
) -> Image:
    """
    Convert a RGB image to a ROS Image message
    :param rgb_img: RGB image in width,height,rgb
    :param header: ROS header
    :return: ROS rgb8 Image message
    """
    rgb_msg = ros_numpy.msgify(Image, rgb_img, "rgb8")

    if header is not None:
        rgb_msg.header = header

    return rgb_msg


def seg_img_to_ros_msg(
    seg_img: NpArray["H,W", int], header: Optional[Header] = None
) -> Image:
    """
    Convert a segmentation image to a ROS Image message
    :param seg_img: Segmentation image in width,height,(true, false)
    :param header: ROS header
    :return: ROS 8UC1 Image message
    """
    seg_msg = ros_numpy.msgify(Image, seg_img.astype(np.uint8), "8UC1")

    if header is not None:
        seg_msg.header = header

    return seg_msg


def cam_intrinsics_to_ros_msg(
    cam_intrinsics: NpArray["3, 3", float],
    height: int,
    width: int,
    header: Optional[Header] = None,
) -> CameraInfo:
    """
    Convert camera intrinsics to a ROS CameraInfo message
    :param cam_intrinsics: Camera intrinsics matrix
    :param height: Camera height
    :param width: Camera width
    :param header: ROS header
    :return: ROS CameraInfo message
    """

    # http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
    camera_info = CameraInfo()

    if header is not None:
        camera_info.header = header
        # camera_info.cam_frame = header.frame_id

    camera_info.height = height
    camera_info.width = width
    camera_info.distortion_model = "plumb_bob"
    camera_info.D = np.zeros(
        5
    )  # Since we got simulated data all of this is pretty much without distortion
    camera_info.K = cam_intrinsics.flatten()
    camera_info.R = np.eye(
        3, dtype=float
    ).flatten()  # Only in stereo cameras, otherwise diagonal 1 matrix

    P = np.zeros((3, 4))
    P[0:3, 0:3] = cam_intrinsics
    camera_info.P = P.flatten()

    return camera_info


def create_grasp_planner_request(
    rgb_img: NpArray["H,W,3", float],
    depth_img: NpArray["H,W", float],
    seg_img: NpArray["H,W", int],
    pc_points: NpArray["N,3", float],
    pc_colors: NpArray["N,3", int],
    cam_pos: NpArray["3", float],
    cam_quat: NpArray["4", float],
    cam_intrinsics: NpArray["3, 3", float],
    cam_height: float,
    cam_width: float,
    num_of_candidates: int = 1,
) -> GraspPlannerRequest:
    """
    Create a GraspPlannerRequest ROS message from all necessary data
    :param rgb_img: RGB image in width,height,rgb
    :param depth_img: Depth image in width,height,meters
    :param seg_img: Segmentation image in width,height,(1, 0)
    :param pc_points: Point cloud in x,y,z
    :param pc_colors: Point cloud colors in r,g,b
    :param cam_pos: Camera position in x,y,z
    :param cam_quat: Camera orientation in x,y,z,w
    :param cam_intrinsics: Camera intrinsics matrix
    :param cam_height: Camera height
    :param cam_width: Camera width
    :param num_of_candidates: Number of grasp candidates to return
    :return: GraspPlannerRequest ROS message
    """
    assert GraspPlannerRequest is not None, "Could not import grasping_benchmarks_ros"

    planner_req = GraspPlannerRequest()

    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "rgbd_cam"

    planner_req.color_image = rgb_img_to_ros_msg(rgb_img, header)
    planner_req.depth_image = depth_img_to_ros_msg(depth_img, header)
    planner_req.seg_image = seg_img_to_ros_msg(seg_img, header)
    planner_req.cloud = pc_to_ros_msg(pc_points, pc_colors, header)
    planner_req.view_point = pos_quat_to_ros_msg(cam_pos, cam_quat, header)
    planner_req.camera_info = cam_intrinsics_to_ros_msg(
        cam_intrinsics, cam_height, cam_width, header
    )
    planner_req.n_of_candidates = num_of_candidates

    return planner_req


# def get_filtered_pc(rgb_img, depth_img, seg_img, seg_obj_id, cam):
#     """
#     Get the PointCloud only of the object of interest
#     If seg_obj_id is None, we assume that the segmentation image is binary
#     :param rgb_img: RGB image in width,height,rgb
#     :param depth_img: Depth image in width,height,meters
#     :param seg_img: Segmentation image in width,height,(true, false)
#     :param seg_obj_id: Segmentation object id
#     :param cam: SimFramework Camera object
#     :return: PointCloud in (x,y,z) (r,g,b)
#     """
#     if seg_obj_id is not None:
#         seg_img = np.where(seg_img == seg_obj_id, 1, 0)

#     # Segmenting the Images before going to the PointCloud
#     seg_depth_img = depth_img.copy()
#     seg_depth_img[seg_img != 1] = np.nan
#     pc_points, pc_colors = cam.calc_point_cloud_from_images(
#         rgb_img=rgb_img, depth_img=seg_depth_img
#     )

#     return pc_points, pc_colors


# def convert_ros_grasp_to_action(reply: GraspPlannerResponse, table_height: float):
#     """
#     Converts the ROS Grasp Planner reply to a grasp action
#     :param reply: ROS Grasp Planner reply
#     :param table_height: Height of the tabletop in world coordinates
#     :return: Grasp Action (Position, Quaternion, Width)
#     """
#     grasps = []
#     for grasp_candidate in reply.grasp_candidates:
#         pose = grasp_candidate.pose.pose
#         grasp = {
#             "position": [pose.position.x, pose.position.y, pose.position.z],
#             "quat": [
#                 pose.orientation.w,
#                 pose.orientation.x,
#                 pose.orientation.y,
#                 pose.orientation.z,
#             ],
#             "width": grasp_candidate.width.data,
#             "score": grasp_candidate.score.data,
#         }

#         grasp_collides = check_grasp_table_collision(
#             grasp_pos=grasp["position"],
#             grasp_quat=grasp["quat"],
#             width=grasp["width"],
#             table_z=table_height,
#         )

#         if not grasp_collides:
#             grasps.append(grasp)

#     if len(grasps) == 0:
#         print("There was no grasp without table collision detected")
#         return None, None, None, True

#     best_grasp_idx = np.argmax([grasp.get("score") for grasp in grasps])
#     best_grasp = grasps[best_grasp_idx]
#     best_grasp_pos = best_grasp.get("position")
#     best_grasp_quat = best_grasp.get("quat")
#     best_grasp_width = best_grasp.get("width")

#     return best_grasp_pos, best_grasp_quat, best_grasp_width, False


# def call_grasp_planner(
#     cam: Camera,
#     service_id: str,
#     seg_obj_id: Optional[int] = None,
#     num_of_candidates: int = 1,
#     filter_pc: bool = True,
#     visualize: bool = False,
#     table_height: float = -0.02,
# ) -> Tuple[NDArray[Shape["3"], Float], NDArray[Shape["4"], Float]]:
#     """
#     Call the grasp planner service and return the grasp poses
#     :param cam: SimFramework Camera object
#     :param service_id: Service id of the grasp planner
#     :param seg_obj_id: Segmentation object id
#     :param num_of_candidates: Number of grasp candidates to return
#     :param filter_pc: Filter the PointCloud to only contain the object of interest (bool)
#     :param visualize: Visualize the PointCloud (bool)
#     :param table_height: Table y position (meters)
#     :return: Grasp poses in (x,y,z) (x,y,z,w), and three images of grasp candidates
#     """

#     # Get Sensor Data
#     seg_img = cam.get_segmentation(depth=False)
#     rgb_img, depth_img = cam.get_image(depth=True)
#     depth_img[depth_img == 0] = np.nan
#     depth_img[depth_img > 1] = np.nan

#     # Get Cam Info
#     cam_pos, cam_quat = cam.get_cart_pos_quat()
#     cam_intrinsics = cam.intrinsics
#     cam_height = cam.height
#     cam_width = cam.width

#     # Create Filtered PointCloud and transform
#     if filter_pc:
#         pc_points, pc_colors = get_filtered_pc(
#             rgb_img, depth_img, seg_img, seg_obj_id, cam
#         )
#         pc_points = transform_pc(pc_points, cam_pos, cam_quat)
#     else:
#         pc_points, pc_colors = cam.calc_point_cloud_from_images(
#             rgb_img=rgb_img, depth_img=depth_img
#         )

#     # Create the Grasp Request out of that
#     grasp_req = create_grasp_planner_request(
#         rgb_img,
#         depth_img,
#         seg_img,
#         pc_points,
#         pc_colors,
#         cam_pos,
#         cam_quat,
#         cam_intrinsics,
#         cam_height,
#         cam_width,
#         num_of_candidates=num_of_candidates,
#     )

#     # And call the GraspService
#     rospy.wait_for_service(service_id, timeout=30.0)
#     grasp_planner = rospy.ServiceProxy(service_id, GraspPlanner)

#     # Convert the ROS Result into a more usable format
#     try:
#         reply: GraspPlannerResponse = grasp_planner(grasp_req)

#         grasps = []
#         counter_collision_grasps = 0
#         for grasp_candidate in reply.grasp_candidates:
#             pose = grasp_candidate.pose.pose
#             grasp = {
#                 "position": [pose.position.x, pose.position.y, pose.position.z],
#                 "quat": [
#                     pose.orientation.w,
#                     pose.orientation.x,
#                     pose.orientation.y,
#                     pose.orientation.z,
#                 ],
#                 "width": grasp_candidate.width.data,
#                 "score": grasp_candidate.score.data,
#             }

#             grasp_collides = check_grasp_table_collision(
#                 grasp_pos=grasp["position"],
#                 grasp_quat=grasp["quat"],
#                 width=grasp["width"],
#                 table_z=table_height,
#             )

#             if not grasp_collides:
#                 grasps.append(grasp)
#             else:
#                 counter_collision_grasps += 1

#         if len(grasps) == 0:
#             print("There was no grasp without table collision detected")
#             return None

#         if counter_collision_grasps > 0:
#             print(
#                 f"Found {counter_collision_grasps} grasps with collision from {len(reply.grasp_candidates)}"
#             )

#     except rospy.ServiceException as e:
#         print("Service {} call failed: {}".format(grasp_planner.resolved_name, e))
#         return None

#     # Visualization for Logging
#     pc_points, pc_colors = cam.calc_point_cloud_from_images(
#         rgb_img=rgb_img, depth_img=depth_img
#     )
#     world_points = transform_pc(pc_points, cam_pos, cam_quat)

#     best_grasp_idx = np.argmax([grasp.get("score") for grasp in grasps])
#     best_grasp = grasps[best_grasp_idx]
#     best_grasp_pos = best_grasp.get("position")
#     best_grasp_quat = best_grasp.get("quat")
#     best_grasp_width = best_grasp.get("width")

#     img_mid, img_left, img_right = render_pointcloud_with_panda_grippers(
#         world_points, pc_colors, best_grasp_pos, best_grasp_quat, best_grasp_width
#     )

#     # Visualize the pointcloud in UI
#     if visualize:
#         visualize_pointcloud_with_panda_grippers(
#             world_points, pc_colors, grasps=grasps, chosen_idx=best_grasp_idx
#         )

#     return grasps, (img_mid, img_left, img_right)
