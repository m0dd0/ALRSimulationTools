from dataclasses import dataclass
from typing import Tuple

import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation

from alr_sim_tools.typing_utils import NpArray


@dataclass
class CameraData:
    rgb_image: NpArray["H,W,3", np.uint8]
    depth_image: NpArray["H,W", np.float32]
    segmentation_image: NpArray["H,W", np.int64]
    segmentation_image_all: NpArray["H,W", np.int32]
    pointcloud_points: NpArray["N,3", np.float64]
    pointcloud_colors: NpArray["N,3", np.float64]  # in 0-1 range
    pointcloud_segmented_points: NpArray["M,3", np.float64]
    pointcloud_segmented_colors: NpArray["M,3", np.float64]
    camera_position: NpArray["3", np.float64]
    camera_quaternion: NpArray["4", np.float64]
    camera_intrinsics: NpArray["3, 3", np.float64]


def quaternion_position_to_homogeneous_matrix(
    position: NpArray["3", float], quaternion: NpArray["4", float]
) -> NpArray["4,4", float]:
    """Converts a position and quaternion to a homogeneous matrix.

    Args:
        position (np.ndarray): Position (3,)
        quaternion (np.ndarray): Quaternion (w,x,y,z) (4,)
    """
    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation
    homogeneous_matrix[:3, 3] = position
    return homogeneous_matrix


def get_segmented_pointcloud(
    depth_image: NpArray["H,W", float],
    segmentation_image: NpArray["H,W", bool],
    rgb_image: NpArray["H,W,3", int],
    camera_position: NpArray["3", float],
    camera_quaternion: NpArray["4", float],
    camera_intrinsics: NpArray["3,3", float],
) -> Tuple[NpArray["N,3", float], NpArray["N,3", float]]:
    """Converts a depth image, segmentation image, and RGB image into a segmented point cloud.

    Args:
        depth_image (np.ndarray): Depth image of the scene (H x W)
        segmentation_image (np.ndarray): Segmentation image of the scene (H x W)
        rgb_image (np.ndarray): RGB image of the scene (H x W x 3)
        camera_position (np.ndarray): Camera position (3,)
        camera_quaternion (np.ndarray): Camera quaternion (4,)
        camera_intrinsics (np.ndarray): Camera intrinsics (3 x 3)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Segmented point cloud points (N x 3), Segmented point cloud colors (N x 3)
    """
    # TODO needs to be tested

    # invalidate all points that are not part of the segmentation
    depth_image[segmentation_image == 0] = -1

    height, width = depth_image.shape
    rgb_image = o3d.geometry.Image(rgb_image.astype(np.uint8))
    depth_image = o3d.geometry.Image(depth_image.astype(np.float32))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_image, depth_image, convert_rgb_to_intensity=False
    )

    intrinsics = o3d.camera.PinholeCameraIntrinsic(height, width, camera_intrinsics)
    pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        image=rgbd,
        extrinsic=quaternion_position_to_homogeneous_matrix(
            camera_position, camera_quaternion
        ),
        intrinsic=intrinsics,
        project_valid_depth_only=True,
        depth_scale=1.0,
    )

    pointcloud_points, pointcloud_colors = np.asarray(pointcloud.points), np.asarray(
        pointcloud.colors
    )

    return pointcloud_points, pointcloud_colors
