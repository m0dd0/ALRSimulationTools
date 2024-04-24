import open3d as o3d
import numpy as np

from alr_sim_tools.typing_utils import NpArray


def get_segmented_pointcloud(
    depth_image: NpArray["H,W", np.float],
    segmentation_image: NpArray["H,W", np.bool],
    rgb_image: NpArray["H,W,3", np.int32],
    camera_position: NpArray["3", np.float],
    camera_quaternion: NpArray["4", np.float],
    camera_intrinsics: NpArray["3,3", np.float],
):
    depth_image[segmentation_image == 0] = -1

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_image, depth_image, convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        image=rgbd,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        project_valid_depth_only=True,
    )

    # flip the orientation, so it looks upright, not upside-down
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # draw_geometries([pcd])  # visualize the point cloud
