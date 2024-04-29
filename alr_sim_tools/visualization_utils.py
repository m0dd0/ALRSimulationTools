from alr_sim_tools.typing_utils import NpArray

from matplotlib import pyplot as plt
import open3d as o3d


def plot_pointcloud(
    pointcloud_points: NpArray["N,3", float],
    pointcloud_colors: NpArray["N,3", float] = None,
    method: str = "matplotlib",
):
    """Plot a point cloud.

    Args:
        pointcloud_points (NpArray["N,3", float]): Point cloud coordinates.
        pointcloud_colors (NpArray["N,3", float], optional): Point cloud colors. Defaults to None.
        method (str, optional): Method to use for plotting. Defaults to "matplotlib".
    """
    if method == "matplotlib":
        # Plot point cloud
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            pointcloud_points[:, 0],
            pointcloud_points[:, 1],
            pointcloud_points[:, 2],
            c=pointcloud_colors,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        return fig, ax

    elif method == "open3d":
        # Create an Open3D point cloud object
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(pointcloud_points)
        pointcloud.colors = o3d.utility.Vector3dVector(pointcloud_colors)

        # Create a visualizer object
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Add the point cloud to the visualizer
        vis.add_geometry(pointcloud)

        # Set the camera view
        vis.get_view_control().set_front([0, 0, -1])
        vis.get_view_control().set_lookat([0, 0, 0])
        vis.get_view_control().set_up([0, -1, 0])
        vis.get_view_control().set_zoom(0.5)

        # Run the visualizer
        vis.run()
        vis.destroy_window()
