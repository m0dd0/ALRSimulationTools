from pathlib import Path
from typing import List, Tuple

import numpy as np

from alr_sim.sims.SimFactory import SimRepository
from alr_sim.sims.universal_sim.PrimitiveObjects import Box


def create_sample_data(
    factory_string: str = "mujoco",
    cam_pos: Tuple[float] = (0.32, 0.0, 0.56),
    cam_quat: Tuple[float] = (0, 1, 0, 0),
    cam_height: int = 224,
    cam_width: int = 224,
    home_pos: Tuple[float] = (0.32, 0.0, 0.56),
    home_quat: Tuple[float] = (0, 1, 0, 0),
    object_list: List = None,
    target_obj_name: str = None,
    destination_path: Path = None,
    render_mode=Scene.RenderMode.BLIND,
):
    if destination_path is None:
        destination_path = (
            Path.home()
            / "Documents"
            / "robotic-grasping"
            / "grconvnet"
            / "data"
            / "examples"
            / "simulation"
        )

    if object_list is None:
        box1 = Box(
            name="box1",
            init_pos=[0.5, -0.2, 0.0],
            init_quat=[0, 1, 0, 0],
            rgba=[0.1, 0.25, 0.3, 1],
        )
        object_list = [box1]
        target_obj_name = "box1"

    # Generate the chosen Scene and Agent
    sim_factory = SimRepository.get_factory(factory_string)
    scene = sim_factory.create_scene(
        object_list=object_list, dt=0.0002, render=render_mode
    )
    agent = sim_factory.create_robot(scene, dt=0.0002)

    # configure camera
    if cam_type == "rgbd_cage":
        cam = scene.get_object("rgbd_cage")
    elif cam_type == "inhand":
        cam = agent.inhand_cam

    cam.set_cam_params(height=cam_height, width=cam_width)

    # start simulation
    scene.start()

    # go to start position
    agent.gotoCartPositionAndQuat(home_pos, home_quat, duration=0.5)

    # get camera data
    rgb_img, depth_img = cam.get_image()
    point_cloud = cam.calc_point_cloud()
    target_obj_id = scene.get_obj_seg_id(obj_name=target_obj_name)
    seg_img_orig = cam.get_segmentation(depth=False)

    seg_img = np.where(seg_img_orig == target_obj_id, 1, 0)

    point_cloud_seg = cam.calc_point_cloud_from_images(
        rgb_img, np.where(depth_img * seg_img == 0, np.nan, depth_img)
    )

    # get camera parameters
    cam_intrinsics = cam.intrinsics
    cam_pos, cam_quat = cam.get_cart_pos_quat()

    reset_scene(factory_string, scene, agent)

    # save data
    return {
        "rgb_img": rgb_img,
        "depth_img": depth_img,
        "seg_img": seg_img,
        "point_cloud_seg": point_cloud_seg,
        "point_cloud": point_cloud,
        "cam_pos": cam_pos,
        "cam_quat": cam_quat,
        "cam_intrinsics": cam_intrinsics,
    }
