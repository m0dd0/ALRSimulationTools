from typing import List, Tuple, Dict
import logging

import numpy as np

from alr_sim.sims.SimFactory import SimRepository
from alr_sim.sims.universal_sim.PrimitiveObjects import Box
from alr_sim.core import Scene, RobotBase
from alr_sim.sims.mj_beta import MjCamera


def reset_scene(factory_string: str, scene: Scene, agent):
    """
    Reset the scene and agent.

    Args:
        factory_string (str): specifying the simulation factory
        scene (Scene): the scene object
        agent (Agent): the agent object
    """
    if factory_string == "pybullet":
        import pybullet as p

        p.disconnect()
    elif factory_string == "mujoco":
        del scene.viewer
        del scene.model
        del scene.sim
    elif factory_string == "mj_beta":
        from alr_sim.sims.mj_beta.mj_utils.mj_render_singleton import (
            reset_singleton,
        )

        reset_singleton()
        if scene.render_mode == Scene.RenderMode.HUMAN:
            scene.viewer.close()
        del scene.viewer
        del scene.model
        del scene.data
    if agent is not None:
        del agent
    del scene


def record_camera_data(
    factory_string: str = "mj_beta",
    cam_pos: Tuple[float] = (0.5, 0.0, 1.0),
    cam_quat: Tuple[float] = (-0.70710678, 0, 0, 0.70710678),
    cam_height: int = 480,
    cam_width: int = 640,
    robot_pos: Tuple[float] = (0.0, 0.5, -0.01),
    robot_quat: Tuple[float] = (0, 1, 0, 0),
    object_list: List = None,
    target_obj_name: str = None,
    render_mode: Scene.RenderMode = Scene.RenderMode.BLIND,
    wait_time: float = 0.1,
    move_duration: float = 4,
) -> Tuple[Dict[str, np.array], Scene, RobotBase]:
    """
    Create a scene with a camera and returns a bunch of data captured by the camera.

    Args:
        factory_string (str, optional): string specifying the simulation factory. Defaults to "mj_beta".
        cam_pos (Tuple[float], optional): position of the camera. Defaults to (0.5, 0.0, 1.0).
        cam_quat (Tuple[float], optional): quaternion of the camera. Defaults to (-0.70710678, 0, 0, 0.70710678).
        cam_height (int, optional): height of the camera image in pixels. Defaults to 480.
        cam_width (int, optional): width of the camera image in pixels. Defaults to 640.
        robot_pos (Tuple[float], optional): position of the robot eef. Defaults to (0.0, 0.5, -0.01).
        robot_quat (Tuple[float], optional): quaternion of the robot eef. Defaults to (0, 1, 0, 0).
        object_list (List, optional): list of objects in the scene. Defaults to a box named "box1".
        target_obj_name (str, optional): name of the object to segment. Defaults to None.
        render_mode (Scene.RenderMode, optional): render mode of the scene. Defaults to Scene.RenderMode.BLIND.
        wait_time (float, optional): wait time after the robot has moved. Defaults to 0.1.
        move_duration (float, optional): duration of the robot movement. Defaults to 4.

    Returns:
        Tuple[Dict[str, np.array], Scene, RobotBase]:
            dictionary containing the camera data. Keys are:
            - rgb_img: rgb image (HxBx3)
            - depth_img: depth image (HxB)
            - seg_img: segmentation image (HxB)
            - seg_img_all: segmentation image with all objects (HxB)
            - point_cloud_points: point cloud (Nx3)
            - point_cloud_colors: point cloud colors (Nx3)
            - point_cloud_seg_points: point cloud of the segmented object (Mx3)
            - point_cloud_seg_colors: point cloud colors of the segmented object (Mx3)
            - cam_pos: camera position (3)
            - cam_quat: camera quaternion (4)
            - cam_intrinsics: camera intrinsics (3x3)
    """
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
    if factory_string == "mj_beta":
        cam = MjCamera("my_cam", init_pos=cam_pos, init_quat=cam_quat)
    elif factory_string == "mujoco":
        raise NotImplementedError()
        # cam = MujocoCamera("my_cam", init_pos=cam_pos, init_quat=cam_quat)
    else:
        raise NotImplementedError()
    scene.add_object(cam)
    cam.set_cam_params(height=cam_height, width=cam_width)

    # start simulation
    scene.start()

    # go to start position
    agent.gotoCartPositionAndQuat(robot_pos, robot_quat, duration=move_duration)
    agent.wait(wait_time)

    # get camera data
    rgb_img, depth_img = cam.get_image()
    point_cloud_points, point_cloud_colors = cam.calc_point_cloud()
    target_obj_id = scene.get_obj_seg_id(obj_name=target_obj_name)
    seg_img_orig = cam.get_segmentation(depth=False)

    seg_img = np.where(seg_img_orig == target_obj_id, 1, 0)

    point_cloud_seg_points, point_cloud_seg_colors = cam.calc_point_cloud_from_images(
        rgb_img, np.where(depth_img * seg_img == 0, np.nan, depth_img)
    )

    # get camera parameters
    cam_intrinsics = cam.intrinsics
    cam_pos, cam_quat = cam.get_cart_pos_quat()

    # save data
    return (
        {
            "rgb_img": rgb_img,
            "depth_img": depth_img,
            "seg_img": seg_img,
            "seg_img_all": seg_img_orig,
            "point_cloud_points": point_cloud_points,
            "point_cloud_colors": point_cloud_colors,
            "point_cloud_seg_points": point_cloud_seg_points,
            "point_cloud_seg_colors": point_cloud_seg_colors,
            "cam_pos": cam_pos,
            "cam_quat": cam_quat,
            "cam_intrinsics": cam_intrinsics,
        },
        scene,
        agent,
    )


def execute_grasping_sequence(
    agent: RobotBase,
    grasp_pos: np.array,
    grasp_quat: np.array,
    home_pos: np.array = (0.5, 0, 0.5),
    home_quat: np.array = (0, 1, 0, 0),
    drop_pos: np.array = (0, 0.5, 0.5),
    drop_quat: np.array = (0, 1, 0, 0),
    hover_offset: float = 0.05,
    movement_time: float = 4,
    grasp_movement_time: float = 2,
    wait_time: float = 1,
):
    """Execute a grasping sequence with the given agent.
    The sequence is:
    - go to home position
    - go to hover position
    - open gripper
    - go to grasp position
    - close gripper
    - go to hover position
    - go to home position
    - go to drop position
    - open gripper
    - close gripper

    Args:
        agent (RobotBase): the agent object
        grasp_pos (np.array): position of the grasp
        grasp_quat (np.array): quaternion of the grasp
        home_pos (np.array, optional): position of the home position. Defaults to (0.5, 0, 0.5).
        home_quat (np.array, optional): quaternion of the home position. Defaults to (0, 1, 0, 0).
        drop_pos (np.array, optional): position of the drop position. Defaults to (0, 0.5, 0.5).
        drop_quat (np.array, optional): quaternion of the drop position. Defaults to (0, 1, 0, 0).
        hover_offset (float, optional): offset along the grasp axis. Defaults to 0.05.
        movement_time (float, optional): duration of the movement. Defaults to 4.
        grasp_movement_time (float, optional): duration of the movement for grasping. Defaults to 2.
        wait_time (float, optional): wait time after the robot has moved. Defaults to 1.
    """
    # hover_offset = np.array(hover_offset)
    hover_offset = np.array([0, 0, hover_offset])  # TODO offste along grasp axis
    hover_positon = grasp_pos + hover_offset

    logging.info(f"Going to home position {home_pos}")
    agent.gotoCartPositionAndQuat(home_pos, home_quat, duration=movement_time)
    agent.wait(wait_time)

    logging.info(f"Going to hover_ position {hover_positon}")
    agent.gotoCartPositionAndQuat(hover_positon, grasp_quat, duration=movement_time)
    agent.wait(wait_time)

    logging.info("Opening gripper")
    agent.open_fingers()
    agent.wait(wait_time)

    logging.info(f"Going to grasp position {grasp_pos}")
    agent.gotoCartPositionAndQuat(grasp_pos, grasp_quat, duration=grasp_movement_time)
    agent.wait(wait_time)

    logging.info("Closing gripper")
    agent.close_fingers()
    agent.wait(wait_time)

    logging.info(f"Going to hover position {hover_positon}")
    agent.gotoCartPositionAndQuat(
        hover_positon, grasp_quat, duration=grasp_movement_time
    )
    agent.wait(wait_time)

    logging.info(f"Going to home position {home_pos}")
    agent.gotoCartPositionAndQuat(home_pos, home_quat, duration=movement_time)
    agent.wait(wait_time)

    logging.info(f"Going to drop position {drop_pos}")
    agent.gotoCartPositionAndQuat(drop_pos, drop_quat, duration=movement_time)
    agent.wait(wait_time)

    logging.info("Opening gripper")
    agent.open_fingers()
    agent.wait(wait_time)

    logging.info("Closing gripper")
    agent.close_fingers()
    agent.wait(wait_time)
