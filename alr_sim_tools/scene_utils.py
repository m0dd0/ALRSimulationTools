from typing import List, Tuple
import logging
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
from scipy.spatial.transform import Rotation

from alr_sim.sims.SimFactory import SimRepository
from alr_sim.sims.universal_sim.PrimitiveObjects import Box
from alr_sim.core import Scene, RobotBase
from alr_sim.sims.mj_beta import MjCamera

from alr_sim_tools.typing_utils import NpArray

TABLE_TOP_Z_OFFSET = -0.02
JOINT_CONFIGURATION_LOOKUP_FILE = (
    Path(__file__).parent / "joint_configuration_lookup.json"
)


@dataclass
class CameraData:
    rgb_img: NpArray["H,W,3", np.uint8]
    depth_img: NpArray["H,W", np.float32]
    seg_img: NpArray["H,W", np.int64]
    seg_img_all: NpArray["H,W", np.int32]
    point_cloud_points: NpArray["N,3", np.float64]
    point_cloud_colors: NpArray["N,3", np.float64]
    point_cloud_seg_points: NpArray["M,3", np.float64]
    point_cloud_seg_colors: NpArray["M,3", np.float64]
    cam_pos: NpArray["3", np.float64]
    cam_quat: NpArray["4", np.float64]
    cam_intrinsics: NpArray["3, 3", np.float64]


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
    cam_pos: Tuple[float, float, float] = (0.5, 0.0, 1.0),
    cam_quat: Tuple[float, float, float, float] = (-0.70710678, 0, 0, 0.70710678),
    cam_height: int = 480,
    cam_width: int = 640,
    robot_pos: Tuple[float, float, float] = (0.0, 0.5, 0.2),
    robot_quat: Tuple[float, float, float, float] = (0, 1, 0, 0),
    object_list: List = None,
    target_obj_name: str = None,
    render_mode: Scene.RenderMode = Scene.RenderMode.BLIND,
    wait_time: float = 0.01,
    move_duration: float = 4,
) -> Tuple[CameraData, Scene, RobotBase]:
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
    beam_to_pos_quat(agent, robot_pos, robot_quat, move_duration)
    agent.wait(wait_time)

    # get camera data
    rgb_img, depth_img = cam.get_image()
    point_cloud_points, point_cloud_colors = cam.calc_point_cloud()
    target_obj_id = scene.get_obj_seg_id(obj_name=target_obj_name)
    seg_img_orig = cam.get_segmentation(depth=False)

    seg_img = np.where(seg_img_orig == target_obj_id, True, False)

    point_cloud_seg_points, point_cloud_seg_colors = cam.calc_point_cloud_from_images(
        rgb_img, np.where(depth_img * seg_img == 0, np.nan, depth_img)
    )

    # get camera parameters
    cam_intrinsics = cam.intrinsics
    cam_pos, cam_quat = cam.get_cart_pos_quat()

    # save data
    return (
        CameraData(
            rgb_img=rgb_img,
            depth_img=depth_img,
            seg_img=seg_img,
            seg_img_all=seg_img_orig,
            point_cloud_points=point_cloud_points,
            point_cloud_colors=point_cloud_colors,
            point_cloud_seg_points=point_cloud_seg_points,
            point_cloud_seg_colors=point_cloud_seg_colors,
            cam_pos=cam_pos,
            cam_quat=cam_quat,
            cam_intrinsics=cam_intrinsics,
        ),
        scene,
        agent,
    )


def execute_grasping_sequence(
    agent: RobotBase,
    grasp_pos: Tuple[float, float, float],
    grasp_quat: Tuple[float, float, float, float],
    drop_pos: Tuple[float, float, float] = (0, 0.5, 0.5),
    drop_quat: Tuple[float, float, float, float] = (0, 1, 0, 0),
    hover_offset: float = 0.05,
    movement_time: float = 4,
    grasp_movement_time: float = 2,
    wait_time: float = 0.5,
):
    """Execute a grasping sequence with the given agent.
    The sequence is:
    - beam to hover position
    - open gripper
    - go to grasp position
    - close gripper
    - go to hover position
    - go to drop position
    - open gripper

    Args:
        agent (RobotBase): the agent object
        grasp_pos (Tuple[float])): position of the grasp
        grasp_quat (Tuple[float])): quaternion of the grasp
        drop_pos (Tuple[float]), optional): position of the drop position. Defaults to (0, 0.5, 0.5).
        drop_quat (Tuple[float]), optional): quaternion of the drop position. Defaults to (0, 1, 0, 0).
        hover_offset (float, optional): offset along the grasp axis. Defaults to 0.05.
        movement_time (float, optional): duration of the movement. Defaults to 4.
        grasp_movement_time (float, optional): duration of the movement for grasping. Defaults to 2.
        wait_time (float, optional): wait time after the robot has moved. Defaults to 1.
    """
    grasp_axis = Rotation.from_quat(grasp_quat[[1, 2, 3, 0]]).as_matrix()[:, 2]
    grasp_axis = grasp_axis / np.linalg.norm(grasp_axis)
    hover_positon = np.array(grasp_pos) + grasp_axis * hover_offset

    logging.info(f"Beam to hover_ position {hover_positon}")
    beam_to_pos_quat(agent, hover_positon, grasp_quat, duration=movement_time)
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

    logging.info(f"Going to drop position {drop_pos}")
    agent.gotoCartPositionAndQuat(drop_pos, drop_quat, duration=movement_time)
    agent.wait(wait_time)

    logging.info("Opening gripper")
    agent.open_fingers()
    agent.wait(wait_time)


def beam_to_pos_quat(
    agent: RobotBase,
    pos: Tuple[float, float, float],
    quat: Tuple[float, float, float, float],
    duration: float,
):
    """
    If a joint configuration is known for the given position and quaternion, the agent beams to this joint configuration.
    Otherwise the agent is moved normally to the given position and quaternion and the joint configuration is saved for future use.

    Args:
        agent (RobotBase): the agent object
        pos (Tuple[float]): position of the desired position
        quat (Tuple[float]): quaternion of the desired orientation
        durarion (float): duration of the movement in case the joint configuration is not known
    """

    with open(JOINT_CONFIGURATION_LOOKUP_FILE, "r") as f:
        joint_configuration_lookup = json.load(f)

    for (
        saved_pos,
        saved_quat,
        saved_joint_configuration,
    ) in joint_configuration_lookup:
        if np.allclose(saved_pos, pos) and np.allclose(saved_quat, quat):
            agent.beam_to_joint_pos(saved_joint_configuration)
            return

    logging.warning(
        f"Joint configuration for position {pos} and quaternion {quat} not found. Moving to position and saving joint configuration."
    )

    agent.gotoCartPositionAndQuat(pos, quat, duration=duration)
    joint_configuration = tuple(agent.current_j_pos)

    joint_configuration_lookup.append((pos, quat, joint_configuration))

    with open(JOINT_CONFIGURATION_LOOKUP_FILE, "w") as f:
        json.dump(joint_configuration_lookup, f)
