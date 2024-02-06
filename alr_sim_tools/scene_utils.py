from typing import List, Tuple
import logging
from dataclasses import dataclass

import numpy as np

from alr_sim.sims.SimFactory import SimRepository
from alr_sim.sims.universal_sim.PrimitiveObjects import Box
from alr_sim.core import Scene, RobotBase
from alr_sim.sims.mj_beta import MjCamera

from alr_sim_tools.typing_utils import NpArray

# a mapping of workd coordinates to joint coordinates
# as the SimulationFramework does not implement the beam_to_cart_pos method and only a beam_to_joint_pos method
# we need to convert the world coordinates to joint coordinates
# The inverse kinematics calculation of the SimulationFramework is not exposed to the user
# so we can not calculate the joint coordinates directly
# However, the data below has been collected by manually moving the robot to the desired position
# and reading the joint coordinates from the simulation
POS_QUAT_2_JOINT_COORDS = {
    ((0.5, 0.0, 0.1), (0, 1, 0, 0)): (
        -0.0001,
        0.26738,
        -0.00014,
        -2.39607,
        -0.00011,
        2.66243,
        0.78532,
    ),
    ((0.5, 0.0, 0.15), (0, 1, 0, 0)): (
        1e-05,
        0.15345,
        -1e-05,
        -2.39756,
        -2e-05,
        2.55245,
        0.78545,
    ),
    ((0.5, 0.0, 0.2), (0, 1, 0, 0)): (
        1e-05,
        0.04767,
        1e-05,
        -2.37694,
        3e-05,
        2.42672,
        0.78542,
    ),
    ((0.5, 0.0, 0.25), (0, 1, 0, 0)): (
        1e-05,
        -0.04058,
        0.0,
        -2.33564,
        5e-05,
        2.29731,
        0.7854,
    ),
    ((0.5, 0.0, 0.3), (0, 1, 0, 0)): (
        0.0,
        -0.10949,
        -0.0,
        -2.27468,
        6e-05,
        2.16741,
        0.7854,
    ),
    ((0.5, 0.0, 0.35), (0, 1, 0, 0)): (
        -0.0,
        -0.1585,
        -1e-05,
        -2.19482,
        7e-05,
        2.03845,
        0.7854,
    ),
    ((0.5, 0.0, 0.4), (0, 1, 0, 0)): (
        -0.0,
        -0.1877,
        -1e-05,
        -2.09644,
        8e-05,
        1.91072,
        0.7854,
    ),
    ((0.5, 0.0, 0.45), (0, 1, 0, 0)): (
        -1e-05,
        -0.19738,
        -2e-05,
        -1.97912,
        8e-05,
        1.78354,
        0.7854,
    ),
    ((0.5, 0.0, 0.5), (0, 1, 0, 0)): (
        -1e-05,
        -0.18743,
        -2e-05,
        -1.84116,
        8e-05,
        1.65532,
        0.7854,
    ),
    ((0.5, 0.0, 0.55), (0, 1, 0, 0)): (
        -2e-05,
        -0.15664,
        -2e-05,
        -1.67855,
        8e-05,
        1.52327,
        0.7854,
    ),
    ((0.5, 0.0, 0.6), (0, 1, 0, 0)): (
        -2e-05,
        -0.10119,
        -2e-05,
        -1.48246,
        7e-05,
        1.38239,
        0.78541,
    ),
    ((0.0, 0.5, 0.1), (0, 1, 0, 0)): (
        1.08803,
        0.30093,
        0.4463,
        -2.38944,
        -0.2721,
        2.64706,
        2.55064,
    ),
    ((0.0, 0.5, 0.15), (0, 1, 0, 0)): (
        0.97897,
        0.18506,
        0.57862,
        -2.39467,
        -0.18042,
        2.54396,
        2.48678,
    ),
    ((0.0, 0.5, 0.2), (0, 1, 0, 0)): (
        0.87854,
        0.06244,
        0.69158,
        -2.37661,
        -0.06162,
        2.42568,
        2.4012,
    ),
    ((0.0, 0.5, 0.25), (0, 1, 0, 0)): (
        0.80679,
        -0.05558,
        0.76152,
        -2.33542,
        0.05036,
        2.29661,
        2.31926,
    ),
    ((0.0, 0.5, 0.3), (0, 1, 0, 0)): (
        0.75482,
        -0.15696,
        0.79618,
        -2.27291,
        0.13406,
        2.16064,
        2.2552,
    ),
    ((0.0, 0.5, 0.35), (0, 1, 0, 0)): (
        0.70971,
        -0.23439,
        0.81244,
        -2.19119,
        0.18799,
        2.02354,
        2.21079,
    ),
    ((0.0, 0.5, 0.4), (0, 1, 0, 0)): (
        0.66535,
        -0.28557,
        0.82181,
        -2.09133,
        0.21849,
        1.88887,
        2.18294,
    ),
    ((0.0, 0.5, 0.45), (0, 1, 0, 0)): (
        0.62226,
        -0.30905,
        0.82974,
        -1.97321,
        0.2301,
        1.75794,
        2.17018,
    ),
    ((0.0, 0.5, 0.5), (0, 1, 0, 0)): (
        0.5863,
        -0.30165,
        0.83854,
        -1.83541,
        0.22301,
        1.63062,
        2.17435,
    ),
    ((0.0, 0.5, 0.55), (0, 1, 0, 0)): (
        0.57222,
        -0.25624,
        0.84689,
        -1.6742,
        0.1914,
        1.50526,
        2.20154,
    ),
    ((0.0, 0.5, 0.6), (0, 1, 0, 0)): (
        0.6178,
        -0.16087,
        0.84007,
        -1.48072,
        0.12192,
        1.37567,
        2.26157,
    ),
}


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
    robot_pos: Tuple[float, float, float] = (0.0, 0.5, -0.01),
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
    try:
        beam_to_near_pos_quat(agent, robot_pos, robot_quat)
    except ValueError:
        agent.gotoCartPositionAndQuat(robot_pos, robot_quat, duration=move_duration)

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
    home_pos: Tuple[float, float, float] = (0.5, 0, 0.5),
    home_quat: Tuple[float, float, float, float] = (0, 1, 0, 0),
    drop_pos: Tuple[float, float, float] = (0, 0.5, 0.5),
    drop_quat: Tuple[float, float, float, float] = (0, 1, 0, 0),
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
        grasp_pos (Tuple[float])): position of the grasp
        grasp_quat (Tuple[float])): quaternion of the grasp
        home_pos (Tuple[float]), optional): position of the home position. Defaults to (0.5, 0, 0.5).
        home_quat (Tuple[float]), optional): quaternion of the home position. Defaults to (0, 1, 0, 0).
        drop_pos (Tuple[float]), optional): position of the drop position. Defaults to (0, 0.5, 0.5).
        drop_quat (Tuple[float]), optional): quaternion of the drop position. Defaults to (0, 1, 0, 0).
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


def beam_to_near_pos_quat(
    agent: RobotBase,
    pos: Tuple[float, float, float],
    quat: Tuple[float, float, float, float],
    max_distance: float = 0.1,
):
    """
    Beams the robot to the closest possible position for which joint coordinates are known.

    Args:
        agent (RobotBase): the agent object
        pos (Tuple[float]): position of the desired position
        quat (Tuple[float]): quaternion of the desired orientation
        max_distance (float, optional): maximum distance to the desired position. Defaults to 0.1.
            If no joint coordinates are known for a position within this distance, a ValueError is raised.
    """
    min_dist = float("inf")
    closest_joint_coords = None
    for (pos_, quat_), joint_coords in POS_QUAT_2_JOINT_COORDS.items():
        dist = np.linalg.norm(np.array(pos) - np.array(pos_))
        if dist < min_dist and quat == quat_:
            min_dist = dist
            closest_joint_coords = joint_coords

    if min_dist is None or min_dist > max_distance:
        raise ValueError(
            f"No appropriate joint coordinates known for position {pos} and quaternion {quat}"
        )

    agent.beam_to_cart_pos_and_quat(closest_joint_coords)
