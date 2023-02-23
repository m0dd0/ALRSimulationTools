from typing import List, Tuple

import numpy as np

from alr_sim.sims.SimFactory import SimRepository
from alr_sim.sims.universal_sim.PrimitiveObjects import Box
from alr_sim.core import Scene
from alr_sim.sims.mj_beta import MjCamera


def reset_scene(factory_string: str, scene: Scene, agent):
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


def create_sample_data(
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
):
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
        "seg_img_all": seg_img_orig,
        "point_cloud_seg": point_cloud_seg,
        "point_cloud": point_cloud,
        "cam_pos": cam_pos,
        "cam_quat": cam_quat,
        "cam_intrinsics": cam_intrinsics,
    }


def execute_grasping_sequence(
    agent,
    home_pos: np.array,
    grasp_pos: np.array,
    drop_pos: np.array,
    grasp_quat: np.array,
    hover_offset: np.array = (0, 0, 0.05),
    movement_time: float = 4,
    grasp_movement_time: float = 2,
    wait_time: float = 1,
    logging: bool = False,
):
    hover_offset = np.array(hover_offset)

    # assumes that the agent is in a safe position similar to home_pos

    # go to position above grasp position
    if logging:
        print(f"Going to position above grasp position {grasp_pos + hover_offset}")
    agent.gotoCartPositionAndQuat(
        grasp_pos + hover_offset, grasp_quat, duration=movement_time
    )
    agent.wait(wait_time)

    # open gripper
    if logging:
        print("Opening gripper")
    agent.open_fingers()
    agent.wait(wait_time)

    # go to grasp position
    if logging:
        print(f"Going to grasp position {grasp_pos}")
    agent.gotoCartPositionAndQuat(grasp_pos, grasp_quat, duration=grasp_movement_time)
    agent.wait(wait_time)

    # close gripper
    if logging:
        print("Closing gripper")
    agent.close_fingers()
    agent.wait(wait_time)

    # go to position above grasp position
    if logging:
        print(f"Going to position above grasp position {grasp_pos + hover_offset}")
    agent.gotoCartPositionAndQuat(
        grasp_pos + hover_offset, grasp_quat, duration=grasp_movement_time
    )
    agent.wait(wait_time)

    # go to home position
    if home_pos is not None:
        if logging:
            print(f"Going to home position {home_pos}")
        agent.gotoCartPositionAndQuat(home_pos, grasp_quat, duration=movement_time)
        agent.wait(wait_time)

    # go to drop position
    if logging:
        print(f"Going to drop position {drop_pos}")
    agent.gotoCartPositionAndQuat(drop_pos, grasp_quat, duration=movement_time)
    agent.wait(wait_time)

    # open gripper
    if logging:
        print("Opening gripper")
    agent.open_fingers()
    agent.wait(wait_time)

    # go to home position
    if home_pos is not None:
        if logging:
            print(f"Going to home position {home_pos}")
        agent.gotoCartPositionAndQuat(home_pos, grasp_quat, duration=movement_time)
        agent.wait(wait_time)

    # close gripper
    if logging:
        print("Closing gripper")
    agent.close_fingers()
    agent.wait(wait_time)
