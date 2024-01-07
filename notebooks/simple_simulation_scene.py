from alr_sim.sims.SimFactory import SimRepository
from alr_sim.sims.universal_sim.PrimitiveObjects import Box
from alr_sim.core import Scene
from alr_sim.sims.mj_beta import MjCamera

from alr_sim_tools.scene_utils import reset_scene

import glfw


FACTORY_SRTING = "mj_beta"
RENDER_MODE = Scene.RenderMode.HUMAN
CAMERA_POSITION = (0.5, 0.0, 1.0)
CAMERA_QUAT = (-0.70710678, 0, 0, 0.70710678)
MOVEMENT_DURATIONS = [2]
WAIT_DURATIONS_BETWEEN_MOVEMENTS = [0.5]
MOVEMENT_TARGET_POSITIONS = [(0.0, 0.5, -0.01)]
MOVEMENT_TARGET_QUATS = [(0, 1, 0, 0)]

box = Box(
    name="box1",
    init_pos=[0.5, -0.2, 0.0],
    init_quat=[0, 1, 0, 0],
    rgba=[0.1, 0.25, 0.3, 1],
)

sim_factory = SimRepository.get_factory(FACTORY_SRTING)
scene = sim_factory.create_scene(object_list=[box], dt=0.0002, render=RENDER_MODE)
agent = sim_factory.create_robot(scene, dt=0.0002)

# cam = MujocoCamera("my_cam", init_pos=CAMERA_POSITION, init_quat=CAMERA_QUAT)
cam = MjCamera("my_cam", init_pos=CAMERA_POSITION, init_quat=CAMERA_QUAT)
cam.set_cam_params(height=480, width=640)
scene.add_object(cam)

scene.start()
scene.start_logging()

for target_pos, target_quat, movement_duration, wait_duration in zip(
    MOVEMENT_TARGET_POSITIONS,
    MOVEMENT_TARGET_QUATS,
    MOVEMENT_DURATIONS,
    WAIT_DURATIONS_BETWEEN_MOVEMENTS,
):
    agent.gotoCartPositionAndQuat(target_pos, target_quat, duration=movement_duration)
    agent.wait(wait_duration)

scene.stop_logging()
reset_scene(FACTORY_SRTING, scene, agent)
# glfw.terminate()
print("Done")
