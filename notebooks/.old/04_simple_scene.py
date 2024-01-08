from alr_sim.sims.SimFactory import SimRepository
from alr_sim.sims.universal_sim.PrimitiveObjects import Box
from alr_sim.core import Scene
from alr_sim.sims.mj_beta import MjCamera

def main():
    box1 = Box(
        name="box1",
        init_pos=[0.5, -0.2, 0.0],
        init_quat=[0, 1, 0, 0],
        rgba=[0.1, 0.25, 0.3, 1],
    )
    object_list = [box1]

    # Generate the chosen Scene and Agent
    sim_factory = SimRepository.get_factory("mj_beta")
    scene = sim_factory.create_scene(
        object_list=object_list, dt=0.0002, render=Scene.RenderMode.BLIND
    )
    agent = sim_factory.create_robot(scene, dt=0.0002)

    cam = MjCamera(
        "my_cam",
        init_pos=(0.5, 0, 1),
        init_quat=(0,1,0,0)
    )
    scene.add_object(cam)

    scene.start()

    agent.wait(1)

    rgb_img = cam.get_image(depth=False)


if __name__=="__main__":
    main()