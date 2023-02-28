from typing import Tuple, List
from pathlib import Path

import trimesh
import numpy as np


class LegacyYCBLoader:
    """Utility class to load and convert and get information about objects of the
    ycb dataset.
    """

    def __init__(
        self,
        ycb_base_folder: Path,
        pos: Tuple[float, float] = (0.5, 0.3, 0),
        quat: Tuple[float, float, float, float] = (0, 1, 0, 0),
        factory_string: str = "mj_beta",
        adjust_object_position: bool = True,
        broken_object_names: List[str] = None,
    ):
        """Instantiate a ycb object loader.

        Args:
            ycb_base_folder (str): The path to the ycb dataset (".../models/ycb/")
        """
        self.ycb_base_folder = ycb_base_folder

        self.object_folders = [
            p.parent for p in Path(self.ycb_base_folder).glob("*/textured")
        ]
        assert (
            len(self.object_folders) > 0
        ), f"No folders found in ycb_base_folder path. ({ycb_base_folder})."
        self.object_folders = sorted(self.object_folders, key=lambda x: x.name)

        self.pos = pos
        self.quat = quat
        self.factory_string = factory_string
        self.adjust_object_position = adjust_object_position
        self.broken_object_names = broken_object_names or [
            "001_chips_can",
            "022_windex_bottle",
            "028_skillet_lid",
            "038_padlock",
            "041_small_marker",
            "049_small_clamp",
            "072-h_toy_airplane",
            "072-k_toy_airplane",
            "073-h_lego_duplo",
            "073-i_lego_duplo",
            "073-k_lego_duplo",
            "073-l_lego_duplo",
            "073-m_lego_duplo",
            "076_timer",
        ]

    def get_obj_index(self, obj_name: str) -> int:
        for i, p in enumerate(self.object_folders):
            if obj_name in (p.name, p.name[:4]):
                return i

        raise ValueError(f"Object {obj_name} not found in ycb dataset.")

    def get_obj_name(self, index: int) -> str:
        return self.object_folders[index].name

    def get_xml_path(self, index: int):
        obj_path = self.object_folders[index]
        return obj_path / "textured" / f"{obj_path.name[4:]}.xml"

    def get_urdf_path(self, index: int):
        obj_path = self.object_folders[index]
        return obj_path / "textured" / f"{obj_path.name[4:]}.urdf"

    def get_obj_path(self, index):
        obj_path = self.object_folders[index]
        return obj_path / "textured" / "textured_vhacd.obj"

    def get_obj_bounds(self, index):
        obj_file = self.get_obj_path(index=index)
        obj_mesh = trimesh.load(obj_file, force="mesh")
        bounds = obj_mesh.bounds
        return bounds

    def is_broken(self, index: int) -> bool:
        return self.get_obj_name(index) in self.broken_object_names

    def get_ycb_object(
        self,
        index: int,
        pos: Tuple[int, int] = None,
        quat: Tuple[int, int] = None,
        factory_string: str = None,
        adjust_object_position: bool = None,
    ):
        pos = pos or self.pos
        quat = quat or self.quat
        factory_string = factory_string or self.factory_string
        adjust_object_position = adjust_object_position or self.adjust_object_position

        if adjust_object_position:
            pos = np.array(pos)
            bounds = self.get_obj_bounds(index)
            to_middle = (bounds[1] - bounds[0]) / 2 + bounds[0]
            pos[:2] -= to_middle[:2]
            pos[2] -= bounds[0][2]

        name = self.object_folders[index].name[4:]
        obj = None

        if factory_string == "mujoco":
            from alr_sim.sims.mujoco.mj_utils.mujoco_scene_object import MujocoObject

            obj = MujocoObject(
                object_name=name,
                pos=pos,
                quat=quat,
                obj_path=self.get_xml_path(index),
            )

        elif factory_string == "mj_beta":
            from alr_sim.sims.mj_beta.mj_utils.mj_scene_object import CustomMujocoObject

            obj = CustomMujocoObject(
                object_name=name,
                pos=pos,
                quat=quat,
                root="/",
                object_dir_path=str(self.object_folders[index] / "textured"),
            )

        else:
            raise ValueError(
                f"Factory string {factory_string} not recognized. "
                "Please use 'mujoco' or 'mj_beta'."
            )

        return (obj, name)

    def __getitem__(self, index: int):
        return self.get_ycb_object(index)

    def __len__(self):
        return len(self.object_folders)
