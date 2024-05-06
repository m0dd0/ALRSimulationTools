from typing import Tuple, List
from pathlib import Path
import os

import numpy as np
import yaml
import trimesh
from scipy.spatial.transform import Rotation

from alr_sim.sims.mj_beta.mj_utils.mj_scene_object import YCBMujocoObject

from alr_sim_tools.scene_utils import TABLE_TOP_Z_OFFSET


class YCBLoader:
    """Class that helps load YCB objects"""

    def __init__(self, ycb_base_folder: Path, factory_string: str = "mj_beta"):
        """Constructor

        Args:
            ycb_base_folder (Path): The folder where the YCB objects are stored
                (folder with the obj_ids as subfolders, ".../SF-ObectDataset/YCB/")
            factory_string (str): The factory referencing the used simulation environment
        """
        self.ycb_base_folder = ycb_base_folder
        self.factory_string = factory_string

    def get_obj_folder(self, object_id: str) -> Path:
        """Get the folder of the object, based on obj_id and ycb_base_folder.

        Args:
            obj_id (str): Object ID, same as folder name (e.g. "003_cracker_box")

        Returns:
            str: Path to the object folder
        """
        return self.ycb_base_folder / object_id

    def get_orig_file_path(self, object_id: str) -> Path:
        """Get the path to the original file of the object. Used for visualization, without any decompositions.

        Args:
            obj_id (str): Object ID, same as folder name (e.g. "003_cracker_box")

        Returns:
            Path: Path to the original file
        """
        obj_folder = self.get_obj_folder(object_id)
        info_file = obj_folder / "info.yml"

        if not os.path.isfile(info_file):
            raise FileNotFoundError(info_file)

        with open(info_file, "r") as f:
            info_dict = yaml.safe_load(f)

        return obj_folder / info_dict["original_file"]

    def get_obj_bounds(
        self, obj_id: str, object_quaternion: List[float] = None
    ) -> Tuple[List[float], List[float]]:
        """Get the bounds of the object. Optionally rotated by the given quaternion.

        Args:
            obj_id (str): Object ID, same as folder name (e.g. "003_cracker_box")
            obj_quat (List[float]): Rotation quaternion as [w, x, y, z].

        Returns:
            Tuple[List[float], List[float]]: [[x_max, y_max, z_max], [x_min, y_min, z_min]]
        """
        obj_mesh = trimesh.load(self.get_orig_file_path(obj_id), force="mesh")

        if object_quaternion is not None:
            object_quaternion = np.array(object_quaternion)
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = Rotation.from_quat(
                object_quaternion[[1, 2, 3, 0]]
            ).as_matrix()
            obj_mesh.apply_transform(transformation_matrix)

        bounds = obj_mesh.bounds
        return bounds

    def get_ycb_object(
        self,
        pos,
        quat,
        object_id: str = None,
        name: str = None,
        factory_string: str = None,
        grounded: bool = False,
    ) -> YCBMujocoObject:
        """Get the object, based on the factory string and the object id.
        Also applies the given position and quaternion.

        Args:
            pos (List[float]): Position [x, y, z] of the object in the scene
            quat (List[float]): Rotation quaternion as [w, x, y, z].
            obj_id (str): Object ID, same as folder name (e.g. "003_cracker_box")
            name (str): Name of the object in the scene
            factory_string (str): The factory referencing the used simulation environment. If None, self.factory_string is used.
            grounded (bool): If the z-coordinate of the object position should be adapted so that the object is grounded.

        Returns:
            YCBMujocoObject: The object
        """
        _factory_string = factory_string or self.factory_string

        pos = np.array(pos)
        if grounded:
            bounds = self.get_obj_bounds(object_id, quat)
            height = bounds[1][2] - bounds[0][2]
            # it seems like the origin of ycb objects is at the top of the object (and not the center)
            pos[2] = height + TABLE_TOP_Z_OFFSET

        if _factory_string == "mj_beta":
            return YCBMujocoObject(
                self.ycb_base_folder,
                object_id,
                name,
                pos,
                quat,
            )
        elif _factory_string == "mujoco":
            raise NotImplementedError
        elif _factory_string == "pybullet":
            raise NotImplementedError
