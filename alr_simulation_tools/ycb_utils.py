from typing import Tuple, List
from pathlib import Path
import os

import numpy as np
import yaml
import trimesh

from alr_sim.utils.geometric_transformation import posRotMat2TFMat, quat2mat
from alr_sim.sims.mj_beta.mj_utils.mj_scene_object import YCBMujocoObject


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

    def get_obj_folder(self, obj_id: str) -> Path:
        """Get the folder of the object, based on obj_id and ycb_base_folder.

        Args:
            obj_id (str): Object ID, same as folder name (e.g. "003_cracker_box")

        Returns:
            str: Path to the object folder
        """
        return self.ycb_base_folder / obj_id

    def get_orig_file_path(self, obj_id: str) -> Path:
        """Get the path to the original file of the object. Used for visualization, without any decompositions.

        Args:
            obj_id (str): Object ID, same as folder name (e.g. "003_cracker_box")

        Returns:
            Path: Path to the original file
        """
        obj_folder = self.get_obj_folder(obj_id)
        info_file = obj_folder / "info.yml"

        if not os.path.isfile(info_file):
            raise FileNotFoundError(info_file)

        with open(info_file, "r") as f:
            info_dict = yaml.safe_load(f)

        return obj_folder / info_dict["original_file"]

    def get_obj_bounds(self, obj_id: str) -> Tuple[List[float], List[float]]:
        """Get the bounds of the object

        Args:
            obj_id (str): Object ID, same as folder name (e.g. "003_cracker_box")

        Returns:
            Tuple[List[float], List[float]]: [[x_max, y_max, z_max], [x_min, y_min, z_min]]
        """
        obj_mesh = trimesh.load(self.get_orig_file_path(obj_id), force="mesh")
        bounds = obj_mesh.bounds
        return bounds

    def get_obj_rotated_bounds(
        self, obj_id: str, obj_quat: List[float]
    ) -> Tuple[List[float], List[float]]:
        """Get the bounds of the object, rotated by the given quaternion

        Args:
            obj_id (str): Object ID, same as folder name (e.g. "003_cracker_box")
            obj_quat (List[float]): Rotation quaternion as [w, x, y, z].

        Returns:
            Tuple[List[float], List[float]]: [[x_max, y_max, z_max], [x_min, y_min, z_min]]
        """
        tf_mat = posRotMat2TFMat(np.zeros(3), quat2mat(obj_quat))

        obj_mesh = trimesh.load(self.get_orig_file_path(obj_id), force="mesh")
        obj_mesh.apply_transform(tf_mat)

        bounds = obj_mesh.bounds
        return bounds

    def get_ycb_object(
        self,
        pos,
        quat,
        obj_id: str = None,
        name: str = None,
        factory_string: str = None,
    ) -> YCBMujocoObject:
        """Get the object, based on the factory string and the object id.
        Also applies the given position and quaternion.

        Args:
            pos (List[float]): Position [x, y, z] of the object in the scene
            quat (List[float]): Rotation quaternion as [w, x, y, z].
            obj_id (str): Object ID, same as folder name (e.g. "003_cracker_box")
            name (str): Name of the object in the scene
            factory_string (str): The factory referencing the used simulation environment. If None, self.factory_string is used.

        Returns:
            YCBMujocoObject: The object
        """
        _factory_string = factory_string or self.factory_string

        if _factory_string == "mj_beta":
            return YCBMujocoObject(
                self.ycb_base_folder,
                obj_id,
                name,
                pos,
                quat,
            )
        elif _factory_string == "mujoco":
            raise NotImplementedError
        elif _factory_string == "pybullet":
            raise NotImplementedError
