from typing import List

import glob
import os

import trimesh
import numpy as np

from alr_sim.utils.geometric_transformation import quat2euler


class YCBLoader:
    """Utility class to load and convert and get information about objects of the
    ycb dataset.
    """

    def __init__(self, ycb_base_folder: str, factory_string: str = None):
        """Instantiate a ycb object loader.

        Args:
            ycb_base_folder (str): The path to the ycb dataset (".../models/ycb/")
            factory_string (str, optional): The simulator for which objects are build
            from the data. Defaults to None.
        """
        folders = glob.glob(os.path.join(ycb_base_folder, "*/textured"))
        assert (
            len(folders) > 0
        ), f"No folders found in ycb_base_folder path. ({ycb_base_folder})."

        self.obj_ids = [folder.split("/")[-2] for folder in folders]
        self.names = [obj_id[4:] for obj_id in self.obj_ids]
        self.ycb_base_folder = ycb_base_folder
        self.factory_string = factory_string

    def _get_obj_index(self, obj_id: str = None, name: str = None):
        if obj_id is None and name is None:
            raise ValueError("Requiring either obj_id or name")

        index = None
        if obj_id is not None:
            index = self.obj_ids.index(obj_id)
        if name is not None:
            index = self.names.index(name)

        return index

    def get_xml_path(self, obj_id: str = None, name: str = None, index: int = None):
        if index is None:
            index = self._get_obj_index(obj_id=obj_id, name=name)

        xml_folder = self.get_xml_folder(index=index)
        return os.path.join(xml_folder, f"{self.names[index]}.xml")

    def get_urdf_path(self, obj_id: str = None, name: str = None, index: int = None):
        if index is None:
            index = self._get_obj_index(obj_id=obj_id, name=name)

        xml_folder = self.get_xml_folder(index=index)
        return os.path.join(xml_folder, f"{self.names[index]}.urdf")

    def get_xml_folder(self, obj_id: str = None, name: str = None, index: int = None):
        if index is None:
            index = self._get_obj_index(obj_id=obj_id, name=name)

        return os.path.join(self.ycb_base_folder, self.obj_ids[index], "textured")

    def get_obj_path(self, obj_id: str = None, name: str = None, index: int = None):
        if index is None:
            index = self._get_obj_index(obj_id=obj_id, name=name)

        xml_folder = self.get_xml_folder(index=index)
        return os.path.join(xml_folder, "textured_vhacd.obj")

    def get_obj_bounds(self, obj_id: str = None, name: str = None, index: int = None):

        if index is None:
            index = self._get_obj_index(obj_id=obj_id, name=name)

        obj_file = self.get_obj_path(index=index)
        obj_mesh = trimesh.load(obj_file, force="mesh")
        bounds = obj_mesh.bounds
        return bounds

    def get_ycb_object(
        self,
        pos: List,
        quat: List,
        obj_id: str = None,
        name: str = None,
        index: int = None,
        factory_string: str = None,
        adjust_object_position: bool = False,
    ):
        """

        Args:
            pos (List): _description_
            quat (List): _description_
            obj_id (str, optional): _description_. Defaults to None.
            name (str, optional): _description_. Defaults to None.
            index (int, optional): _description_. Defaults to None.
            factory_string (str, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        _factory_string = factory_string or self.factory_string

        if index is None:
            index = self._get_obj_index(obj_id=obj_id, name=name)

        if adjust_object_position:
            pos = np.array(pos)
            bounds = self.get_obj_bounds(obj_id=obj_id)
            to_middle = (bounds[1] - bounds[0]) / 2 + bounds[0]
            pos[:2] -= to_middle[:2]
            pos[2] -= bounds[0][2]

        if _factory_string == "mujoco":
            from alr_sim.sims.mujoco.mj_utils.mujoco_scene_object import MujocoObject

            return (
                MujocoObject(
                    object_name=self.names[index],
                    pos=pos,
                    quat=quat,
                    obj_path=self.get_xml_path(index=index),
                ),
                self.names[index],
            )
        elif _factory_string == "mj_beta":
            from alr_sim.sims.mj_beta.mj_utils.mj_scene_object import CustomMujocoObject

            return (
                CustomMujocoObject(
                    object_name=self.names[index],
                    pos=pos,
                    quat=quat,
                    root="/",
                    object_dir_path=self.get_xml_folder(index=index),
                ),
                self.names[index],
            )
        elif _factory_string == "pybullet":
            from alr_sim.sims.pybullet.pb_utils.pybullet_scene_object import (
                PyBulletURDFObject,
            )

            orientation = quat2euler(quat)
            return (
                PyBulletURDFObject(
                    urdf_name=f"{self.names[index]}.urdf",
                    object_name=self.names[index],
                    position=pos,
                    orientation=orientation,
                    data_dir=self.get_xml_folder(index=index),
                ),
                self.names[index],
            )


# TODO implement index protocol

if __name__ == "__main__":
    loader = YCBLoader("/media/nicolasschreiber/0C340EEC0C340EEC/YCBObjects/models/ycb")
    print(loader.get_xml_path(obj_id="048_hammer"))


# 001_chips_can/textured
# 002_master_chef_can/textured
# 003_cracker_box/textured
# 004_sugar_box/textured
# 005_tomato_soup_can/textured
# 006_mustard_bottle/textured
# 007_tuna_fish_can/textured
# 008_pudding_box/textured
# 009_gelatin_box/textured
# 010_potted_meat_can/textured
# 011_banana/textured
# 012_strawberry/textured
# 013_apple/textured
# 014_lemon/textured
# 015_peach/textured
# 016_pear/textured
# 017_orange/textured
# 018_plum/textured
# 019_pitcher_base/textured
# 021_bleach_cleanser/textured
# 022_windex_bottle/textured
# 023_wine_glass/textured
# 024_bowl/textured
# 025_mug/textured
# 026_sponge/textured
# 028_skillet_lid/textured
# 029_plate/textured
# 030_fork/textured
# 031_spoon/textured
# 032_knife/textured
# 033_spatula/textured
# 035_power_drill/textured
# 036_wood_block/textured
# 037_scissors/textured
# 038_padlock/textured
# 040_large_marker/textured
# 041_small_marker/textured
# 042_adjustable_wrench/textured
# 043_phillips_screwdriver/textured
# 044_flat_screwdriver/textured
# 048_hammer/textured
# 049_small_clamp/textured
# 050_medium_clamp/textured
# 052_extra_large_clamp/textured
# 053_mini_soccer_ball/textured
# 054_softball/textured
# 055_baseball/textured
# 056_tennis_ball/textured
# 057_racquetball/textured
# 058_golf_ball/textured
# 059_chain/textured
# 061_foam_brick/textured
# 062_dice/textured
# 063-a_marbles/textured
# 063-b_marbles/textured
# 065-a_cups/textured
# 065-b_cups/textured
# 065-c_cups/textured
# 065-d_cups/textured
# 065-e_cups/textured
# 065-f_cups/textured
# 065-g_cups/textured
# 065-h_cups/textured
# 065-i_cups/textured
# 065-j_cups/textured
# 070-a_colored_wood_blocks/textured
# 070-b_colored_wood_blocks/textured
# 071_nine_hole_peg_test/textured
# 072-a_toy_airplane/textured
# 072-b_toy_airplane/textured
# 072-c_toy_airplane/textured
# 072-d_toy_airplane/textured
# 072-e_toy_airplane/textured
# 072-h_toy_airplane/textured
# 072-k_toy_airplane/textured
# 073-a_lego_duplo/textured
# 073-b_lego_duplo/textured
# 073-c_lego_duplo/textured
# 073-d_lego_duplo/textured
# 073-e_lego_duplo/textured
# 073-f_lego_duplo/textured
# 073-g_lego_duplo/textured
# 073-h_lego_duplo/textured
# 073-i_lego_duplo/textured
# 073-j_lego_duplo/textured
# 073-k_lego_duplo/textured
# 073-l_lego_duplo/textured
# 073-m_lego_duplo/textured
# 076_timer/textured
# 077_rubiks_cube/textured
