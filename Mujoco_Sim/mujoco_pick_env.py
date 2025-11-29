from mujoco_playground._src.manipulation.franka_emika_panda.pick import PandaPickCube, PandaPickCubeOrientation
from mujoco_playground._src.manipulation.franka_emika_panda.pick import default_config
from ml_collections import config_dict
from typing import  Any, Dict, Optional, Union
import numpy as np
import mujoco
from pathlib import Path
import os

"""
mjx_panda.xml has the content of panda_updated_robotiq_2f85.xml
"""
class StereoPickCube(PandaPickCube):
    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        sample_orientation: bool = False,
        img_h_size: int = 64,
        img_w_size: int = 64,
    ):  
        self.img_h_size = img_h_size
        self.img_w_size = img_w_size
        
        super().__init__(config, config_overrides, sample_orientation)
        self._xml_path = (Path(os.getcwd()) / "Mujoco_Sim" / "franka_emika_panda" / "mjx_single_cube_stereo.xml").as_posix()
        self.model = mujoco.MjModel.from_xml_path(self._xml_path)
        self.model.vis.global_.offwidth = self.img_w_size
        self.model.vis.global_.offheight = self.img_h_size
    
        # Fast offscreen renderer
        self._mjv_cam = mujoco.MjvCamera()
        self._mjv_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._mjv_opt = mujoco.MjvOption()
        self._mjv_scene = mujoco.MjvScene(self.model, maxgeom=1000)
        self._gl_ctx = None
        if hasattr(mujoco, "GLContext") and mujoco.GLContext is not None:
            self._gl_ctx = mujoco.GLContext(int(self.img_w_size), int(self.img_h_size))
            self._gl_ctx.make_current()
        else:
            raise RuntimeError(
                "MuJoCo GLContext unavailable. Ensure MUJOCO_GL is set (e.g., 'egl' or 'osmesa') "
                "BEFORE importing mujoco."
            )
        self._mjr_ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._mjr_ctx)
        self._mjr_ctx.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZEROFAR
        self._mjr_rect = mujoco.MjrRect(0, 0, int(self.img_w_size), int(self.img_h_size))
        # Preallocate CPU buffers used by mjr_readPixels (uint8, HxWx3)
        self._rgb_left  = np.empty((self.img_h_size, self.img_w_size, 3), dtype=np.uint8)
        self._rgb_right = np.empty((self.img_h_size, self.img_w_size, 3), dtype=np.uint8)
        
if __name__ == "__main__":
    env = StereoPickCube()
    print(env.xml_path)
        