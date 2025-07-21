from FrankaSim.franka import FrankaEnv
import numpy as np
import os
import mujoco
import cv2

env = FrankaEnv(
    model_path=os.path.join(os.getcwd(), "FrankaSim", "pick_place.xml"),
    render_mode="rgb_array",
    n_substeps=25,
    reward_type="dense",
    distance_threshold=0.05,
    goal_xy_range=0.3,
    obj_xy_range=0.3,
    goal_x_offset=0.0,
    goal_z_range=0.2,
)

obs, info = env.reset()
print("Initial observation:", obs)

renderer = mujoco.Renderer(env.model)
left_cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "left_eye")
right_cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "right_eye")

for step in range(10_000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

    renderer.update_scene(env.data, camera=left_cam_id)
    renderer.update_scene(env.data, camera=right_cam_id)
    left_img = renderer.render()
    right_img = renderer.render()

    stereo_view = np.concatenate((left_img, right_img), axis=1)
    stereo_view_bgr = cv2.cvtColor(stereo_view, cv2.COLOR_RGB2BGR)
    cv2.imshow("Stereo View (Left | Right)", stereo_view_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

env.close()
cv2.destroyAllWindows()
