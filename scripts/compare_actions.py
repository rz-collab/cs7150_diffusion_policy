import numpy as np
import h5py
import imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import os
from PIL import Image, ImageDraw
from pathlib import Path

DATASET_DIR = "data/libero_abs"

TASK_SUITE = "libero_10"

VIDEO_SAVE_DIR = "temp"

def add_text(img, text):
    pil = Image.fromarray(img)
    ImageDraw.Draw(pil).text((5, 5), text, fill=(255, 0, 0))
    return np.array(pil)


def verify_abs_actions(suite_name, task_name, abs_hdf5_path, demo_id=0):
    # Get env from benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()

    bddl_file = None
    for task_id in range(len(task_suite.tasks)):
        task = task_suite.get_task(task_id)
        if task.name == task_name:
            bddl_file = os.path.join(
                get_libero_path("bddl_files"),
                task.problem_folder,
                task.bddl_file,
            )
            break

    if bddl_file is None:
        print(f"Available tasks in {suite_name}:")
        for task_id in range(len(task_suite.tasks)):
            print(f"  {task_suite.get_task(task_id).name}")
        raise ValueError(f"Task '{task_name}' not found in suite '{suite_name}'")

    # Get init state, original images, and abs actions
    with h5py.File(abs_hdf5_path, "r") as f:
        init_state = f[f"data/demo_{demo_id}/states"][0]
        orig_images = f[f"data/demo_{demo_id}/obs/agentview_rgb"][:]
        abs_actions = f[f"data/demo_{demo_id}/actions"][:]

    # Replay with absolute actions
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(0)
    env.reset()
    env.set_init_state(init_state)
    for _ in range(10):
        env.step(np.array([0, 0, 0, 0, 0, 0, -1]))
    for robot in env.env.robots:
        robot.controller.use_delta = False

    abs_images = []
    for action in abs_actions:
        obs, reward, done, info = env.step(action)
        abs_images.append(obs["agentview_image"])
    env.close()

    # Stack side by side: original (left) | absolute replay (right)
    min_len = min(len(orig_images), len(abs_images))
    combined = []
    for i in range(min_len):
        orig = orig_images[i][::-1, ::-1]
        abs_img = abs_images[i][::-1, ::-1]

        # Put abs actions onto image for video
        action = abs_actions[i]
        a_str = " ".join(f"{x:.2f}" for x in action)
        abs_img = add_text(abs_img, f"t={i} [{a_str}]")

        if orig.shape != abs_img.shape:
            from PIL import Image

            orig = np.array(
                Image.fromarray(orig).resize((abs_img.shape[1], abs_img.shape[0]))
            )
        frame = np.concatenate([orig, abs_img], axis=1)
        combined.append(frame)

    save_path = f"{VIDEO_SAVE_DIR}/{task_name}_verify_side_by_side.mp4"
    imageio.mimsave(save_path, combined, fps=30)
    print(f"Saved {save_path} ({min_len} frames)")
    print(f"Final reward: {reward}, done: {done}")

# Get Full Path
target_dir = Path(DATASET_DIR) / TASK_SUITE

# Gets files to verify
files_to_verify = target_dir.glob("*.hdf5")

# Iterates through those files verifying each of them
for f in files_to_verify:
    print(f"Task: {f.stem}")
    verify_abs_actions(
        suite_name=TASK_SUITE,
        # Stem returns just the file name without file extension, and then cut off
        # last five characters to remove _demo from it, as that is task name
        task_name=f.stem[:-5],
        abs_hdf5_path=f,
        demo_id=0,
    )