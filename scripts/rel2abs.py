# Adapted from https://github.com/2toinf/X-VLA/blob/main/evaluation/libero/rel2abs.py
# Refined with Claude to change all hdf5 files of a give input dir to map relative actions to absolute actions

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import os
import pathlib
import numpy as np
import h5py
import shutil
from tqdm import tqdm
import robosuite.utils.transform_utils as T


def get_env_and_metadata(suite_name, task_name):
    """Get env and init states from LIBERO benchmark API."""
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()

    for task_id in range(len(task_suite.tasks)):
        task = task_suite.get_task(task_id)
        if task.name == task_name:
            bddl_file = os.path.join(
                get_libero_path("bddl_files"),
                task.problem_folder,
                task.bddl_file,
            )
            env = OffScreenRenderEnv(
                bddl_file_name=bddl_file,
                camera_heights=1,  # minimal
                camera_widths=1,  # minimal
                has_offscreen_renderer=False,
                use_camera_obs=False,
            )
            env.seed(0)
            return env, task

    raise ValueError(f"Task '{task_name}' not found in suite '{suite_name}'")


def convert_demo_to_abs(env, init_states, actions):
    """Step through env with delta actions, read absolute goal pos/ori and ctrl-frame ee_ori."""
    env.reset()
    env.set_init_state(init_states)

    for _ in range(10):
        env.step(np.array([0, 0, 0, 0, 0, 0, -1]))

    eff_site_name = "gripper0_grip_site"

    abs_actions = []
    ee_pos_site = []
    ee_ori_site = []
    for action in actions:
        obs, reward, done, info = env.step(action)
        ctrl = env.env.robots[0].controller
        goal_pos = ctrl.goal_pos
        goal_ori = T.quat2axisangle(T.mat2quat(ctrl.goal_ori))
        gripper = action[-1:]
        abs_actions.append(np.concatenate([goal_pos, goal_ori, gripper]))

        site_id = env.sim.model.site_name2id(eff_site_name)
        ee_pos_site.append(np.array(env.sim.data.site_xpos[site_id]))
        ee_ori_site.append(
            T.quat2axisangle(T.mat2quat(env.sim.data.site_xmat[site_id].reshape(3, 3)))
        )

    return np.stack(abs_actions), np.stack(ee_pos_site), np.stack(ee_ori_site)


def convert_hdf5(input_path, output_path, env):
    """Convert a single hdf5 file from delta to absolute actions."""
    print(f"Copying {input_path} -> {output_path}")
    shutil.copy(str(input_path), str(output_path))

    with h5py.File(input_path, "r") as in_f, h5py.File(output_path, "r+") as out_f:
        num_demos = len(in_f["data"])
        for i in tqdm(range(num_demos), desc="Converting demos"):
            demo = in_f[f"data/demo_{i}"]
            states = demo["states"][:]
            actions = demo["actions"][:]

            abs_actions, ee_pos_site, ee_ori_site = convert_demo_to_abs(
                env, states[0], actions
            )
            out_f[f"data/demo_{i}"]["actions"][:] = abs_actions
            out_f[f"data/demo_{i}/obs/ee_pos"][:] = ee_pos_site
            out_f[f"data/demo_{i}/obs/ee_ori"][:] = ee_ori_site


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir).resolve()
    output_dir = input_dir.parent / f"{input_dir.name}_abs"

    hdf5_files = sorted(input_dir.rglob("*.hdf5"))

    if not hdf5_files:
        print(f"No hdf5 files found in {input_dir}")
        exit(1)

    print(f"Found {len(hdf5_files)} hdf5 files")
    print(f"Output: {output_dir}")

    converted = 0
    failed = 0

    for hdf5_file in hdf5_files:
        rel_path = hdf5_file.relative_to(input_dir)
        out_file = output_dir / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if out_file.exists():
            print(f"Skipping (exists): {out_file}")
            continue

        # Get suite and task name from path
        suite_name = hdf5_file.parent.name
        task_name = hdf5_file.stem.replace("_demo", "")

        print(f"\nConverting: {hdf5_file}")
        print(f"  Suite: {suite_name}, Task: {task_name}")

        try:
            env, task = get_env_and_metadata(suite_name, task_name)
            convert_hdf5(hdf5_file, out_file, env)
            env.close()
            converted += 1
        except Exception as e:
            print(f"FAILED: {hdf5_file} — {e}")
            print(e)
            failed += 1

    print(f"\nDone. Converted: {converted}, Failed: {failed}")
