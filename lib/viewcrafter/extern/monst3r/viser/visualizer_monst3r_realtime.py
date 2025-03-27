"""Record3D visualizer

Parse and stream record3d captures. To get the demo data, see `./assets/download_record3d_dance.sh`.
"""

import time
from pathlib import Path

import numpy as onp
import tyro
from tqdm.auto import tqdm

import viser
import viser.extras
import viser.transforms as tf

import matplotlib.cm as cm  # For colormap
from glob import glob
import numpy as np
from scipy.spatial.transform import Rotation
import imageio.v3 as iio

# python viser/visualizer_monst3r_realtime.py --data_path demo_tmp/lady-running

def main(
    data_path: str = "./demo_tmp/NULL",
    up_dir: str = "-z", # should be +z or -z
    max_frames: int = 200,
    share: bool = False,
    point_size: float = 0.002,
    head2: bool = True,
    downsample_factor: int = 1,
    num_traj_points: int = 100,
    conf_thre: float = 1.,
    visualize_cams: bool = False,
    camera_frustum_scale: float = 0.02,
    cam_thickness: float = 1.5,
) -> None:
    server = viser.ViserServer()
    if share:
        server.request_share_url()

    server.scene.set_up_direction(up_dir)
    print("Loading frames!")
    # if data_path is a directory, load all npy files in the directory
    if Path(data_path).is_dir():
        if head2:
            traj_3d_paths = sorted(glob(data_path + '/pts3d2_p*.npy'), key=lambda x: int(x.split('_p')[-1].split('.')[0]))
            conf_pathes = sorted(glob(data_path + '/conf2_p*.npy'), key=lambda x: int(x.split('_p')[-1].split('.')[0]))
        else:
            traj_3d_paths = sorted(glob(data_path + '/pts3d1_p*.npy'), key=lambda x: int(x.split('_p')[-1].split('.')[0]))
            conf_pathes = sorted(glob(data_path + '/conf1_p*.npy'), key=lambda x: int(x.split('_p')[-1].split('.')[0]))
        traj_3d = onp.stack([onp.load(p) for p in traj_3d_paths], axis=0)  # (T, H, W, 6)
        traj_3d = traj_3d.reshape(traj_3d.shape[0], -1, 6)  # (T, N, 6)
        if len(conf_pathes) > 0:
            conf = onp.stack([onp.load(p) for p in conf_pathes], axis=0)  # (T, H, W)
            conf = conf.reshape(conf.shape[0], -1)  # (T, N)
            conf_mask = conf > conf_thre
        else:
            conf_mask = None
        print(f"Loaded {len(traj_3d_paths)} files, with shape {traj_3d.shape}")
    else:
        traj_3d_path = data_path
        traj_3d = onp.load(traj_3d_path)     # (T, N, 6)
        conf_mask = None
        print(f"Loaded {traj_3d_path}, with shape {traj_3d.shape}")

    xyz = traj_3d[:, :, :3]
    # center the point cloud
    center_point = onp.mean(xyz, axis=(0, 1), keepdims=True)
    xyz -= center_point
    rgb = traj_3d[:, :, 3:6]
    if rgb.sum(axis=(-1)).max() > 125:
        rgb /= 255.0
    F, N, _ = traj_3d.shape
    num_frames = min(max_frames, F)

    if visualize_cams:
        intrinsics_path = Path(data_path) / "pred_intrinsics.txt"
        assert intrinsics_path.exists(), f"Cannot find intrinsics file at {intrinsics_path}"
        intrinsics = np.loadtxt(intrinsics_path)
        Ks = np.array(intrinsics, np.float32).reshape(-1, 3, 3)
        poses_path = Path(data_path) / "pred_traj.txt"
        poses_matrix_path = Path(data_path) / "pred_traj_matrix.txt"
        if poses_matrix_path.exists():
            T_world_cameras: onp.ndarray = np.loadtxt(poses_matrix_path)
            T_world_cameras = T_world_cameras.astype(np.float32).reshape(-1, 4, 4)
        else:
            assert poses_path.exists(), f"Cannot find poses file at {poses_path}"
            poses = np.loadtxt(poses_path)
            T_world_cameras: onp.ndarray = np.array(poses, np.float32)
            T_world_cameras = np.concatenate(
                [
                    # Convert TUM pose to SE3 pose
                    Rotation.from_quat(np.concatenate([T_world_cameras[:, 5:], T_world_cameras[:, 4:5]], -1)).as_matrix(),
                    T_world_cameras[:, 1:4, None],
                ],
                -1,
            )
            T_world_cameras = T_world_cameras.astype(np.float32)
            ones = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (F, 1, 1))
            T_world_cameras = np.concatenate([T_world_cameras, ones], axis=1)
        # center the camera poses
        T_world_cameras[:, :3, 3] -= center_point.squeeze()
        rgb_paths = sorted(Path(data_path).glob("img2_p*.png"), key=lambda p: int(p.stem.split("_p")[-1])) if head2 else sorted(Path(data_path).glob("img1_p*.png"), key=lambda p: int(p.stem.split("_p")[-1]))
        frame_rgbs = [iio.imread(p) for p in rgb_paths]
        assert len(frame_rgbs) == num_frames, f"Number of frames {len(frame_rgbs)} does not match number of poses {num_frames}"

    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=20
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30", "60")
        )
        gui_show_all_frames = server.gui.add_checkbox("Show all frames", False)
        gui_stride = server.gui.add_slider(
            "Stride",
            min=1,
            max=num_frames,
            step=1,
            initial_value=1,
            disabled=True,  # Initially disabled
        )

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value or gui_show_all_frames.value
        gui_next_frame.disabled = gui_playing.value or gui_show_all_frames.value
        gui_prev_frame.disabled = gui_playing.value or gui_show_all_frames.value

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    prev_timestep = gui_timestep.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        if not gui_show_all_frames.value:
            with server.atomic():
                frame_nodes[current_timestep].visible = True
                frame_nodes[prev_timestep].visible = False
        prev_timestep = current_timestep
        server.flush()  # Optional!

    # Show or hide all frames based on the checkbox.
    @gui_show_all_frames.on_update
    def _(_) -> None:
        gui_stride.disabled = not gui_show_all_frames.value  # Enable/disable stride slider
        if gui_show_all_frames.value:
            # Show frames with stride
            stride = gui_stride.value
            with server.atomic():
                for i, frame_node in enumerate(frame_nodes):
                    frame_node.visible = (i % stride == 0)
            # Disable playback controls
            gui_playing.disabled = True
            gui_timestep.disabled = True
            gui_next_frame.disabled = True
            gui_prev_frame.disabled = True
        else:
            # Show only the current frame
            current_timestep = gui_timestep.value
            with server.atomic():
                for i, frame_node in enumerate(frame_nodes):
                    frame_node.visible = i == current_timestep
            # Re-enable playback controls
            gui_playing.disabled = False
            gui_timestep.disabled = gui_playing.value
            gui_next_frame.disabled = gui_playing.value
            gui_prev_frame.disabled = gui_playing.value

    # Update frame visibility when the stride changes.
    @gui_stride.on_update
    def _(_) -> None:
        if gui_show_all_frames.value:
            # Update frame visibility based on new stride
            stride = gui_stride.value
            with server.atomic():
                for i, frame_node in enumerate(frame_nodes):
                    frame_node.visible = (i % stride == 0)

    # Load in frames.
    server.scene.add_frame(
        "/frames",
        wxyz=tf.SO3.exp(onp.array([onp.pi / 2.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )
    frame_nodes: list[viser.FrameHandle] = []
    for i in tqdm(range(num_frames)):
        position, color = xyz[i], rgb[i]
        if conf_mask is None:
            conf_mask_i = onp.ones(N, dtype=bool)
        else:
            conf_mask_i = conf_mask[i]
        position = position[conf_mask_i]
        color = color[conf_mask_i]
        
        # Add base frame.
        frame_nodes.append(server.scene.add_frame(f"/frames/t{i}", show_axes=False))

        # Place the point cloud in the frame.
        server.scene.add_point_cloud(
            name=f"/frames/t{i}/point_cloud",
            points=position[::downsample_factor],
            colors=color[::downsample_factor],
            point_size=point_size,
            point_shape="rounded",
        )
        if visualize_cams:
            K = Ks[i]
            T_world_camera = T_world_cameras[i]
            frame_rgb = frame_rgbs[i]
            frame_rgb = frame_rgb[..., :3]
            # Compute color for frustum based on frame index.
            norm_i = i / (num_frames - 1) if num_frames > 1 else 0  # Normalize index to [0, 1]
            color_rgba = cm.viridis(norm_i)  # Get RGBA color from colormap
            color_rgb = color_rgba[:3]  # Use RGB components

            fov = 2 * onp.arctan2(rgb.shape[0] / 2, K[0, 0])
            aspect = frame_rgb.shape[1] / frame_rgb.shape[0]
            server.scene.add_camera_frustum(
                f"/frames/t{i}/frustum",
                fov=fov,
                aspect=aspect,
                scale=camera_frustum_scale,
                image=frame_rgb[::downsample_factor, ::downsample_factor],
                wxyz=tf.SO3.from_matrix(T_world_camera[:3, :3]).wxyz,
                position=T_world_camera[:3, 3],
                color=color_rgb,  # Set the color for the frustum
                thickness=cam_thickness,
            )

        if not head2 and num_traj_points > 0 and i>2:
            for n in range(N):
                if n % (N // num_traj_points) == 0:
                    color = cm.viridis(n / N)[:3]
                    server.scene.add_spline_catmull_rom(
                        name=f"/frames/t{i}/trajectory{n}",
                        positions=xyz[i-2:i][:, n],
                        line_width=3.0,
                        # add a color based on n
                        color=color,
                    )

    # Hide all but the current frame.
    for i, frame_node in enumerate(frame_nodes):
        if gui_show_all_frames.value:
            frame_node.visible = (i % gui_stride.value == 0)
        else:
            frame_node.visible = i == gui_timestep.value

    # Playback update loop.
    prev_timestep = gui_timestep.value
    while True:
        if gui_playing.value and not gui_show_all_frames.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    tyro.cli(main)
