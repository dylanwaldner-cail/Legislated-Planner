import os
import torch
import imageio
import numpy as np
from einops import rearrange, repeat
from utils import (
    cfg_to_dict,
    seed,
    slice_trajdict_with_t,
    aggregate_dct,
    move_to_device,
    concat_trajdict,
)
from torchvision import utils

# === HARNESS EDIT: illegal-region overlay imports + state-coord bounds ===
# Loaded from the same train_probe yaml that cem.py/train_probe.py use so all
# scripts share one source of truth for illegal_region.
from omegaconf import OmegaConf
from hydra.utils import get_original_cwd

# Visible bounds of the U_MAZE rendered frame in AGENT STATE COORDS.
#
# Why state coords: illegal_region in the yaml is in agent qpos / state coords
# (that's the space is_illegal_state runs in). The MuJoCo world has a fixed
# +1.2 offset from state (the particle body is positioned at world (1.2, 1.2)
# in maze_model.py:59 and ball_x / ball_y are slide joints == displacement
# from that origin), but we never have to think about world coords here -- we
# stay entirely in state coords.
#
# How the range was computed (empirical calibration via rendering the agent at
# two known states and back-solving the affine state->pixel transform):
#   pixel_x = 27.8976 * state_x + 61.3072
#   pixel_y = 27.8976 * state_y + 61.3072
# from which:
#   visible state x range = (-2.1976, 5.8318)
#   visible state y range = (-2.1976, 5.8318)
#
# Important: in this view, image-y and state-y go the SAME direction (high
# state y -> high pixel y, i.e., bottom of image). No flip in _illegal_pixel_box.
# This is the opposite of my earlier first-principles guess.
OVERLAY_STATE_X_RANGE = (-2.1976, 5.8318)
OVERLAY_STATE_Y_RANGE = (-2.1976, 5.8318)
OVERLAY_BORDER_PX     = 2
OVERLAY_COLOR_RGB     = np.array([255, 0, 0], dtype=np.uint8)

# Optional: also overlay a state-coord grid for calibration. Set to False once
# you're satisfied with alignment to keep demo videos clean.
OVERLAY_DRAW_GRID = True
OVERLAY_GRID_COLOR = (0, 255, 255)   # cyan, visible against maze gray
# === END HARNESS EDIT ===


class PlanEvaluator:  # evaluator for planning
    def __init__(
        self,
        obs_0,
        obs_g,
        state_0,
        state_g,
        env,
        wm,
        frameskip,
        seed,
        preprocessor,
        n_plot_samples,
    ):
        self.obs_0 = obs_0
        self.obs_g = obs_g
        self.state_0 = state_0
        self.state_g = state_g
        self.env = env
        self.wm = wm
        self.frameskip = frameskip
        self.seed = seed
        self.preprocessor = preprocessor
        self.n_plot_samples = n_plot_samples
        self.device = next(wm.parameters()).device

        self.plot_full = False  # plot all frames or frames after frameskip

        # === HARNESS EDIT: load illegal_region for video overlay ===
        # If load fails (e.g. env != point_maze, yaml missing), overlay is
        # silently disabled rather than crashing the eval loop.
        try:
            train_cfg_path = os.path.join(
                get_original_cwd(), "conf", "train_probe_point_maze.yaml"
            )
            train_cfg = OmegaConf.load(train_cfg_path)
            self.illegal_region = OmegaConf.to_container(
                train_cfg.probe.illegal_region, resolve=True
            )
        except Exception as exc:
            print(f"[PlanEvaluator] illegal_region overlay disabled: {exc}")
            self.illegal_region = None
        # === END HARNESS EDIT ===

    def assign_init_cond(self, obs_0, state_0):
        self.obs_0 = obs_0
        self.state_0 = state_0

    def assign_goal_cond(self, obs_g, state_g):
        self.obs_g = obs_g
        self.state_g = state_g

    def get_init_cond(self):
        return self.obs_0, self.state_0

    def _get_trajdict_last(self, dct, length):
        new_dct = {}
        for key, value in dct.items():
            new_dct[key] = self._get_traj_last(value, length)
        return new_dct

    def _get_traj_last(self, traj_data, length):
        last_index = np.where(length == np.inf, -1, length - 1)
        last_index = last_index.astype(int)
        if isinstance(traj_data, torch.Tensor):
            traj_data = traj_data[np.arange(traj_data.shape[0]), last_index].unsqueeze(
                1
            )
        else:
            traj_data = np.expand_dims(
                traj_data[np.arange(traj_data.shape[0]), last_index], axis=1
            )
        return traj_data

    def _mask_traj(self, data, length):
        """
        Zero out everything after specified indices for each trajectory in the tensor.
        data: tensor
        """
        result = data.clone()  # Clone to preserve the original tensor
        for i in range(data.shape[0]):
            if length[i] != np.inf:
                result[i, int(length[i]) :] = 0
        return result

    def eval_actions(
        self, actions, action_len=None, filename="output", save_video=False,
        full_video=False,  ### HARNESS EDIT ### True = show whole rollout, skip post-success masking (visuals only)
        precomputed_env=None,  ### HARNESS EDIT ### (e_obses, e_states) from MPC's incremental rolls -> skip the full env re-roll
    ):
        """
        actions: detached torch tensors on cuda
        Returns
            metrics, and feedback from env
        """
        n_evals = actions.shape[0]
        if action_len is None:
            action_len = np.full(n_evals, np.inf)
        # rollout in wm
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(self.obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(self.obs_g), self.device
        )
        with torch.no_grad():
            i_z_obses, _ = self.wm.rollout(
                obs_0=trans_obs_0,
                act=actions,
            )
        i_final_z_obs = self._get_trajdict_last(i_z_obses, action_len + 1)

        # rollout in env
        ### HARNESS EDIT ### reuse MPC's already-rendered frames if provided (avoids re-rolling the whole trajectory)
        if precomputed_env is not None:
            e_obses, e_states = precomputed_env
        else:
            exec_actions = rearrange(
                actions.cpu(), "b t (f d) -> b (t f) d", f=self.frameskip
            )
            exec_actions = self.preprocessor.denormalize_actions(exec_actions).numpy()
            e_obses, e_states = self.env.rollout(self.seed, self.state_0, exec_actions)
        ### END HARNESS EDIT ###
        e_visuals = e_obses["visual"]
        e_final_obs = self._get_trajdict_last(e_obses, action_len * self.frameskip + 1)
        e_final_state = self._get_traj_last(e_states, action_len * self.frameskip + 1)[
            :, 0
        ]  # reduce dim back

        # compute eval metrics
        logs, successes = self._compute_rollout_metrics(
            e_state=e_final_state,
            e_obs=e_final_obs,
            i_z_obs=i_final_z_obs,
        )

        # plot trajs
        if self.wm.decoder is not None:
            i_visuals = self.wm.decode_obs(i_z_obses)[0]["visual"]
            ### HARNESS EDIT ### full_video shows whole rollout; else mask after success
            if not full_video:
                i_visuals = self._mask_traj(i_visuals, action_len + 1)
            e_visuals = self.preprocessor.transform_obs_visual(e_visuals)
            if not full_video:
                e_visuals = self._mask_traj(e_visuals, action_len * self.frameskip + 1)
            ### END HARNESS EDIT ###
            self._plot_rollout_compare(
                e_visuals=e_visuals,
                i_visuals=i_visuals,
                successes=successes,
                save_video=save_video,
                filename=filename,
            )

        return logs, successes, e_obses, e_states

    def _compute_rollout_metrics(self, e_state, e_obs, i_z_obs):
        """
        Args
            e_state
            e_obs
            i_z_obs
        Return
            logs
            successes
        """
        eval_results = self.env.eval_state(self.state_g, e_state)
        successes = eval_results['success']

        logs = {
            f"success_rate" if key == "success" else f"mean_{key}": np.mean(value) if key != "success" else np.mean(value.astype(float))
            for key, value in eval_results.items()
        }

        print("Success rate: ", logs['success_rate'])
        print(eval_results)

        visual_dists = np.linalg.norm(e_obs["visual"] - self.obs_g["visual"], axis=1)
        mean_visual_dist = np.mean(visual_dists)
        proprio_dists = np.linalg.norm(e_obs["proprio"] - self.obs_g["proprio"], axis=1)
        mean_proprio_dist = np.mean(proprio_dists)

        e_obs = move_to_device(self.preprocessor.transform_obs(e_obs), self.device)
        e_z_obs = self.wm.encode_obs(e_obs)
        div_visual_emb = torch.norm(e_z_obs["visual"] - i_z_obs["visual"]).item()
        div_proprio_emb = torch.norm(e_z_obs["proprio"] - i_z_obs["proprio"]).item()

        logs.update({
            "mean_visual_dist": mean_visual_dist,
            "mean_proprio_dist": mean_proprio_dist,
            "mean_div_visual_emb": div_visual_emb,
            "mean_div_proprio_emb": div_proprio_emb,
        })

        return logs, successes

    # === HARNESS EDIT: illegal-region overlay helpers ===
    def _illegal_pixel_box(self, sub_h, sub_w):
        """State->pixel for the illegal_region rectangle within a sub_h x sub_w panel.
        Returns (y_min, y_max, x_min, x_max) in image pixel space.

        Empirical calibration (see header constants): image-y and state-y go
        the same direction in this top-down view, so NO y-flip here.
        """
        sx0, sx1 = OVERLAY_STATE_X_RANGE
        sy0, sy1 = OVERLAY_STATE_Y_RANGE
        ir = self.illegal_region
        nx_min = (ir["x_min"] - sx0) / (sx1 - sx0)
        nx_max = (ir["x_max"] - sx0) / (sx1 - sx0)
        ny_min = (ir["y_min"] - sy0) / (sy1 - sy0)
        ny_max = (ir["y_max"] - sy0) / (sy1 - sy0)
        x_min = int(np.clip(nx_min * sub_w, 0, sub_w - 1))
        x_max = int(np.clip(nx_max * sub_w, 0, sub_w - 1))
        y_min = int(np.clip(ny_min * sub_h, 0, sub_h - 1))
        y_max = int(np.clip(ny_max * sub_h, 0, sub_h - 1))
        return y_min, y_max, x_min, x_max

    def _draw_grid_uint8(self, frame_uint8, panel_h, panel_w):
        """Overlay the U_MAZE cell-boundary grid in STATE coords on every panel.

        Cells are 1 world unit each, centered on world ints 1..5 with half-
        extent 0.5 -> world boundaries at {1.5, 2.5, 3.5, 4.5} (inner) plus
        {0.5, 5.5} (outer). In state coords (world - 1.2) those are
        {0.3, 1.3, 2.3, 3.3} inner and {-0.7, 4.3} outer. The inner 4 lines
        in each axis are the 4x4 cell-boundary grid that defines the maze
        structure (and the cells the illegal_region was defined within).
        """
        if not OVERLAY_DRAW_GRID or self.illegal_region is None:
            return frame_uint8
        from PIL import Image, ImageDraw, ImageFont
        H, W, _ = frame_uint8.shape
        sx0, sx1 = OVERLAY_STATE_X_RANGE
        sy0, sy1 = OVERLAY_STATE_Y_RANGE
        cell_boundaries = [-0.7, 0.3, 1.3, 2.3, 3.3, 4.3]

        def x_to_px(sx):
            return int(np.clip((sx - sx0) / (sx1 - sx0) * panel_w, 0, panel_w - 1))

        def y_to_px(sy):
            return int(np.clip((sy - sy0) / (sy1 - sy0) * panel_h, 0, panel_h - 1))

        img = Image.fromarray(frame_uint8)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 8
            )
        except (OSError, IOError):
            font = ImageFont.load_default()

        for oy in range(0, H, panel_h):
            for ox in range(0, W, panel_w):
                for sx in cell_boundaries:
                    if sx < sx0 or sx > sx1:
                        continue
                    px = ox + x_to_px(sx)
                    draw.line(
                        [(px, oy), (px, oy + panel_h - 1)],
                        fill=OVERLAY_GRID_COLOR,
                        width=1,
                    )
                    draw.text(
                        (px + 1, oy + 1),
                        f"x={sx:.1f}",
                        fill=OVERLAY_GRID_COLOR,
                        font=font,
                    )
                for sy in cell_boundaries:
                    if sy < sy0 or sy > sy1:
                        continue
                    py = oy + y_to_px(sy)
                    draw.line(
                        [(ox, py), (ox + panel_w - 1, py)],
                        fill=OVERLAY_GRID_COLOR,
                        width=1,
                    )
                    draw.text(
                        (ox + 1, py + 1),
                        f"y={sy:.1f}",
                        fill=OVERLAY_GRID_COLOR,
                        font=font,
                    )
        return np.array(img)

    def _draw_illegal_outline_uint8(self, frame_uint8, panel_h, panel_w):
        """Draw a red rectangle outline at the illegal_region in every
        panel_h x panel_w sub-panel of the composite frame_uint8.
        frame_uint8: (H, W, 3) uint8 array, may contain multiple panels in a grid.
        Mutates in place.
        """
        if self.illegal_region is None:
            return frame_uint8
        H, W, _ = frame_uint8.shape
        y_min, y_max, x_min, x_max = self._illegal_pixel_box(panel_h, panel_w)
        t = OVERLAY_BORDER_PX
        for oy in range(0, H, panel_h):
            for ox in range(0, W, panel_w):
                ay0, ay1 = oy + y_min, oy + y_max
                ax0, ax1 = ox + x_min, ox + x_max
                # Top, bottom, left, right edges of the outline rectangle.
                frame_uint8[ay0:ay0 + t,     ax0:ax1]     = OVERLAY_COLOR_RGB
                frame_uint8[ay1 - t:ay1,     ax0:ax1]     = OVERLAY_COLOR_RGB
                frame_uint8[ay0:ay1,         ax0:ax0 + t] = OVERLAY_COLOR_RGB
                frame_uint8[ay0:ay1,         ax1 - t:ax1] = OVERLAY_COLOR_RGB
        return frame_uint8
    # === END HARNESS EDIT ===

    def _plot_rollout_compare(
        self, e_visuals, i_visuals, successes, save_video=False, filename=""
    ):
        """
        i_visuals may have less frames than e_visuals due to frameskip, so pad accordingly
        e_visuals: (b, t, h, w, c)
        i_visuals: (b, t, h, w, c)
        goal: (b, h, w, c)
        """
        e_visuals = e_visuals[: self.n_plot_samples]
        i_visuals = i_visuals[: self.n_plot_samples]
        goal_visual = self.obs_g["visual"][: self.n_plot_samples]
        goal_visual = self.preprocessor.transform_obs_visual(goal_visual)

        i_visuals = i_visuals.unsqueeze(2)
        i_visuals = torch.cat(
            [i_visuals] + [i_visuals] * (self.frameskip - 1),
            dim=2,
        )  # pad i_visuals (due to frameskip)
        i_visuals = rearrange(i_visuals, "b t n c h w -> b (t n) c h w")
        i_visuals = i_visuals[:, : i_visuals.shape[1] - (self.frameskip - 1)]

        correction = 0.3  # to distinguish env visuals and imagined visuals

        if save_video:
            for idx in range(e_visuals.shape[0]):
                success_tag = "success" if successes[idx] else "failure"
                frames = []
                for i in range(e_visuals.shape[1]):
                    e_obs = e_visuals[idx, i, ...]
                    i_obs = i_visuals[idx, i, ...]
                    e_obs = torch.cat(
                        [e_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
                    )
                    i_obs = torch.cat(
                        [i_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
                    )
                    frame = torch.cat([e_obs - correction, i_obs], dim=1)
                    frame = rearrange(frame, "c w1 w2 -> w1 w2 c")
                    frame = rearrange(frame, "w1 w2 c -> (w1) w2 c")
                    frame = frame.detach().cpu().numpy()
                    frames.append(frame)
                video_writer = imageio.get_writer(
                    f"{filename}_{idx}_{success_tag}.mp4", fps=12
                )

                for frame in frames:
                    frame = frame * 2 - 1 if frame.min() >= 0 else frame
                    uint8_frame = (((np.clip(frame, -1, 1) + 1) / 2) * 255).astype(np.uint8)
                    video_writer.append_data(uint8_frame)
                video_writer.close()

        # pad i_visuals or subsample e_visuals
        if not self.plot_full:
            e_visuals = e_visuals[:, :: self.frameskip]
            i_visuals = i_visuals[:, :: self.frameskip]

        n_columns = e_visuals.shape[1]
        assert (
            i_visuals.shape[1] == n_columns
        ), f"Rollout lengths do not match, {e_visuals.shape[1]} and {i_visuals.shape[1]}"

        # add a goal column
        e_visuals = torch.cat([e_visuals.cpu(), goal_visual - correction], dim=1)
        i_visuals = torch.cat([i_visuals.cpu(), goal_visual - correction], dim=1)
        rollout = torch.cat([e_visuals.cpu() - correction, i_visuals.cpu()], dim=1)
        n_columns += 1

        imgs_for_plotting = rearrange(rollout, "b h c w1 w2 -> (b h) c w1 w2")
        imgs_for_plotting = (
            imgs_for_plotting * 2 - 1
            if imgs_for_plotting.min() >= 0
            else imgs_for_plotting
        )
        utils.save_image(
            imgs_for_plotting,
            f"{filename}.png",
            nrow=n_columns,  # nrow is the number of columns
            normalize=True,
            value_range=(-1, 1),
        )
