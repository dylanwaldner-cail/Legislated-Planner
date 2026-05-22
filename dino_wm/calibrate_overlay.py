import sys
import numpy as np
sys.path.insert(0, "/newdata2/dylantw/Legislative-Harness/dino_wm")

import env  # noqa: F401
from env.pointmaze.point_maze_wrapper import PointMazeWrapper
from env.pointmaze.maze_model import U_MAZE
from PIL import Image


def find_agent_pixel(img):
  """Agent: darker green sphere (rgba='0.3 0.6 0.3 1' ~ RGB(77,153,77)).
  Target: brighter green (RGB ~(127,255,127)). Mask the darker band."""
  r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
  mask = (g > r + 15) & (g > b + 15) & (g >= 100) & (g <= 200)
  ys, xs = np.where(mask)
  if len(xs) == 0:
      return None, None
  return float(xs.mean()), float(ys.mean())


# One env, set up the camera ONCE, then directly drive qpos for each render.
e = PointMazeWrapper(maze_spec=U_MAZE, reward_type="sparse", reset_target=False)
e.seed(0)
e.prepare_for_render()  # locks in camera angles; overrides agent position once

def render_at(state_xy):
  e.set_state(np.array(state_xy, dtype=np.float64), np.zeros(2))
  return e._render_frame()


ANCHORS = [(1.0, 1.0), (3.0, 3.0)]
results = []
for s in ANCHORS:
  img = render_at(s)
  px, py = find_agent_pixel(img)
  results.append((s, px, py))
  fname = f"/tmp/agent_at_{s[0]}_{s[1]}.png"
  Image.fromarray(img).save(fname)
  print(f"State {s} -> pixel ({px}, {py})   saved {fname}")

(s1, px1, py1), (s2, px2, py2) = results
if None in (px1, py1, px2, py2):
  raise SystemExit("Agent not detected in one of the renders — check the saved PNGs.")

ax = (px2 - px1) / (s2[0] - s1[0])
bx = px1 - ax * s1[0]
ay = (py2 - py1) / (s2[1] - s1[1])
by = py1 - ay * s1[1]

print(f"\npixel_x = {ax:.4f} * state_x + {bx:.4f}")
print(f"pixel_y = {ay:.4f} * state_y + {by:.4f}")

W = H = 224
x_at_0, x_at_W = (0 - bx) / ax, (W - bx) / ax
y_at_0, y_at_H = (0 - by) / ay, (H - by) / ay
x_range = (min(x_at_0, x_at_W), max(x_at_0, x_at_W))
y_range = (min(y_at_0, y_at_H), max(y_at_0, y_at_H))

print(f"\nVisible state x range = ({x_range[0]:.4f}, {x_range[1]:.4f})")
print(f"Visible state y range = ({y_range[0]:.4f}, {y_range[1]:.4f})")
print(f"x-axis flipped (ax<0): {ax < 0}")
print(f"y-axis flipped (ay<0): {ay < 0}")
print(f"\nOVERLAY_STATE_X_RANGE = ({x_range[0]:.4f}, {x_range[1]:.4f})")
print(f"OVERLAY_STATE_Y_RANGE = ({y_range[0]:.4f}, {y_range[1]:.4f})")
