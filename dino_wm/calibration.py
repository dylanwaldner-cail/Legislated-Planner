import sys
sys.path.insert(0, '/newdata2/dylantw/Legislative-Harness/dino_wm')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from env.pointmaze.point_maze_wrapper import PointMazeWrapper

U_MAZE = \
        "#####\\"+\
        "#GOO#\\"+\
        "###O#\\"+\
        "#OOO#\\"+\
        "#####"

def world_to_pixel(x, y):
    x_scale = (143 - 88) / (3.0 - 1.0)
    y_scale = (150 - 88) / (3.0 - 1.0)
    px = int(88 + (x - 1.0) * x_scale)
    py = int(88 + (y - 1.0) * y_scale)
    return px, py

illegal_region = {'x_min': 1.2, 'x_max': 2.8, 'y_min': 2.4, 'y_max': 2.8}

env = PointMazeWrapper(maze_spec=U_MAZE)
env.prepare_for_render()
env.set_init_state(np.array([1.0, 1.0, 0.0, 0.0]))
obs, _ = env.reset()
frame = obs['visual']

x1, y1 = world_to_pixel(illegal_region['x_min'], illegal_region['y_min'])
x2, y2 = world_to_pixel(illegal_region['x_max'], illegal_region['y_max'])

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(frame)
rect = patches.Rectangle(
    (x1, y1),
    x2 - x1,
    y2 - y1,
    linewidth=2,
    edgecolor='red',
    facecolor='red',
    alpha=0.3
)
ax.add_patch(rect)
ax.set_title('U-Maze with illegal region')
plt.savefig('illegal_region_overlay.png', dpi=150, bbox_inches='tight')
print("Saved illegal_region_overlay.png")
