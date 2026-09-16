"""Draw the 3x3 legal grid in world coordinates, for every beat that happens on the table.

Geometry comes from env/isaaclab/grid_metadata.py via the repo's own accessor, so the cells, the
forbidden centre and the cube footprint are placed exactly where the planner and the metrics think
they are -- nothing here re-derives the layout. Styling follows scripts/plot_rrt_tree_schematic.py
so the video and the paper's RRT figures read as the same picture.
"""
from __future__ import annotations

import sys

from PIL import ImageDraw

sys.path.insert(0, "/newdata2/dylantw/Legislated-Planner/src")
from probes.probe_cube_position import gm  # noqa: E402
from probes.probe_cube_cells import CUBE_HALF  # noqa: E402
from scripts.video import common as c  # noqa: E402

YELLOW_CELLS, RED_CELL = (3, 5), 4
# The paper's figures tint these very lightly (alpha 0.12-0.22) because they sit under dense plot
# ink. On screen at video scale that reads as washed out, so the same hues are composited at a
# higher alpha -- same colours, more saturation.
CELL_FILL = {"green": (168, 218, 187), "yellow": (250, 219, 104), "red": (243, 157, 164)}
GRIDLINE, OUTER = (140, 146, 154), (48, 52, 58)
TREE_EDGE, TREE_NODE = (110, 155, 209), c.LAW


class GridCanvas:
    """Maps env-local metres to pixels inside a square box."""

    def __init__(self, cx: int, cy: int, size: int):
        self.cx, self.cy, self.size = cx, cy, size
        self.s = size / (2.0 * gm.GRID_HALF)          # px per metre

    def px(self, x: float, y: float):
        """World (x,y) -> pixel. y is flipped: world +y is up, pixels grow downward."""
        return (self.cx + x * self.s, self.cy - y * self.s)

    def m(self, metres: float) -> float:
        return metres * self.s

    # ---------------------------------------------------------------- drawing
    def draw_grid(self, d: ImageDraw.ImageDraw, label_cells=True, dim_red=False):
        for cid in range(gm.N_CELLS):
            gx, gy = gm.cell_center(cid)
            kind = "red" if cid == RED_CELL else ("yellow" if cid in YELLOW_CELLS else "green")
            fill = CELL_FILL[kind]
            if kind == "red" and not dim_red:
                fill = (238, 132, 141)          # the forbidden cell is the subject: strongest tint
            x0, y0 = self.px(gx - gm.CELL / 2, gy + gm.CELL / 2)
            x1, y1 = self.px(gx + gm.CELL / 2, gy - gm.CELL / 2)
            d.rectangle([x0, y0, x1, y1], fill=fill, outline=GRIDLINE, width=2)
            if label_cells:
                c.text(d, self.px(gx, gy), str(cid), kind="sans_b", size=26,
                       fill=(150, 154, 160), anchor="mm")
        h = self.size / 2
        d.rectangle([self.cx - h, self.cy - h, self.cx + h, self.cy + h], outline=OUTER, width=3)

    def draw_footprint(self, d, x, y, colour, width=3, half=CUBE_HALF, dash=False):
        """The cube's axis-aligned footprint -- the body the law is actually enforced against."""
        x0, y0 = self.px(x - half, y + half)
        x1, y1 = self.px(x + half, y - half)
        if dash:
            self._dashed_rect(d, x0, y0, x1, y1, colour, width)
        else:
            d.rectangle([x0, y0, x1, y1], outline=colour, width=width)

    @staticmethod
    def _dashed_rect(d, x0, y0, x1, y1, colour, width, dash=9, gap=7):
        for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                     ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
            dx, dy = b[0] - a[0], b[1] - a[1]
            n = max(1.0, (dx * dx + dy * dy) ** 0.5)
            steps = int(n // (dash + gap)) + 1
            for k in range(steps):
                t0 = min(1.0, (k * (dash + gap)) / n)
                t1 = min(1.0, (k * (dash + gap) + dash) / n)
                d.line([a[0] + dx * t0, a[1] + dy * t0, a[0] + dx * t1, a[1] + dy * t1],
                       fill=colour, width=width)

    def draw_edge(self, d, p0, p1, colour, width=4, dash=False):
        a, b = self.px(*p0), self.px(*p1)
        if not dash:
            d.line([a, b], fill=colour, width=width)
            return
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = max(1.0, (dx * dx + dy * dy) ** 0.5)
        k, step = 0, 16
        while k * step < n:
            t0, t1 = (k * step) / n, min(1.0, (k * step + 9) / n)
            d.line([a[0] + dx * t0, a[1] + dy * t0, a[0] + dx * t1, a[1] + dy * t1],
                   fill=colour, width=width)
            k += 1

    def draw_node(self, d, x, y, colour, r=9, outline=None):
        px, py = self.px(x, y)
        d.ellipse([px - r, py - r, px + r, py + r], fill=colour,
                  outline=outline or (255, 255, 255), width=2)

    def draw_cross(self, d, x, y, colour, r=13, width=5):
        px, py = self.px(x, y)
        d.line([px - r, py - r, px + r, py + r], fill=colour, width=width)
        d.line([px - r, py + r, px + r, py - r], fill=colour, width=width)
