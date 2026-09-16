"""Shared canvas, palette, type and encoding helpers for the project-page intro video.

DESIGN RULES
  * Every frame is composed in PIL at a fixed canvas and handed to ffmpeg as raw RGB. We never use
    ffmpeg's `drawtext`, so typography is fully under our control and identical across beats.
  * Type is Nimbus Roman / Nimbus Sans -- the URW clones of the two faces the paper actually ships
    (`mathptmx` for prose, `helvet` for the Figure 1 diagram). Annotation therefore matches the
    paper rather than looking like a different document.
  * Colours are the paper's own diagram palette (Jurix_paper/main.tex:24-52). Nothing is invented.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- canvas
W, H = 1920, 1080
FPS = 30

# ---------------------------------------------------------------- palette (Jurix_paper/main.tex)
INK        = (21, 24, 29)
MUTE       = (92, 99, 110)
FAINT      = (135, 142, 153)
RULE       = (226, 229, 234)
BG         = (255, 255, 255)
PANEL      = (247, 248, 250)

LAW        = (0x1D, 0x6F, 0xE0)   # diagLaw     - law / social agent
PLAN       = (0xE6, 0x39, 0x46)   # diagPlan    - planner / realistic agent
ENV        = (0x12, 0xA1, 0x50)   # diagEnv
LATENT     = (0x7C, 0x3A, 0xED)   # diagLatent  - world model / oracle
PROBE      = (0xE8, 0x89, 0x0C)   # diagProbe
DEV        = (0xE3, 0x9E, 0x14)   # deviant agent
GAP_GROUND = (0x00, 0xB4, 0xD8)   # gapblue     - grounding isomorphism gap
GAP_ONTO   = (0xC2, 0x18, 0x5B)   # gapred      - ontological isomorphism gap

SIGN_RGB = {"white": (255, 255, 255), "yellow": (246, 208, 47),
            "red": (230, 57, 70), "green": (0, 204, 0)}

# ---------------------------------------------------------------- type
_FD = Path("/usr/share/fonts/opentype/urw-base35")
_ROMAN, _ROMAN_B, _ROMAN_I = "NimbusRoman-Regular.otf", "NimbusRoman-Bold.otf", "NimbusRoman-Italic.otf"
_SANS, _SANS_B = "NimbusSans-Regular.otf", "NimbusSans-Bold.otf"
_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"          # for atoms / rule text
_MONO_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
# Lato for display captions -- the same face dino-wm.github.io uses, and a cleaner, more modern
# voice than the paper's Times for on-screen titling.
_LATO = "/usr/share/fonts/truetype/lato/Lato-Black.ttf"
_LATO_B = "/usr/share/fonts/truetype/lato/Lato-Bold.ttf"

_cache: dict = {}


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    """kind in {roman, roman_b, roman_i, sans, sans_b, display, display_b, mono, mono_b}."""
    key = (kind, size)
    if key not in _cache:
        path = {"roman": _FD / _ROMAN, "roman_b": _FD / _ROMAN_B, "roman_i": _FD / _ROMAN_I,
                "sans": _FD / _SANS, "sans_b": _FD / _SANS_B,
                "display": Path(_LATO), "display_b": Path(_LATO_B),
                "mono": Path(_MONO), "mono_b": Path(_MONO_B)}[kind]
        _cache[key] = ImageFont.truetype(str(path), size)
    return _cache[key]


def new_frame(bg=BG) -> Image.Image:
    return Image.new("RGB", (W, H), bg)


def text(draw: ImageDraw.ImageDraw, xy, s, *, kind="roman", size=40, fill=INK,
         anchor="la", spacing=10, align="left"):
    draw.multiline_text(xy, s, font=font(kind, size), fill=fill, anchor=anchor,
                        spacing=spacing, align=align)


def text_tracked(draw, xy, s, *, kind="display", size=80, fill=INK, track=2, anchor_mid=True):
    """Draw `s` with letter-spacing `track` px, centered on xy[0]; xy[1] is the BASELINE.

    Glyphs are drawn one at a time, so they must be anchored on the baseline ("ls"). Anchor "lt"
    aligns each glyph's INK TOP instead, which drops the tall letters and raises the x-height ones
    -- the text visibly jitters along its own baseline.
    """
    f = font(kind, size)
    widths = [f.getlength(ch) for ch in s]
    total = sum(widths) + track * (len(s) - 1)
    x = xy[0] - total / 2 if anchor_mid else xy[0]
    for ch, w in zip(s, widths):
        draw.text((x, xy[1]), ch, font=f, fill=fill, anchor="ls")
        x += w + track
    return total


def measure(s, kind="roman", size=40):
    f = font(kind, size)
    box = f.getbbox(s)
    return box[2] - box[0], box[3] - box[1]


def fit_image(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    """Scale to fit inside (box_w, box_h), preserving aspect. Never upscales past 2x."""
    s = min(box_w / img.width, box_h / img.height)
    return img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)


# ---------------------------------------------------------------- easing / compositing
def ease(t: float) -> float:
    """cubic-bezier(.22,.61,.36,1)-ish ease-out, matching the page's reveal easing."""
    t = min(1.0, max(0.0, t))
    return 1.0 - (1.0 - t) ** 3


def blend(a: Image.Image, b: Image.Image, t: float) -> Image.Image:
    return Image.blend(a, b, min(1.0, max(0.0, t)))


def hold(frames: list, n: int) -> list:
    return frames + [frames[-1]] * n if frames else frames


def crossfade(a: list, b: list, n: int) -> list:
    """Concatenate two frame lists with an n-frame crossfade between them."""
    if not a:
        return list(b)
    if not b:
        return list(a)
    n = min(n, len(a), len(b))
    if n == 0:
        return list(a) + list(b)
    mid = [blend(a[-n + i], b[i], (i + 1) / n) for i in range(n)]
    return list(a[:-n]) + mid + list(b[n:])


# ---------------------------------------------------------------- encoding
def write_mp4(frames, out, fps: int = FPS, crf: int = 18):
    """Pipe RGB frames straight into libx264. yuv420p + even dims so browsers can play it."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h = frames[0].size
    assert w % 2 == 0 and h % 2 == 0, f"odd dimensions {w}x{h} break yuv420p"
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.asarray(f.convert("RGB"), dtype=np.uint8).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {out}")
    mb = out.stat().st_size / 1e6
    print(f"[video] wrote {out}  {len(frames)} frames  {len(frames)/fps:.1f}s  {mb:.1f} MB")
    return out


def read_mp4(path, max_frames=None) -> list:
    """Decode an mp4 to a list of PIL frames (used for the episode clips)."""
    import imageio.v2 as imageio
    rd = imageio.get_reader(str(path))
    out = []
    for i, fr in enumerate(rd):
        if max_frames and i >= max_frames:
            break
        out.append(Image.fromarray(np.asarray(fr)[..., :3]))
    rd.close()
    return out
