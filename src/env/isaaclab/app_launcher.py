"""Lazy, idempotent SimulationApp boot.

Must run before any `isaaclab.*` or `isaaclab_tasks.*` import. Call
`get_app()` first, then import everything else.

The `renderer` arg lets you swap Omniverse's render delegate. Default RTX
needs driver >= 535.129; pass `renderer="pxr"` to try Pixar's Hydra Storm
(OpenGL-based, no RTX driver requirement) when the host driver is too old.
"""
from __future__ import annotations

import os

# Env vars that trigger GLX/X11 paths inside Kit and cause GLXBadFBConfig in
# headless containers without a real display. Scrubbed before AppLauncher boot.
_PRUNE_ENV = (
    "DISPLAY",
    "XAUTHORITY",
    "LIBGL_DRIVERS_PATH",
    "__EGL_VENDOR_LIBRARY_FILENAMES",
    "__GLX_VENDOR_LIBRARY_NAME",
)

_app = None


def get_app(
    headless: bool = True,
    enable_cameras: bool = True,
    render_mode: str = "PathTracing",
    spp: int = 128,
    device: str = "cuda:0",  ### HARNESS EDIT ### selects the GPU; AppLauncher derives active_gpu (renderer) + physics_gpu from it
):
    """Boot Kit's SimulationApp (idempotent).

    render_mode selects the RTX rendering mode:
      "PathTracing"       - Monte Carlo path tracing, clean output, slower per frame.
      "RaytracedLighting" - real-time ray tracing, faster but noisier.
    Both are RTX modes; the IsaacLab Camera sensor only works with RTX.

    spp is samples-per-pixel for PathTracing. Noise floor scales ~1/sqrt(spp):
    128 (default) is fine for training data; 256+ for demo videos. Ignored
    in RaytracedLighting mode.
    """
    global _app
    if _app is None:
        for v in _PRUNE_ENV:
            os.environ.pop(v, None)
        from isaaclab.app import AppLauncher

        # Kit decodes NVIDIA's Vulkan driverVersion with an 8-bit minor field;
        # drivers with minor >= 256 (e.g. 535.261.03) overflow to e.g. "535.5"
        # and then fail the RTX driver check against the 535.129 floor.
        kit_args_parts = ["--/rtx/verifyDriverVersion/enabled=false"]
        _launcher = AppLauncher(
            headless=headless,
            enable_cameras=enable_cameras,
            device=device,  ### HARNESS EDIT ### pins renderer (active_gpu) + physics_gpu to this device
            kit_args=" ".join(kit_args_parts),
        )
        _app = _launcher.app

        # Set RTX settings via carb at runtime — kit_args strings have been
        # going silent. carb.settings is the official API; we read back values
        # to verify.
        #
        # OptiX denoiser is NOT enabled: this container's OptiX install fails
        # to load denoiser weights (OPTIX_ERROR_INTERNAL_ERROR) and asking for
        # it just spams errors per frame without actually denoising. Quality
        # instead comes from raw samples-per-pixel (SPP=128) plus per-frame
        # averaging in callers (e.g. isaac_calibration.py --avg_frames).
        try:
            import carb
            s = carb.settings.get_settings()
            s.set_string("/rtx/rendermode", render_mode)
            if render_mode == "PathTracing":
                for path in ("/rtx/pathtracing/spp",
                             "/rtx/pathtracing/totalSpp",
                             "/rtx/pathtracing/clampSpp"):
                    s.set_int(path, spp)
                s.set_int("/rtx/pathtracing/maxSamplesPerLaunch", 16)
                s.set_int("/rtx/pathtracing/maxBounces", 4)
                # Both denoisers stay off (container's OptiX install can't
                # load weights and the post-denoiser routes through the
                # same OptiX backend in this Kit build).
                for path in ("/rtx/pathtracing/optixDenoiser/enabled",
                             "/rtx/post/denoising/enabled"):
                    s.set_bool(path, False)
            print(f"[get_app] /rtx/rendermode = {s.get_as_string('/rtx/rendermode')!r}")
            if render_mode == "PathTracing":
                print(f"[get_app] /rtx/pathtracing/spp = {s.get_as_int('/rtx/pathtracing/spp')}")
        except Exception as e:
            print(f"[get_app] Warning: failed to set runtime carb settings: {e}")
    return _app


def close_or_exit(env=None, timeout: float = 10.0):
    """Try to close env+Kit cleanly; force-exit if Kit's teardown hangs.

    Kit's plugin shutdown (especially Replicator annotators) is known to hang
    in headless containers. By the time this is called, torch.save / PIL.save
    have flushed data to the OS page cache, so a force-exit is safe.
    """
    import threading
    import time

    def _kill():
        time.sleep(timeout)
        os._exit(0)

    threading.Thread(target=_kill, daemon=True).start()
    if env is not None:
        try:
            env.close()
        except Exception:
            pass
    os._exit(0)
