"""Lazy, idempotent SimulationApp boot.

Must run before any `isaaclab.*` or `isaaclab_tasks.*` import. Call
`get_app()` first, then import everything else.

The `renderer` arg lets you swap Omniverse's render delegate. Default RTX
needs driver >= 535.129; pass `renderer="pxr"` to try Pixar's Hydra Storm
(OpenGL-based, no RTX driver requirement) when the host driver is too old.
"""

_app = None


def get_app(headless: bool = True, enable_cameras: bool = True, renderer: str | None = None):
    global _app
    if _app is None:
        from isaaclab.app import AppLauncher

        # Kit decodes NVIDIA's Vulkan driverVersion with an 8-bit minor field;
        # drivers with minor >= 256 (e.g. 535.261.03) overflow to e.g. "535.5"
        # and then fail the RTX driver check against the 535.129 floor.
        kit_args_parts = ["--/rtx/verifyDriverVersion/enabled=false"]
        if renderer:
            kit_args_parts.append(f"--/renderer/enabled={renderer} --/renderer/active={renderer}")

        _launcher = AppLauncher(
            headless=headless,
            enable_cameras=enable_cameras,
            kit_args=" ".join(kit_args_parts),
        )
        _app = _launcher.app
    return _app
