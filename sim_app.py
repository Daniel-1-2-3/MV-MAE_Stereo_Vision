"""Launching Isaac Sim.

Isaac Sim must be started before anything imports Isaac Lab's simulation
modules (surgical_env.env, surgical_env.env_cfg, ...). Every script therefore
calls ``launch(parser)`` first and imports those modules afterwards.

Cameras need the rendering "experience", so ``--enable_cameras`` is always on,
and scripts run headless (no window, as on a cloud pod) unless ``--gui`` is given.
"""

from __future__ import annotations

import argparse


def launch(parser: argparse.ArgumentParser):
    from isaaclab.app import AppLauncher

    parser.add_argument("--gui", action="store_true", help="open the Isaac Sim window instead of running headless")
    AppLauncher.add_app_launcher_args(parser)
    args, overrides = parser.parse_known_args()
    args.enable_cameras = True
    if not args.gui:
        args.headless = True
    app_launcher = AppLauncher(args)
    return app_launcher.app, args, overrides
