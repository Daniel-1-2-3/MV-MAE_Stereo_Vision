# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# From Isaac for Healthcare v0.4.0 (workflows/robotic_surgery/scripts/simulation/utils/assets.py),
# reduced to the assets this project uses.

"""Asset paths from the Isaac for Healthcare asset catalog (i4h-asset-catalog v0.3.0).

Reading an attribute downloads the file (once) into ~/.cache/i4h-assets from the public
NVIDIA S3 bucket and returns the local path. Plain HTTPS GETs are used instead of
i4h_asset_helper: its boto3 folder listing fails with "NoSuchBucket" on some fresh installs
even though the files are publicly reachable. Each USD is self-contained (the catalog
folders hold only these files), so no sibling files are needed.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

_BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
_ROOT = "Assets/Isaac/Healthcare/0.3.0/5de056f4a2f0f49d2b296d51e25cbdfc054c4390712cd0b438cc6d92fa816c1a"
_CACHE = Path(os.environ.get("I4H_ASSET_CACHE", Path.home() / ".cache" / "i4h-assets")) / _ROOT.rsplit("/", 1)[-1]


def _fetch(rel: str) -> str:
    local = _CACHE / rel
    if not local.is_file() or local.stat().st_size == 0:
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_suffix(local.suffix + ".part")
        print(f"[assets] downloading {rel}", flush=True)
        with urllib.request.urlopen(f"{_BUCKET}/{_ROOT}/{rel}", timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(local)
    return str(local)


class Assets:
    _files = {
        "dVRK_PSM": "Robots/dVRK/PSM/psm.usd",
        "Needle": "Props/SutureNeedle/needle.usd",
        "Needle_SDF": "Props/SutureNeedle/needle_sdf.usd",
    }

    def __getattr__(self, name: str) -> str:
        if name not in self._files:
            raise AttributeError(name)
        return _fetch(self._files[name])


robotic_surgery_assets = Assets()
