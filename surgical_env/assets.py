# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# From Isaac for Healthcare v0.4.0 (workflows/robotic_surgery/scripts/simulation/utils/assets.py),
# reduced to the assets this project uses.

"""Asset paths from the Isaac for Healthcare asset catalog (i4h-asset-catalog v0.3.0).

Reading an attribute downloads the asset (once) into ~/.cache/i4h-assets from the
public NVIDIA S3 bucket and returns the local path.
"""

from i4h_asset_helper import BaseI4HAssets


class Assets(BaseI4HAssets):
    dVRK_PSM = "Robots/dVRK/PSM/psm.usd"
    Needle = "Props/SutureNeedle/needle.usd"
    Needle_SDF = "Props/SutureNeedle/needle_sdf.usd"


robotic_surgery_assets = Assets()
