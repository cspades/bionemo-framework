# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Sequence

from bionemo.core.data.load import default_ngc_client, default_pbss_client, entrypoint, load


_ = entrypoint
# This needs to be around so that ruff doesn't automatically remove it as it's unused.
# We don't want to include it in __all__.
# But older installations __may__ be using the old CLI path (bionemo.core.data.load:entrypoint)
# so this is here for backwards compatability.


__all__: Sequence[str] = (
    "default_ngc_client",
    "default_pbss_client",
    "load",
)
