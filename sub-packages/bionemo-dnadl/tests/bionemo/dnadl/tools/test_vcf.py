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


import pandas as pd
import pytest

from bionemo.dnadl.tools.genome_interval import GenomeInterval
from bionemo.dnadl.tools.vcf import read_variants_in_interval


@pytest.fixture
def interval():
    return GenomeInterval("chr1", 1, 10)


def test_read_variants_in_interval(temp_vcf_file, interval):
    sample_to_alleles = read_variants_in_interval(interval, temp_vcf_file, ["sample_1"])

    assert sample_to_alleles.keys() == {"sample_1"}
    assert sample_to_alleles["sample_1"].equals(pd.DataFrame([{"POS": 5, "REF": "A", "ALT": "G"}]))
