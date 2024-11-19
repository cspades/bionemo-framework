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


import pytest
from rdkit import Chem

from bionemo.geometric.bond_featurizers import RingFeaturizer


@pytest.fixture(scope="module")
def test_mol2():
    return Chem.MolFromSmiles("C[C@H]1CN(c2ncnc3[nH]cc(-c4cccc(F)c4)c23)CCO1")  # CHEMBL3927167


def test_ring_featurizer(test_mol2):
    rf = RingFeaturizer()
    feat_mol = rf.compute_features(test_mol2)

    bidx0_feats = rf.get_features(feat_mol.GetBondWithIdx(0))
    bidx0_feats_ref = [False, False, False, False, False, False, False]
    assert bidx0_feats == bidx0_feats_ref

    bidx1_feats = rf.get_features(feat_mol.GetBondWithIdx(1))
    bidx1_feats_ref = [False, False, False, True, False, False, False]
    assert bidx1_feats == bidx1_feats_ref

    bidx24_feats = rf.get_features(feat_mol.GetBondWithIdx(24))
    bidx24_feats_ref = [False, False, True, True, False, False, False]
    assert bidx24_feats == bidx24_feats_ref
