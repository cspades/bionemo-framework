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


from abc import ABC, abstractmethod, abstractproperty
from typing import List


class BaseFeaturizer(ABC):
    @abstractproperty
    def n_dim(self):
        """Number of dimensions of compute feature"""
        pass

    @abstractmethod
    def compute_features(self):
        """Implement this if precomputation of features is needed"""
        pass

    @abstractmethod
    def get_features(self):
        """Function for getting features"""
        pass


def one_hot_enc(val: int, num_class: int) -> List[bool]:
    one_hot = [False] * num_class
    one_hot[val] = True
    return one_hot


def get_boolean_atomic_prop(atom, prop_list=None):
    if prop_list is not None:
        _prop_list = prop_list
    else:
        _prop_list = atom.GetPropNames()

    return [atom.GetBoolProp(prop) for prop in _prop_list]


def get_double_atomic_prop(atom, prop_list=None):
    if prop_list is not None:
        _prop_list = prop_list
    else:
        _prop_list = atom.GetPropNames()

    return [atom.GetDoubleProp(prop) for prop in _prop_list]
