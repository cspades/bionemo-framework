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
