from rdkit import Chem
from typing import List
from bionemo.geometric.base_featurizer import BaseFeaturizer
ALL_BOND_FEATURIZERS = ["RingFeaturizer"]


class RingFeaturizer(BaseFeaturizer):
    """Multi-label one-hot encoding"""
    def __init__(self, n_ring_sizes=7) -> None:
        self.n_ring_sizes = n_ring_sizes # ring size 3 - 8 and UNK

    @property
    def n_dim(self) -> int:
        return self.n_ring_sizes
    
    def compute_features(self, mol) -> Chem.Mol:
        ri = mol.GetRingInfo()

        for bond in mol.GetBonds():
            bidx = bond.GetIdx()
            bond_ring_sizes = ",".join(map(str, ri.BondRingSizes(bidx)))
            bond.SetProp("bond_ring_sizes", bond_ring_sizes)
        return mol

    def get_features(self, bond: Chem.Bond) -> List[bool]:

        ring_sizes = bond.GetProp("bond_ring_sizes").split(",")
        ring_sizes = [int(r) for r in ring_sizes if r != ""]
        
        feats = [False] * self.n_ring_sizes
        for r in ring_sizes:
            if r > 8:
                feats[-1] = True
            else:
                feats[r-3] = True # ring of size 3 is at first position
        return feats