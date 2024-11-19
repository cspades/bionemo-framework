from rdkit import Chem
from typing import List
import pandas as pd
from pathlib import Path
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
from rdkit.Chem import rdMolDescriptors, Mol, Atom
from rdkit.Chem.rdchem import ChiralType, HybridizationType
from bionemo.geometric.base_featurizer import BaseFeaturizer, one_hot_enc, get_boolean_atomic_prop, get_double_atomic_prop

# Extremum for molar refractivity
MIN_MR = 0.0
MAX_MR = 6.0

# Extremum for logP
MIN_LOGP = -2.996
MAX_LOGP = 0.8857

ALL_ATOM_FEATURIZERS = ["PeriodicTableFeaturizer", "ElectronicPropertyFeaturizer", "ScaffoldFeaturizer",
                        "SmartsFeaturizer", "CrippenFeaturizer"]

MAX_ATOMIC_NUM = 100
MAX_CHIRAL_TYPES = len(ChiralType.values)
MAX_HYBRIDIZATION_TYPES = len(HybridizationType.values)
MAX_NUM_HS = 5 # 4 + 1 (no hydrogens)

class AtomicNumberFeaturizer(BaseFeaturizer):
    def __init__(self) -> None:
        pass

    @property
    def n_dim(self) -> int:
        return MAX_ATOMIC_NUM

    def compute_features(self, mol: Mol) -> Mol:
        return mol

    def get_features(self, atom: Atom) -> List[bool]:
        return one_hot_enc(atom.GetAtomicNum()-1, MAX_ATOMIC_NUM)


class DegreeFeaturizer(BaseFeaturizer):
    def __init__(self) -> None:
        pass

    @property
    def n_dim(self) -> int:
        return 6

    def compute_features(self, mol: Mol) -> Mol:
        return mol

    def get_features(self, atom: Atom) -> List[bool]:
        return one_hot_enc(atom.GetDegree(), self.n_dim)


class TotalDegreeFeaturizer(BaseFeaturizer):
    def __init__(self) -> None:
        pass

    @property
    def n_dim(self) -> int:
        return 6

    def compute_features(self, mol: Mol) -> Mol:
        return mol

    def get_features(self, atom: Atom) -> List[bool]:
        return one_hot_enc(atom.GetTotalDegree(), self.n_dim)


class ChiralTypeFeaturizer(BaseFeaturizer):
    def __init__(self) -> None:
        pass

    @property
    def n_dim(self) -> int:
        return MAX_CHIRAL_TYPES

    def compute_features(self, mol: Mol) -> Mol:
        return mol

    def get_features(self, atom: Atom) -> List[bool]:
        return one_hot_enc(int(atom.GetChiralTag()), MAX_CHIRAL_TYPES)


class TotalNumHFeaturizer(BaseFeaturizer):
    def __init__(self) -> None:
        pass

    @property
    def n_dim(self) -> int:
        return MAX_NUM_HS

    def compute_features(self, mol: Mol) -> Mol:
        return mol

    def get_features(self, atom: Atom) -> List[bool]:
        return one_hot_enc(atom.GetTotalNumHs(), self.n_dim)


class HybridizationFeaturizer(BaseFeaturizer):
    def __init__(self) -> None:
        pass

    @property
    def n_dim(self) -> int:
        return MAX_HYBRIDIZATION_TYPES

    def compute_features(self, mol: Mol) -> Mol:
        return mol

    def get_features(self, atom: Atom) -> List[bool]:
        return one_hot_enc(atom.GetHybridization(), self.n_dim)


class AromaticityFeaturizer(BaseFeaturizer):
    def __init__(self) -> None:
        pass

    @property
    def n_dim(self) -> int:
        return 1

    def compute_features(self, mol: Mol) -> Mol:
        return mol

    def get_features(self, atom: Atom) -> List[bool]:
        return [atom.GetIsAromatic()]
        

class PeriodicTableFeaturizer(BaseFeaturizer):

    def __init__(self) -> None:
        self.pt = Chem.GetPeriodicTable()

    @property
    def n_dim(self) -> int:
        return 25
        
    def get_period(self, atom: Chem.Atom) -> List[bool]:
        atomic_number = atom.GetAtomicNum()
        # The number of elements per period in the periodic table
        period_limits = [2, 10, 18, 36, 54, 86, 118]
        
        # Determine the period based on atomic number
        for period, limit in enumerate(period_limits, start=1):
            if atomic_number <= limit:
                return one_hot_enc(period-1, 7)
        return None

    def get_group(self, atom: Chem.Atom) -> List[bool]:
        group = self.pt.GetNOuterElecs(atom.GetAtomicNum())
        return one_hot_enc(group - 1, 15)

    def get_atomic_radii(self, atom: Chem.Atom) -> List[float]:
        """Computes bond radius, covalent radius, and van der Waals radius
         and normalizes them to range within 0 and 1"""
        atomic_num = atom.GetAtomicNum()
        Rb0 = self.pt.GetRb0(atomic_num)
        Rb0 = Rb0 / 2.4 # Standard scaler

        Rcovalent = self.pt.GetRcovalent(atomic_num)
        Rcovalent = (Rcovalent - 0.28) / (2.6 - 0.28) # Standard scaler

        Rvdw = self.pt.GetRvdw(atomic_num)
        Rvdw = (Rvdw - 1.2) / (3.0 - 1.2)

        return [Rb0, Rcovalent, Rvdw]
    
    def compute_features(self, mol: Chem.Mol) -> Chem.Mol:
        return mol
    
    def get_features(self, atom: Chem.Atom) -> List[float]:
        return (self.get_period(atom) + self.get_group(atom) + self.get_atomic_radii(atom))
    

class ElectronicPropertyFeaturizer(BaseFeaturizer):
    def __init__(self, data_file=None) -> None:
        if data_file is None:
            # Use default
            root_path = Path(__file__).resolve().parent
            data_file = root_path / "data" / "electronic_data.csv"
            print(f"{data_file}")
        self.data_df = pd.read_csv(data_file).set_index("AtomicNumber")

        self.pauling_en_dict = self.data_df["Electronegativity"].to_dict()
        self.ie_dict = self.data_df["IonizationEnergy"].to_dict()
        self.ea_dict = self.data_df["ElectronAffinity"].to_dict()

    @property
    def n_dim(self) -> int:
        return 3

    def compute_features(self, mol) -> Chem.Mol:
        return mol

    def get_pauling_electronegativity(self, atom: Chem.Atom) -> List[float]:
        atomic_num = atom.GetAtomicNum()
        en = self.pauling_en_dict[atomic_num]
        en = (en - 0.7) / (3.98-0.7)
        return [en]
    
    def get_ie(self, atom: Chem.Atom) -> List[float]:
        atomic_num = atom.GetAtomicNum()
        ie = self.ie_dict[atomic_num]
        ie = (ie - 3.894) / (24.587 - 3.894)
        return [ie]

    def get_ea(self, atom: Chem.Atom) -> List[float]:
        atomic_num = atom.GetAtomicNum()
        ea = self.ea_dict[atomic_num]
        ea = (ea - 0.079) / (3.617 - 0.079)
        return [ea]
    
    def get_features(self, atom: Chem.Atom) -> List[float]:
        return (self.get_pauling_electronegativity(atom) +
                self.get_ie(atom) + 
                self.get_ea(atom))
    

class ScaffoldFeaturizer(BaseFeaturizer):
    def __init__(self):
        pass
    
    @property
    def n_dim(self) -> int:
        return 1

    def compute_features(self, mol):
        """Computes if atom is part of scaffold and sets it as
        boolean atomic property with key 'InScaffold'"""
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_atom_idx = set(mol.GetSubstructMatch(scaffold))
        
        # Set everything to False first
        for atom in mol.GetAtoms():
            atom.SetBoolProp("InScaffold", False)

        # Set only scaffold atoms to True
        for idx in scaffold_atom_idx:
            iatom = mol.GetAtomWithIdx(idx)
            iatom.SetBoolProp("InScaffold", True)

        return mol
    
    def get_features(self, atom) -> List[bool]:
        return get_boolean_atomic_prop(atom, prop_list=["InScaffold"])
    

class SmartsFeaturizer(BaseFeaturizer):
    def __init__(self):
        self.hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
        self.hydrogen_acceptor = Chem.MolFromSmarts(
            "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),"
            "n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
        self.acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
        self.basic = Chem.MolFromSmarts(
            "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);"
            "!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")
        
    @property
    def n_dim(self) -> int:
        return 4

    def compute_features(self, mol):
        hydrogen_donor_match = sum(mol.GetSubstructMatches(self.hydrogen_donor), ())
        hydrogen_acceptor_match = sum(mol.GetSubstructMatches(self.hydrogen_acceptor), ())
        acidic_match = sum(mol.GetSubstructMatches(self.acidic), ())
        basic_match = sum(mol.GetSubstructMatches(self.basic), ())
        
        # Set everything to False by default
        for atom in mol.GetAtoms():
            atom.SetBoolProp("hba", False)
            atom.SetBoolProp("hbd", False)
            atom.SetBoolProp("acidic", False)
            atom.SetBoolProp("basic", False)

        # Set matching substructures to True
        for idx in hydrogen_acceptor_match:
            mol.GetAtomWithIdx(idx).SetBoolProp("hba", True)
            
        for idx in hydrogen_donor_match:
            mol.GetAtomWithIdx(idx).SetBoolProp("hbd", True)
            
        for idx in acidic_match:
            mol.GetAtomWithIdx(idx).SetBoolProp("acidic", True)
        
        for idx in basic_match:
            mol.GetAtomWithIdx(idx).SetBoolProp("basic", True)
    
        return mol
    
    def get_features(self, atom):
        return get_boolean_atomic_prop(atom, prop_list=["hba", "hbd", "acidic", "basic"])
    

class CrippenFeaturizer(BaseFeaturizer):
    def __init__(self):
        pass

    @property
    def n_dim(self) -> int:
        return 2
    
    def compute_features(self, mol):
        logp_mr_list = np.array(rdMolDescriptors._CalcCrippenContribs(mol))
        logp_mr_list[:, 0] = np.clip(logp_mr_list[:, 0], a_min=MIN_LOGP, a_max=MAX_LOGP)
        logp_mr_list[:, 1] = np.clip(logp_mr_list[:, 1], a_min=MIN_MR, a_max=MAX_MR)
        
        logp_mr_list[:, 0] = (logp_mr_list[:, 0] - MIN_LOGP) / (MAX_LOGP - MIN_LOGP)
        logp_mr_list[:, 1] = (logp_mr_list[:, 1] - MIN_MR) / (MAX_MR - MIN_MR)
        
        for iatom, atom in enumerate(mol.GetAtoms()):
            atom.SetDoubleProp("crippen_logp", logp_mr_list[iatom][0])
            atom.SetDoubleProp("crippen_mr", logp_mr_list[iatom][1])
            
        return mol

    def get_features(self, atom):
        return get_double_atomic_prop(atom, prop_list=["crippen_logp", "crippen_mr"])

# TODO Implement more features
## - Size of ring atom is present in
## - 2D partial charges like Gasteiger charges
