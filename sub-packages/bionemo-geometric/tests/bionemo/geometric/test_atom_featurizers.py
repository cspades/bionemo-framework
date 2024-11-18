import pytest
from rdkit import Chem
from bionemo.geometric.atom_featurizers import (PeriodicTableFeaturizer, 
    ElectronicPropertyFeaturizer,
    ScaffoldFeaturizer,
    SmartsFeaturizer)


@pytest.fixture(scope="module")
def test_mol():
    return Chem.MolFromSmiles("NC(=O)c1cn(-c2ccc(S(N)(=O)=O)cc2)nc1-c1ccc(Cl)cc1") # CHEMBL3126825

@pytest.fixture(scope="module")
def acetic_acid():
    return Chem.MolFromSmiles("CC(=O)O")

@pytest.fixture(scope="module")
def methylamine():
    return Chem.MolFromSmiles("CN")

def test_periodic_table_featurizer(test_mol):
    pt = PeriodicTableFeaturizer()

    n_feats = pt.get_features(test_mol.GetAtomWithIdx(0)) # N
    n_feats_ref = [False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, 0.2916666666666667, 0.18534482758620685, 0.2222222222222223]
    assert len(n_feats) == pt.n_dim, f"Dimension of generated features ({len(n_feats)}) not equal to expected dimension ({pt.n_dim})"
    assert n_feats == n_feats_ref

    c_feats = pt.get_features(test_mol.GetAtomWithIdx(1)) # C
    c_feats_ref = [False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, 0.32083333333333336, 0.2068965517241379, 0.2777777777777778]
    assert c_feats == c_feats_ref

    o_feats = pt.get_features(test_mol.GetAtomWithIdx(2)) # O
    o_feats_ref = [False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, 0.275, 0.16379310344827586, 0.19444444444444448]
    assert o_feats == o_feats_ref

    s_feats = pt.get_features(test_mol.GetAtomWithIdx(10)) # S
    s_feats_ref = [False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, 0.43333333333333335, 0.3318965517241379, 0.33333333333333337]
    assert s_feats == s_feats_ref

    cl_feats = pt.get_features(test_mol.GetAtomWithIdx(22)) # Cl
    cl_feats_ref = [False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, 0.41541666666666666, 0.3189655172413793, 0.33333333333333337]
    assert cl_feats == cl_feats_ref


def test_electronic_property_featurizer(test_mol):
    ep = ElectronicPropertyFeaturizer()

    n_feats = ep.get_features(test_mol.GetAtomWithIdx(0)) # N
    n_feats_ref = [0.7134146341463413, 0.5141835403276471, 0.28070671308004325]
    assert n_feats == n_feats_ref

    c_feats = ep.get_features(test_mol.GetAtomWithIdx(1)) # C
    c_feats_ref = [0.5640243902439024, 0.3559657855313391, 0.33465234595816845]
    assert c_feats == c_feats_ref

    o_feats = ep.get_features(test_mol.GetAtomWithIdx(2)) # O
    o_feats_ref = [0.8353658536585366, 0.4699173633595903, 0.39061616732617305]
    assert o_feats == o_feats_ref

    s_feats = ep.get_features(test_mol.GetAtomWithIdx(10)) # S
    s_feats_ref = [0.573170731707317, 0.31247281689460205, 0.5647258338044093]

    cl_feats = ep.get_features(test_mol.GetAtomWithIdx(22)) # Cl
    cl_feats_ref = [0.7499999999999999, 0.4385057748997246, 1.0]
    assert cl_feats == cl_feats_ref


def test_scaffold_featurizer(test_mol):
    sf = ScaffoldFeaturizer()
    test_mol_featurized = sf.compute_features(test_mol)

    prop_list = []
    for atom in test_mol_featurized.GetAtoms():
        prop_list.append(sf.get_features(atom))
    
    prop_list_ref = [[False], [False], [False], [True], [True], [True], [True], [True], [True], [True], [False], [False], [False], [False], [True], [True], [True], [True], [True], [True], [True], [True], [False], [True], [True]]
    assert prop_list == prop_list_ref

def test_smarts_featurizer(test_mol, acetic_acid, methylamine):
    sf = SmartsFeaturizer()

    test_mol_featurized = sf.compute_features(test_mol)
    test_mol_props = []
    for atom in test_mol_featurized.GetAtoms():
        test_mol_props.append(sf.get_features(atom))
    test_mol_props_ref = [[False, True, False, False], [False, False, False, False], [True, False, False, False], [False, False, False, False], [False, False, False, False], [True, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False], [False, True, False, False], [True, False, False, False], [True, False, False, False], [False, False, False, False], [False, False, False, False], [True, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False]]

    assert test_mol_props == test_mol_props

    # acetic acid
    aa_featurized = sf.compute_features(acetic_acid)
    aa_props = []
    for atom in aa_featurized.GetAtoms():
        aa_props.append(sf.get_features(atom))
    aa_props_ref = [[False, False, False, False], [False, False, True, False], [True, False, False, False], [False, True, False, False]]
    assert aa_props == aa_props_ref

    # methylamine
    ma_featurized = sf.compute_features(methylamine)
    ma_props = []
    for atom in ma_featurized.GetAtoms():
        ma_props.append(sf.get_features(atom))

    ma_props_ref = [[False, False, False, False], [True, True, False, True]]

    assert ma_props == ma_props_ref
