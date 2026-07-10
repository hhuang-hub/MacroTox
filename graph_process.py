import torch
from rdkit import Chem
import random
from rdkit.Chem import AllChem, rdMolTransforms
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import os
import yaml
import argparse

def read_csv(file,smi,label):
    df=pd.read_csv(file)
    smis=df[smi].tolist()
    labels=df[label].tolist()
    return smis,labels


class Graph_custom(InMemoryDataset):
    def __init__(self, dataframe, root, smiles_col='smiles', label_col='label', test=False, transform=None, pre_transform=None):  # modified
        self.test = test
        self.dataframe = dataframe
        self.smiles_col = smiles_col
        self.label_col = label_col
        self._data = None
        self.error_indices = []  # bad indices: To keep track of error indices
        super(Graph_custom, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """
        (The download func. is not implemented here)  
        """
        return 'dataframe'

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        return ['data_test.pt' if self.test else 'data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        for index, mol in tqdm(self.dataframe.iterrows(), total=self.dataframe.shape[0]):
            try:
                mol_obj = Chem.MolFromSmiles(mol[self.smiles_col])
                mol_obj = Chem.AddHs(mol_obj)
                AllChem.EmbedMolecule(mol_obj, randomSeed=0, 
                                      useRandomCoords = True, maxAttempts = 5000 # Use this when Bad Conformer ID error
                                     )
                AllChem.MMFFOptimizeMolecule(mol_obj)
                mol_obj = Chem.RemoveHs(mol_obj)
                AllChem.ComputeGasteigerCharges(mol_obj)
                
                ################################################################
                node_feats = self._get_node_features(mol_obj)
                edge_feats = self._get_edge_features(mol_obj)
                edge_index = self._get_adjacency_info(mol_obj)
                label = self._get_labels(mol[self.label_col])

                data = Data(x=node_feats, 
                            edge_index=edge_index,
                            edge_attr=edge_feats,
                            y=label,
                            smiles=mol[self.smiles_col]
                            ) 
                data_list.append(data)
            except Exception as e:
                print(f"Error processing molecule at index {index}: {e}")
                self.error_indices.append(index)  # bad_indices: Track the index of the molecule that caused an error

        if self.test:
            torch.save(data_list, os.path.join(self.processed_dir, 'data_test.pt'))
        else:
            torch.save(data_list, os.path.join(self.processed_dir, 'data.pt'))

    def _get_node_features(self, mol):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        all_node_feats = []

        for atom in mol.GetAtoms():
            node_feats = []
            # Feature 1: Atomic number        
            node_feats.append(atom.GetAtomicNum())
            # Feature 2: Atom degree -> Number of directly-bonded neighbours
            node_feats.append(atom.GetDegree())
            # Feature 3: Formal charge -> charge of the atom
            node_feats.append(atom.GetFormalCharge())
            # Feature 4: Hybridization -> hybridization state i.e. sp3
            node_feats.append(atom.GetHybridization())
            # Feature 5: Aromaticity
            node_feats.append(atom.GetIsAromatic())
            # Feature 6: Total Num Hs
            node_feats.append(atom.GetTotalNumHs())
            # Feature 7: Radical Electrons
            node_feats.append(atom.GetNumRadicalElectrons())
            # Feature 8: In Ring
            node_feats.append(atom.IsInRing())
            # Feature 9: Chirality
            node_feats.append(atom.GetChiralTag())

            # Feature 10: Gasteiger Charges
            node_feats.append(atom.GetDoubleProp("_GasteigerCharge"))
            # Feature 11: Total Valence
            node_feats.append(atom.GetTotalValence())
            # Feature 12: Explicit Valence
            node_feats.append(atom.GetExplicitValence())

            # Append node features to matrix
            all_node_feats.append(node_feats)

        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_edge_features(self, mol):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        conf = mol.GetConformer() # will be used to calculate bond length, force field optimised conformer
        all_edge_feats = []
        
        for bond in mol.GetBonds():
            edge_feats = []
            # Feature 1: Bond type (as double)
            edge_feats.append(bond.GetBondTypeAsDouble())
            # Feature 2: Bond length
            edge_feats.append(np.round(rdMolTransforms.GetBondLength(conf, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),3))
            # Feature 3: Rings
            edge_feats.append(bond.IsInRing())
            # Append node features to matrix (twice, per direction)
            all_edge_feats += [edge_feats, edge_feats]

        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_adjacency_info(self, mol):
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices

    def _get_labels(self, label):
        label = np.asarray([[label]])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.dataframe.shape[0]

    def get(self, idx):
        if self._data is None: 
            if self.test:
                self._data = torch.load(os.path.join(self.processed_dir, 'data_test.pt'),weights_only=False)
            else:
                self._data = torch.load(os.path.join(self.processed_dir, 'data.pt'),weights_only=False)
        return self._data[idx]

def graph_compute(data_root,smi,label,graph_root):
    smis,labels=read_csv(data_root,smi,label)
    df=pd.DataFrame({'smiles':smis,'labels':labels})
    graph=Graph_custom(df,graph_root,smiles_col='smiles', label_col='labels')


def set_ls(des):
    l=[]
    for i,d in enumerate(des):
        l.append(d)
    return l
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='path to the config file')
    args = parser.parse_args()
    cfg = load_config(args.config)
    root_test=cfg['paths']['graph_path']
    seed=cfg['settings']['seed']
    set_seed(seed)
    graph_compute(cfg['paths']['csv_path'],cfg['data']['smiles_col'],cfg['data']['target_col'],root_test)