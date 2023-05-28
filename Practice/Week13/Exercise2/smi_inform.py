import pickle
from rdkit import Chem
import torch
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
class smi_inform(object):
    def __init__(self,radius=2, nBits=1024):
        self.radius = radius
        self.nBits = nBits
    def get_mol(self,smi):
        mol = Chem.MolFromSmiles(smi)
        return mol
    def get_prop(self,mol):
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol,self.radius,self.nBits,useChirality=True)
        self.fp = torch.Tensor(fp1)
        self.TPSA = Descriptors.TPSA(mol)
        self.logp = Descriptors.MolLogP(mol)
        self.rings = Descriptors.RingCount(mol)
        return self
        
            
    