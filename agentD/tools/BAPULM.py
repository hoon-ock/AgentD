
import torch
import torch.nn as nn
import numpy as np
import random
import re
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from transformers import T5Tokenizer, T5EncoderModel
import torch.nn.functional as F
import pandas as pd


# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(2102)

class BindingAffinityDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        protein_seq = item['seq']
        ligand_smiles = item['smiles'] #item['smiles_can']
        # log_affinity = item['neg_log10_affinity_M']
        return protein_seq, ligand_smiles #, log_affinity

class BAPULM(nn.Module):
    def __init__(self):
        super(BAPULM, self).__init__()
        self.prot_linear = nn.Linear(1024, 512)
        self.mol_linear = nn.Linear(768, 512)
        self.norm = nn.BatchNorm1d(1024, eps=0.001, momentum=0.1, affine=True)
        self.dropout = nn.Dropout(p=0.1)
        self.linear1 = nn.Linear(1024, 768)
        self.dropout = nn.Dropout(p=0.1)
        self.linear2 = nn.Linear(768, 512)
        self.linear3 = nn.Linear(512, 32)
        self.final_linear = nn.Linear(32, 1)

    def forward(self, prot,mol):
        prot_output = torch.relu(self.prot_linear(prot))
        mol_output = torch.relu(self.mol_linear(mol))
        combined_output = torch.cat((prot_output, mol_output), dim=1)
        combined_output = self.norm(combined_output)
        combined_output = self.dropout(combined_output)
        x = torch.relu(self.linear1(combined_output))
        x = torch.relu(self.linear2(x))
        x = self.dropout(x)
        x = torch.relu(self.linear3(x))
        output = self.final_linear(x)
        return output

class EmbeddingExtractor:
    def __init__(self, device):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.prot_tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
        self.prot_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)
        
        self.mol_tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
        self.mol_model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True).to(self.device)

    def get_protein_embedding(self, sequence):
        tokens = self.prot_tokenizer(sequence, return_tensors='pt', padding=True, truncation=True, max_length=3200).to(self.device)
        with torch.no_grad():
            embedding = self.prot_model(**tokens).last_hidden_state.mean(dim=1)
        return embedding
    
    def get_molecule_embedding(self, smiles):
        tokens = self.mol_tokenizer(smiles, return_tensors='pt', padding=True, truncation=True, max_length=278).to(self.device)
        with torch.no_grad():
            embedding = self.mol_model(**tokens).last_hidden_state.mean(dim=1)
        return embedding

    def get_combined_embedding(self, sequence, smiles):
        prot_embedding = self.get_protein_embedding(sequence)
        mol_embedding = self.get_molecule_embedding(smiles)
        return prot_embedding, mol_embedding

def preprocess_function(df):
    df['seq'] = df['seq'].apply(lambda x: " ".join(re.sub(r"[UZOB]", "X", x)))
    return df
    