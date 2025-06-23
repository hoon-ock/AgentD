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

# Adapted from the official BAPULM implementation:
# https://github.com/radh55sh/BAPULM/tree/main
# Reference: "BAPULM: Binding Affinity Prediction Using Protein and Molecule Language Models" (https://arxiv.org/abs/2411.04150)

# Set random seeds for reproducibility
def set_seed(seed):
    """
    Set random seeds for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



class BindingAffinityDataset(Dataset):
    """
    Custom PyTorch Dataset for binding affinity data.
    Expects a pandas DataFrame with 'seq' and 'smiles' columns.
    """
    def __init__(self, data):
        self.data = data
        set_seed(2102)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        protein_seq = item['seq']
        ligand_smiles = item['smiles']  # SMILES string for the ligand
        # log_affinity = item['neg_log10_affinity_M']  # Uncomment if affinity is needed
        return protein_seq, ligand_smiles  # , log_affinity

class BAPULM(nn.Module):
    """
    Neural network model for predicting binding affinity using protein and molecule embeddings.
    """
    def __init__(self):
        super(BAPULM, self).__init__()
        # Linear layers for protein and molecule embeddings
        self.prot_linear = nn.Linear(1024, 512)
        self.mol_linear = nn.Linear(768, 512)
        # Batch normalization for concatenated embeddings
        self.norm = nn.BatchNorm1d(1024, eps=0.001, momentum=0.1, affine=True)
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.1)
        # Additional linear layers for feature transformation
        self.linear1 = nn.Linear(1024, 768)
        self.dropout = nn.Dropout(p=0.1)
        self.linear2 = nn.Linear(768, 512)
        self.linear3 = nn.Linear(512, 32)
        self.final_linear = nn.Linear(32, 1)

    def forward(self, prot, mol):
        """
        Forward pass for the model.
        Args:
            prot: Protein embedding tensor
            mol: Molecule embedding tensor
        Returns:
            Predicted binding affinity
        """
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
    """
    Utility class to extract protein and molecule embeddings using pretrained models.
    """
    def __init__(self, device):
        # Set device (GPU if available, else CPU)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load protein tokenizer and encoder model
        self.prot_tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
        self.prot_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)
        
        # Load molecule tokenizer and encoder model
        self.mol_tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
        self.mol_model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True).to(self.device)

    def get_protein_embedding(self, sequence):
        """
        Get protein embedding from sequence using ProtT5 model.
        Args:
            sequence: Protein sequence string
        Returns:
            Embedding tensor
        """
        tokens = self.prot_tokenizer(sequence, return_tensors='pt', padding=True, truncation=True, max_length=3200).to(self.device)
        with torch.no_grad():
            embedding = self.prot_model(**tokens).last_hidden_state.mean(dim=1)
        return embedding
    
    def get_molecule_embedding(self, smiles):
        """
        Get molecule embedding from SMILES using MoLFormer model.
        Args:
            smiles: SMILES string
        Returns:
            Embedding tensor
        """
        tokens = self.mol_tokenizer(smiles, return_tensors='pt', padding=True, truncation=True, max_length=278).to(self.device)
        with torch.no_grad():
            embedding = self.mol_model(**tokens).last_hidden_state.mean(dim=1)
        return embedding

    def get_combined_embedding(self, sequence, smiles):
        """
        Get both protein and molecule embeddings.
        Args:
            sequence: Protein sequence string
            smiles: SMILES string
        Returns:
            Tuple of (protein_embedding, molecule_embedding)
        """
        prot_embedding = self.get_protein_embedding(sequence)
        mol_embedding = self.get_molecule_embedding(smiles)
        return prot_embedding, mol_embedding

def preprocess_function(df):
    """
    Preprocess protein sequences in the DataFrame by replacing uncommon amino acids
    (U, Z, O, B) with 'X' and adding spaces between residues.
    Args:
        df: pandas DataFrame with a 'seq' column
    Returns:
        Modified DataFrame
    """
    df['seq'] = df['seq'].apply(lambda x: " ".join(re.sub(r"[UZOB]", "X", x)))
    return df
