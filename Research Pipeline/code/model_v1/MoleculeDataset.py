import os
import time
import warnings
warnings.filterwarnings('ignore')

import chemprop
from chemprop import data, featurizers, models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, AdamW, get_linear_schedule_with_warmup , BertModel


class MoleculeDataset(Dataset):
    def __init__(self, ds):
        self.data = ds
        self.features = self.data.drop(columns=['smiles', 'label']).astype(np.float32)

        # necessary features for ChemBERTa model
        self.smiles_list = self.data['smiles'].tolist()
        self.tokenizer = RobertaTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
        self.encoded_smiles = self.tokenizer(self.smiles_list, truncation=True, padding=True, return_tensors="pt")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        smiles = self.data.iloc[idx, 0]
        label = np.array(self.data.iloc[idx, 1], dtype=np.float32)  
        features = np.array(self.features.iloc[idx].values, dtype=np.float32)

        input_ids = self.encoded_smiles["input_ids"][idx]
        attention_mask = self.encoded_smiles["attention_mask"][idx]
        
        return (smiles, features, input_ids, attention_mask, label)