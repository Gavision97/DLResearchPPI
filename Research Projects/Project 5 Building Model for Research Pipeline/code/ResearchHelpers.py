##### Research helpers (functions) #####

from MoleculeDatasets import *
from ResearchModels import *

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
from torch import Tensor

from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, AdamW, get_linear_schedule_with_warmup , BertModel

import pandas as pd
import numpy as np


# Utility functions for printing
def PRINT() -> None: print(f"{'-'*80}\nDone\n{'-'*80}")
def PRINTC() -> None: print(f"{'-'*80}")
def PRINTM(M) -> None: print(f"{'-'*80}\n{M}\n{'-'*80}")


# Verify GPU Availability
def check_device():
    if torch.cuda.is_available():
        PRINTM("GPU is available.")
        return "cuda"
    else:
        PRINTM("GPU is not available. Using CPU instead.")
        return "cpu"

# Data processing functions
def merge_datasets(dataset, features_df):
    dataset = dataset.merge(features_df, how='left', left_on='uniprot_id1', right_on='UniProt_ID', suffixes=('', '_id1'))
    dataset.drop(columns=['UniProt_ID'], inplace=True)

    features_df_renamed = features_df.add_suffix('_id2')
    features_df_renamed.rename(columns={'UniProt_ID_id2': 'UniProt_ID'}, inplace=True)
    dataset = dataset.merge(features_df_renamed, how='left', left_on='uniprot_id2', right_on='UniProt_ID', suffixes=('', '_id2'))
    dataset.drop(columns=['UniProt_ID', 'uniprot_id1', 'uniprot_id2'], inplace=True)
    
    return dataset.drop_duplicates()

def convert_uniprot_ids(dataset, mapping_df):
    mapping_dict = mapping_df.set_index('From')['Entry'].to_dict()
    dataset['uniprot_id1'] = dataset['uniprot_id1'].map(mapping_dict)
    dataset['uniprot_id2'] = dataset['uniprot_id2'].map(mapping_dict)
    return dataset.drop_duplicates()


def data_augmentation_with_uniprots_order_switchings(df):
    # generate a copy of the DataFrame with swapped uniprot_id1 and uniprot_id2
    swapped_df = df.copy()
    swapped_df[['uniprot_id1', 'uniprot_id2']] = swapped_df[['uniprot_id2', 'uniprot_id1']]

    # concatenate the original and swapped DataFrames & drop duplicated samples
    combined_df = pd.concat([df, swapped_df])
    combined_df = combined_df.drop_duplicates()

    return combined_df

def generate_dataloaders(train_data, val_data,
                         test_data, batch_size, shuffle):
    train_dataset = MoleculeDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    val_dataset = MoleculeDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    
    test_dataset = MoleculeDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return (train_loader, val_loader, test_loader)

def generate_model(checkpoints_path,
                   batch_size,
                  dropout) -> nn.Module:
    pretrained_chemprop_model = PretrainedChempropModel(checkpoints_path, batch_size)
    chemberta_model = ChemBERTaPT()
    device = check_device()
    ft_model = AUVG_PPI(pretrained_chemprop_model, chemberta_model, dropout).to(device)

    PRINTM('Generated combined model for fine-tuning successfully !')
    return ft_model