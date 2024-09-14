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
        
        
class FeatureReducer_(nn.Module):
    # Feature reducer for joint attention in PPI structure feature - in order to reduce tensors dim for math operations
    # Use this class if |UniProt_NumOfAminoAcidComp| < 128
    def __init__(self, in_channels, out_channels):
        super(FeatureReducer_, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        # x shape: [batch_size, sequence_length, in_channels]
        x = x.transpose(1, 2)  # Change shape to [batch_size, in_channels, sequence_length]
        x = self.conv(x)       
        x = x.transpose(1, 2)  # Change shape back to [batch_size, target_length, out_channels]
        return x
        
class FeatureReducer(nn.Module):
    # Feature reducer for joint attention in PPI structure feature - in order to reduce tensors dim for math operations
    # Use this class if |UniProt_NumOfAminoAcidComp| >= 128
    def __init__(self, in_channels, out_channels, target_length):
        super(FeatureReducer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(target_length)
    
    def forward(self, x):
        # x shape: [batch_size, sequence_length, in_channels]
        x = x.transpose(1, 2)  # Change shape to [batch_size, in_channels, sequence_length]
        x = self.conv(x)    
        x = self.pool(x) 
        x = x.transpose(1, 2)  # Change shape back to [batch_size, target_length, out_channels]
        return x
        
        
class custom_self_attention(nn.Module):
    def __init__(self, embed_dim_, num_heads_, dropout_):
        super(custom_self_attention, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim_, num_heads=num_heads_, dropout= dropout_)
        self.norm_layer = nn.LayerNorm(embed_dim_)

    def forward(self, embeddings_mat):
        # Apply self-attention for PPI
        embeddings_mat = embeddings_mat.permute(1, 0, 2)  # Change to (num_heads, batch_size, embed_dim) for MultiheadAttention
        attn_output, attn_weights = self.self_attention(embeddings_mat, embeddings_mat, embeddings_mat)
        attn_output = attn_output.permute(1, 0, 2)  # shape ->> (batch_size, num_heads, embed_dim)

        # Add & Norm
        embeddings_mat = embeddings_mat.permute(1, 0, 2)  # Back to original shape for residual connection
        attn_output = (0.5*attn_output) + (0.5*embeddings_mat)  # Add (residual connection) & apply weighted residual connection 
        attn_output = self.norm_layer(attn_output)  # Apply LayerNorm

        # Flatten the output for the next MLP layer
        embeddings_mat = attn_output.flatten(start_dim=1)  # Shape: (batch_size, num_heads*embed_dim)
        
        return embeddings_mat

def generate_model(checkpoints_path,
                   batch_size,
                  dropout) -> nn.Module:
    pretrained_chemprop_model = PretrainedChempropModel(checkpoints_path, batch_size)
    chemberta_model = ChemBERTaPT()
    device = check_device()
    ft_model = AUVG_PPI(pretrained_chemprop_model, chemberta_model, dropout).to(device)

    PRINTM('Generated combined model for fine-tuning successfully !')
    return ft_model

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

    
