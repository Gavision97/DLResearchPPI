import os
import time
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


class AUVG_PPI(AbstractModel):
    def __init__(self, pretrained_chemprop_model, chemberta_model, dropout):
        super(AUVG_PPI, self).__init__()
        self.pretrained_chemprop_model = pretrained_chemprop_model
        self.chemberta_model = chemberta_model
        self.dropout = dropout
        
        # MLP for ppi_features
        self.ppi_mlp = nn.Sequential(
            nn.Linear(in_features=6558 + 6558, out_features=2048),
            nn.ReLU(),
            #nn.BatchNorm1d(2048),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            #nn.BatchNorm1d(1024),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            #nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=512, out_features=320)
        )

        self.fp_mlp = nn.Sequential(
            nn.Linear(in_features=2100, out_features=1050),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=1050, out_features=600), 
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=600, out_features=300)
        )

        # Additional layrs in order to concatinate chemprop fingerprints, chemBERTa embeddings & ppi features all together
        self.additional_layers = nn.Sequential(
            nn.Linear(in_features=300 + 384 + 320, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=128, out_features=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, bmg, proteins, input_ids, attention_mask):
        # Forward pass batch mol graph through pretrained chemprop model in order to get fingerprints embeddings
        # Afterwards, pass the fingerprints through MLP layer
        fingerprints = self.pretrained_chemprop_model(bmg)
        fingerprints = self.fp_mlp(fingerprints)

        # Forward pass ids & attention mask in through chemBERTa pretrained model in order to get embeddings
        chemberta_embeddings = self.chemberta_model(input_ids, attention_mask)

        # Move PPI features to device and then pass them through MLP layer
        ppi_features = proteins.to(device)
        ppi_features = self.ppi_mlp(ppi_features)

        # Concatinate chemprop fingerprints embeddings, chemberta embeddings and PPI embeddings together into one tensor
        # Afterwards, pass them through MLP layer and make prediction
        combined_embeddings = torch.cat([fingerprints, chemberta_embeddings, ppi_features], dim=1).to(device)
        output = self.additional_layers(combined_embeddings)
        output = self.sigmoid(output)
        return output