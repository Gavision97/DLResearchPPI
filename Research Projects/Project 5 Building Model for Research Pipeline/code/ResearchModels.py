# All Research Models #
## Under v1 -> Models without attention ##
## Under v2 -> Models with attention ## 

import os
import time
import warnings
warnings.filterwarnings('ignore')

import chemprop
from chemprop import data, featurizers, models
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

# import necessary libraries for Chemberta model
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, AdamW, get_linear_schedule_with_warmup , BertModel

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


##### v1 models #####

# Without Attention & without data augmentation and prob in testin
class AUVG_PPI_v1(AbstractModel):
    def __init__(self, pretrained_chemprop_model, chemberta_model, dropout):
        
        super(AUVG_PPI_v1, self).__init__()
        self.pretrained_chemprop_model = pretrained_chemprop_model
        self.chemberta_model = chemberta_model
        self.dropout = dropout
        
        # PPI Features MLP layers: (esm, custom, fegs, gae)
        self.esm_mlp = nn.Sequential(
            nn.Linear(in_features=1280 + 1280 , out_features=1750),
            nn.ReLU(),
            nn.BatchNorm1d(1750),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=1750, out_features=1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=1000, out_features=750),
            nn.ReLU(),
            nn.BatchNorm1d(750),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=750, out_features=512)
        )

        self.fegs_mlp = nn.Sequential(
            nn.Linear(in_features=578 + 578, out_features=750),
            nn.ReLU(),
            nn.BatchNorm1d(750),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=750, out_features=512)
        )        

        self.custom_mlp = nn.Sequential(
            nn.Linear(in_features=4700 + 4700 , out_features=8000),
            nn.ReLU(),
            nn.BatchNorm1d(8000),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=8000, out_features=6500),
            nn.ReLU(),
            nn.BatchNorm1d(6500),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=6500, out_features=5000),
            nn.ReLU(),
            nn.BatchNorm1d(5000),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=5000, out_features=3500),
            nn.ReLU(),
            nn.BatchNorm1d(3500),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=3500, out_features=2000),
            nn.ReLU(),
            nn.BatchNorm1d(2000),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=2000, out_features=1028),
            nn.ReLU(),
            nn.BatchNorm1d(1028),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=1028, out_features=512)
        )

        self.gae_mlp = nn.Sequential(
            nn.Linear(in_features=500 + 500, out_features=750),
            nn.ReLU(),
            nn.BatchNorm1d(750),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=750, out_features=512)
        )

        # MLP for ppi_features
        self.ppi_mlp = nn.Sequential(
            nn.Linear(in_features=512 * 4 , out_features= 1536),
            nn.ReLU(),
            nn.BatchNorm1d(1536),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=1536, out_features=1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=1024, out_features=512)
        )
        
        self.fp_mlp = nn.Sequential(
            nn.Linear(in_features=2100, out_features=1536),
            nn.ReLU(),
            nn.BatchNorm1d(1536),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=1536, out_features=1024), 
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=512, out_features=384)
        )

        # Morgan fingerprints & chemical descriptors MLP layers
        self.mfp_cd_mlp = nn.Sequential(
            nn.Linear(in_features=1024 + 194, out_features= 750),
            nn.ReLU(),
            nn.BatchNorm1d(750),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=750, out_features=512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=512, out_features=384)
        )

        # MLP for smiles_embeddings
        self.smiles_mlp = nn.Sequential(
            nn.Linear(in_features=384 * 3 , out_features= 750),
            nn.ReLU(),
            nn.BatchNorm1d(750),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=750, out_features=512)
        )

        self.additional_layers = nn.Sequential(
            nn.Linear(in_features=512 + 512, out_features=768),
            nn.ReLU(),
            nn.BatchNorm1d(768),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=768, out_features=512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=64, out_features=1)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, bmg, esm, custom, fegs, gae,
                input_ids, attention_mask,
                morgan_fingerprints, chemical_descriptors):
        # Forward pass batch mol graph through pretrained chemprop model in order to get fingerprints embeddings
        # Afterwards, pass the fingerprints through MLP layer
        cp_fingerprints = self.pretrained_chemprop_model(bmg)
        cp_fingerprints = self.fp_mlp(cp_fingerprints)

        chemberta_embeddings = self.chemberta_model(input_ids, attention_mask)
        #chemberta_embeddings = self.chemberta_mlp(chemberta_embeddings)
        mfp_chem_descriptors = torch.cat([morgan_fingerprints, chemical_descriptors], dim=1)
        mfp_chem_descriptors = self.mfp_cd_mlp(mfp_chem_descriptors)
        
        # Concatenate all 3 smiles embeddings along a new dimension (3x384) 
        smiles_embeddings = torch.cat([cp_fingerprints,chemberta_embeddings, mfp_chem_descriptors], dim=1).to(device)  # shape ->> (batch_size, 3*384)
        smiles_embeddings = self.smiles_mlp(smiles_embeddings)

        # Pass all PPI features  through MLP layers, and then pass them all together into another MLP layer
        #ppi_features = proteins.to(device)
        esm_embeddings = self.esm_mlp(esm)
        custom_embeddings = self.custom_mlp(custom)
        fegs_embeddings = self.fegs_mlp(fegs)
        gae_embeddings = self.gae_mlp(gae)

        # Concatenate all 4 ppi embeddings along a new dimension (4x512) 
        ppi_embeddings = torch.cat([esm_embeddings, custom_embeddings, fegs_embeddings, gae_embeddings], dim=1).to(device)  # shape ->> (batch_size, 4*512)
        ppi_features = self.ppi_mlp(ppi_embeddings)

        combined_embeddings = torch.cat([smiles_embeddings, ppi_features], dim=1)
        output = self.additional_layers(combined_embeddings)
        
        return self.sigmoid(output)




##### v2 models #####

### Helper class for self attention mechanism ###
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

# With Attention (first version)
class AUVG_PPI_v2_1(AbstractModel):
    def __init__(self, pretrained_chemprop_model, chemberta_model, dropout):
        
        super(AUVG_PPI_v2_1, self).__init__()
        self.pretrained_chemprop_model = pretrained_chemprop_model
        self.chemberta_model = chemberta_model
        self.dropout = dropout
        self.ppi_self_attention = custom_self_attention(512, 8, 0.2)
        self.smiles_self_attention = custom_self_attention(384, 4, 0.2)
        self.cross_attention = nn.MultiheadAttention(512, 8, 0.2)
        self.max_pool = nn.MaxPool1d(2)
        
        # PPI Features MLP layers: (esm, custom, fegs, gae)
        self.esm_mlp = nn.Sequential(
            nn.Linear(in_features=1280 + 1280 , out_features=1750),
            nn.ReLU(),
            nn.BatchNorm1d(1750),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=1750, out_features=1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=1000, out_features=750),
            nn.ReLU(),
            nn.BatchNorm1d(750),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=750, out_features=512)
        )

        self.fegs_mlp = nn.Sequential(
            nn.Linear(in_features=578 + 578, out_features=750),
            nn.ReLU(),
            nn.BatchNorm1d(750),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=750, out_features=512)
        )        

        self.custom_mlp = nn.Sequential(
            nn.Linear(in_features=4700 + 4700 , out_features=8000),
            nn.ReLU(),
            nn.BatchNorm1d(8000),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=8000, out_features=6500),
            nn.ReLU(),
            nn.BatchNorm1d(6500),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=6500, out_features=5000),
            nn.ReLU(),
            nn.BatchNorm1d(5000),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=5000, out_features=3500),
            nn.ReLU(),
            nn.BatchNorm1d(3500),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=3500, out_features=2000),
            nn.ReLU(),
            nn.BatchNorm1d(2000),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=2000, out_features=1028),
            nn.ReLU(),
            nn.BatchNorm1d(1028),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=1028, out_features=512)
        )

        self.gae_mlp = nn.Sequential(
            nn.Linear(in_features=500 + 500, out_features=750),
            nn.ReLU(),
            nn.BatchNorm1d(750),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=750, out_features=512)
        )

        # MLP for ppi_features
        self.ppi_mlp = nn.Sequential(
            nn.Linear(in_features=512 * 4 , out_features= 1536),
            nn.ReLU(),
            nn.BatchNorm1d(1536),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=1536, out_features=1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=1024, out_features=512)
        )
        
        self.fp_mlp = nn.Sequential(
            nn.Linear(in_features=2100, out_features=1536),
            nn.ReLU(),
            nn.BatchNorm1d(1536),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=1536, out_features=1024), 
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=512, out_features=384)
        )

        # Morgan fingerprints & chemical descriptors MLP layers
        self.mfp_cd_mlp = nn.Sequential(
            nn.Linear(in_features=1024 + 194, out_features= 750),
            nn.ReLU(),
            nn.BatchNorm1d(750),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=750, out_features=512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=512, out_features=384)
        )

        # MLP for smiles_embeddings
        self.smiles_mlp = nn.Sequential(
            nn.Linear(in_features=384 * 3 , out_features= 750),
            nn.ReLU(),
            nn.BatchNorm1d(750),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=750, out_features=512)
        )

        self.additional_layers = nn.Sequential(
            nn.Linear(in_features=256 + 256, out_features=256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=64, out_features=1)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, bmg, esm, custom, fegs, gae,
                input_ids, attention_mask,
                morgan_fingerprints, chemical_descriptors):
        # Forward pass batch mol graph through pretrained chemprop model in order to get fingerprints embeddings
        # Afterwards, pass the fingerprints through MLP layer
        cp_fingerprints = self.pretrained_chemprop_model(bmg)
        cp_fingerprints = self.fp_mlp(cp_fingerprints)

        chemberta_embeddings = self.chemberta_model(input_ids, attention_mask)
        #chemberta_embeddings = self.chemberta_mlp(chemberta_embeddings)
        mfp_chem_descriptors = torch.cat([morgan_fingerprints, chemical_descriptors], dim=1)
        mfp_chem_descriptors = self.mfp_cd_mlp(mfp_chem_descriptors)
        
        # Concatenate all 3 smiles embeddings along a new dimension (3x384) & pass them throw self-attention layer
        smiles_embeddings = torch.stack([cp_fingerprints, chemberta_embeddings, mfp_chem_descriptors], dim=1).to(device)  # shape ->> (batch_size, 3, 384)
        smiles_features = self.smiles_self_attention(smiles_embeddings)
        smiles_embeddings = self.smiles_mlp(smiles_features).unsqueeze(1)

        # Pass all PPI features  through MLP layers, and then pass them all together into another MLP layer
        #ppi_features = proteins.to(device)
        esm_embeddings = self.esm_mlp(esm)
        custom_embeddings = self.custom_mlp(custom)
        fegs_embeddings = self.fegs_mlp(fegs)
        gae_embeddings = self.gae_mlp(gae)

        # Concatenate all 4 ppi embeddings along a new dimension (4x320) & pass them throw self-attention layer
        ppi_embeddings = torch.stack([esm_embeddings, custom_embeddings, fegs_embeddings, gae_embeddings], dim=1).to(device)  # shape ->> (batch_size, 4, 320)
        ppi_features = self.ppi_self_attention(ppi_embeddings)
        ppi_features = self.ppi_mlp(ppi_features).unsqueeze(1)

        #Cross-attention between smiles and PPI to capture the interaction relationships
        ppi_QKV = ppi_features.permute(1, 0, 2)
        smiles_QKV = smiles_embeddings.permute(1, 0, 2)
        
        smiles_att, _ = self.cross_attention(smiles_QKV, ppi_QKV, ppi_QKV)
        ppi_att, _ = self.cross_attention(ppi_QKV, smiles_QKV, smiles_QKV)

        # permute attention outputrs to match (batch_size, embed_dim, num_heads) shape
        smiles_attn_output = (0.5* smiles_att.permute(1, 2, 0)) + (0.5* smiles_embeddings.permute(0, 2, 1))  # Add (residual connection) & apply weighted residual connection 
        ppi_attn_output = (0.5* ppi_att.permute(1, 2, 0)) + (0.5* ppi_features.permute(0, 2, 1))  # Add (residual connection) & apply weighted residual connection 

        # Drop the last dim in order to get (batch_size, embed_dim) & 
        # Pass cross-attention norm outputs throw max-pool layer before passing throw MLP layers
        smiles_att = self.max_pool(smiles_attn_output.squeeze(2))
        ppi_att = self.max_pool(ppi_attn_output.squeeze(2)) 
        combined_embeddings = torch.cat([smiles_att, ppi_att], dim=1)
        output = self.additional_layers(combined_embeddings)
        
        return self.sigmoid(output)

########################################################################################################################################################################

class AbstractModel(ABC, nn.Module):
    def __init__(self):
    	super(AbstractModel, self).__init__()
    
    @abstractmethod
    def forward(self, bmg, esm, custom, fegs, gae,
    	input_ids, attention_mask,
    	morgan_fingerprints, chemical_descriptors):
    	pass

    def train_model(self, num_epochs, train_loader, val_loader, optimizer, criterion, device):
        PRINTM(f'Start training !')
        for epoch in range(num_epochs):
            start_time = time.time()
            self.train()
            running_loss = 0.0
            for (batch_smiles, batch_esm_features, batch_custom_features, batch_fegs_features, batch_gae_features,
                 batch_input_ids, batch_attention_mas, batch_morgan, batch_chem_desc, batch_labels) in train_loader:
                # Move tensors to the configured device
                batch_attention_mas = batch_attention_mas.to(device)
                batch_input_ids = batch_input_ids.to(device)
                batch_esm_features = batch_esm_features.to(device)
                batch_custom_features = batch_custom_features.to(device)
                batch_fegs_features = batch_fegs_features.to(device)
                batch_gae_features = batch_gae_features.to(device)
                batch_morgan = batch_morgan.to(device)
                batch_chem_desc = batch_chem_desc.to(device)
                batch_labels = batch_labels.to(device)

                
                optimizer.zero_grad()
                outputs = self(batch_smiles, batch_esm_features,batch_custom_features,
                               batch_fegs_features, batch_gae_features, batch_input_ids, batch_attention_mas,batch_morgan, batch_chem_desc)

                loss = criterion(outputs.squeeze(), batch_labels)    
                loss.backward()
                optimizer.step()
    
            # Validate the model on the validation set
            val_loss, val_accuracy, val_auc = self.validate_model(val_loader, criterion, device)
            end_time = time.time()
            epoch_time = (end_time - start_time) / 60
            PRINTC()
            print(f"Epoch: {epoch+1}")
            print(f"Validation BCELoss: {val_loss:.5f}")
            print(f"Validation Accuracy (>0.8): {val_accuracy:.2f}")
            print(f"Validation AUC: {val_auc:.5f}")
            print(f"Epoch time: {epoch_time:.2f} minutes")
            PRINTC()
    
        print("Finish training !")

    def test_model(self, test_dataset, criterion, batch_size, shuffle, device):
        test_dataset = MoleculeDataset(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        self.eval()
        
        test_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_outputs = []
                
        with torch.no_grad():
            for (batch_smiles, batch_esm_features, batch_custom_features, batch_fegs_features, batch_gae_features,
                 batch_input_ids, batch_attention_mas, batch_morgan, batch_chem_desc, batch_labels) in test_loader:
                # Move tensors to the configured device
                batch_attention_mas = batch_attention_mas.to(device)
                batch_input_ids = batch_input_ids.to(device)
                batch_esm_features = batch_esm_features.to(device)
                batch_custom_features = batch_custom_features.to(device)
                batch_fegs_features = batch_fegs_features.to(device)
                batch_gae_features = batch_gae_features.to(device)
                batch_morgan = batch_morgan.to(device)
                batch_chem_desc = batch_chem_desc.to(device)
                batch_labels = batch_labels.to(device)
    
                outputs = self(batch_smiles, batch_esm_features, batch_custom_features,
                               batch_fegs_features, batch_gae_features, batch_input_ids, batch_attention_mas, batch_morgan, batch_chem_desc)
    
                loss = criterion(outputs.squeeze(), batch_labels)
                test_loss += loss.item()

                all_labels.extend(batch_labels.cpu().numpy())  
                all_outputs.extend(outputs.squeeze().cpu().numpy())  
    
                predicted = (outputs.squeeze() > 0.8).float()
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
    
        test_loss /= len(test_loader)
        accuracy = correct / total
        test_auc = roc_auc_score(all_labels, all_outputs) 
        PRINTC() 
        print(f"Test AUC: {test_auc:.5f}")
        PRINTC()
        return test_auc

    def validate_model(self, val_loader, criterion, device):
        self.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_outputs = []
        with torch.no_grad():
            for (batch_smiles, batch_esm_features, batch_custom_features, batch_fegs_features, batch_gae_features,
                 batch_input_ids, batch_attention_mas, batch_morgan, batch_chem_desc ,batch_labels) in val_loader:
                # Move tensors to the configured device
                batch_attention_mas = batch_attention_mas.to(device)
                batch_input_ids = batch_input_ids.to(device)
                batch_esm_features = batch_esm_features.to(device)
                batch_custom_features = batch_custom_features.to(device)
                batch_fegs_features = batch_fegs_features.to(device)
                batch_gae_features = batch_gae_features.to(device)
                batch_morgan = batch_morgan.to(device)
                batch_chem_desc = batch_chem_desc.to(device)
                batch_labels = batch_labels.to(device)
    
                outputs = self(batch_smiles, batch_esm_features,batch_custom_features,
                               batch_fegs_features, batch_gae_features, batch_input_ids, batch_attention_mas, batch_morgan, batch_chem_desc)
                loss = criterion(outputs.squeeze(), batch_labels)
                val_loss += loss.item()
    
                all_labels.extend(batch_labels.cpu().numpy())  
                all_outputs.extend(outputs.squeeze().cpu().numpy())  
    
                predicted = (outputs.squeeze() > 0.8).float()
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
    
        val_loss /= len(val_loader)
        accuracy = correct / total
        val_auc = roc_auc_score(all_labels, all_outputs)  
        return val_loss, accuracy, val_auc

    def cross_validate(self, dataset, num_folds=5,num_epochs=10, batch_size=32, learning_rate=0.0001, weight_decay=1e-5, shuffle=True, device='cuda'):
        kf = KFold(n_splits=num_folds, shuffle=shuffle)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            
            print(f"Fold {fold+1}/{num_folds}")
            
            # Split dataset
            train_subset = dataset.iloc[train_idx].reset_index(drop=True)
            val_subset = dataset.iloc[val_idx].reset_index(drop=True)
            
            train_dataset = MoleculeDataset(train_subset)
            val_dataset = MoleculeDataset(val_subset)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
            
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
            
            self.train_model(num_epochs, train_loader, val_loader, optimizer, criterion, device)
            
            # Validate the model
            val_loss, val_accuracy, val_auc = self.validate_model(val_loader, criterion, device)
            fold_results.append((val_loss, val_accuracy, val_auc))

            PRINTC()
            print(f"Fold {fold+1} - Validation BCELoss: {val_loss:.5f}, Accuracy: {val_accuracy:.2f}, AUC: {val_auc:.5f}")
            PRINTC()
            
        avg_val_loss = sum([result[0] for result in fold_results]) / num_folds
        avg_val_accuracy = sum([result[1] for result in fold_results]) / num_folds
        avg_val_auc = sum([result[2] for result in fold_results]) / num_folds
        
        print(f"\nAverage Validation BCELoss: {avg_val_loss:.5f}")
        print(f"Average Validation Accuracy: {avg_val_accuracy:.2f}")
        print(f"Average Validation AUC: {avg_val_auc:.5f}")
        
        return fold_results




