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

# import necessary libraries for Chemberta model
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, AdamW, get_linear_schedule_with_warmup , BertModel

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

from abc import ABC, abstractmethod


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
            for i, (batch_smiles, batch_protein_features, batch_input_ids, batch_attention_mas, batch_labels) in enumerate(train_loader):
                
                # Move tensors to the configured device
                batch_attention_mas = batch_attention_mas.to(device)
                batch_input_ids = batch_input_ids.to(device)
                batch_protein_features = batch_protein_features.to(device)
                batch_labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                outputs = self(batch_smiles, batch_protein_features, batch_input_ids, batch_attention_mas)
                loss = criterion(outputs.squeeze(), batch_labels)
    
                loss.backward()
                optimizer.step()
    
                #running_loss += loss.item()
                #if i % 100 == 99 and i > 0:
                    #print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 100:.4f}")
                    #running_loss = 0.0
    
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

    def test_model(self, test_dataset,
                   criterion,batch_size,
                   shuffle, device):
        test_dataset = MoleculeDataset(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        self.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_outputs = []
        with torch.no_grad():
            for batch_smiles, batch_protein_features, batch_input_ids, batch_attention_mas, batch_labels in test_loader:
                # Move tensors to the configured device

                batch_attention_mas = batch_attention_mas.to(device)
                batch_input_ids = batch_input_ids.to(device)
                batch_protein_features = batch_protein_features.to(device)
                batch_labels = batch_labels.to(device)
    
                outputs = self(batch_smiles, batch_protein_features, batch_input_ids, batch_attention_mas)

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
        #print(f"Test BCELoss: {test_loss:.5f}")
        #print(f"Test Accuracy: {accuracy:.2f}")
        print(f"Test AUC: {test_auc:.5f}")
        PRINTC()

    def validate_model(self, val_loader, criterion, device):
        self.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_outputs = []
        with torch.no_grad():
            for batch_smiles, batch_protein_features, batch_input_ids, batch_attention_mas, batch_labels in val_loader:
                # Move tensors to the configured device
                batch_input_ids = batch_input_ids.to(device)
                batch_attention_mas = batch_attention_mas.to(device)
                batch_protein_features = batch_protein_features.to(device)
                batch_labels = batch_labels.to(device)
    
                outputs = self(batch_smiles, batch_protein_features, batch_input_ids, batch_attention_mas)
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

    def cross_validate(self, dataset,
                       num_folds=5,num_epochs=10,
                       batch_size=32,
                       learning_rate=0.0001, weight_decay=1e-5,
                       shuffle=True, device='cuda'):
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
        