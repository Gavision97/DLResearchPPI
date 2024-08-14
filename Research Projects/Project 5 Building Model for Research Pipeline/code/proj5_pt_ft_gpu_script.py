#!/usr/bin/env python
# coding: utf-8

# # Pre Train Chemprop Model #

# ## Import Libraries ##

# In[1]:


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


# In[2]:


def PRINT() -> None: print(f"{'-'*80}\nDone\n{'-'*80}")
def PRINTC() -> None: print(f"{'-'*80}")
def PRINTM(M) -> None: print(f"{'-'*80}\n{M}\n{'-'*80}")


# ## Verify GPU Availability ##

# In[3]:


get_ipython().system('nvidia-smi')


# For this task, we'll use the BGU cluster GPU `NVIDIA RTX 6000 Ada Generation` to achieve better performance during the training of our pre-trained and fine-tuned models, allowing for more efficient processing of large datasets and complex computations.

# In[4]:


if torch.cuda.is_available():
    PRINTM(f"GPU is available.")
    device = "cuda"
else:
    PRINTM(f"GPU is not available. Using CPU instead.")
    device = "cpu"


# In[5]:


PRINTM(f"PyTorch version: {torch.__version__}")
PRINTM(f"CUDA available: {torch.cuda.is_available()}")
PRINTM(f"CUDA version:  {torch.version.cuda}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA'}")


# # Fine Tune Chemprop Pre-trained Model & Generate AUVG_PPI Model #

# In[21]:


def merge_datasets(dataset, features_df):
    # Merge features for uniprot_id1
    dataset = dataset.merge(features_df, how='left', left_on='uniprot_id1', right_on='UniProt_ID', suffixes=('', '_id1'))
    dataset = dataset.drop(columns=['UniProt_ID'])
    
    # Merge features for uniprot_id2
    features_df_renamed = features_df.add_suffix('_id2')
    features_df_renamed = features_df_renamed.rename(columns={'UniProt_ID_id2': 'UniProt_ID'})
    dataset = dataset.merge(features_df_renamed, how='left', left_on='uniprot_id2', right_on='UniProt_ID', suffixes=('', '_id2'))
    dataset = dataset.drop(columns=['UniProt_ID', 'uniprot_id1', 'uniprot_id2'])
    
    return dataset.drop_duplicates()


# In[22]:


def convert_uniprot_ids(dataset, mapping_df):
    # Create a dictionary from the mapping dataframe
    mapping_dict = mapping_df.set_index('From')['Entry'].to_dict()

    # Map the uniprot_id1 and uniprot_id2 columns to their respective Entry values
    dataset['uniprot_id1'] = dataset['uniprot_id1'].map(mapping_dict)
    dataset['uniprot_id2'] = dataset['uniprot_id2'].map(mapping_dict)
    return dataset.drop_duplicates()
    
def merge_datasets(dataset, features_df):
    # Merge features for uniprot_id1
    dataset = dataset.merge(features_df, how='left', left_on='uniprot_id1', right_on='UniProt_ID', suffixes=('', '_id1'))
    dataset = dataset.drop(columns=['UniProt_ID'])
    
    # Merge features for uniprot_id2
    features_df_renamed = features_df.add_suffix('_id2')
    features_df_renamed = features_df_renamed.rename(columns={'UniProt_ID_id2': 'UniProt_ID'})
    dataset = dataset.merge(features_df_renamed, how='left', left_on='uniprot_id2', right_on='UniProt_ID', suffixes=('', '_id2'))
    dataset = dataset.drop(columns=['UniProt_ID', 'uniprot_id1', 'uniprot_id2'])
    
    return dataset.drop_duplicates()


# ## Finetune Step ##

# In[23]:


class PretrainedChempropModel(nn.Module):
    def __init__(self, checkpoints_path, batch_size):
        super(PretrainedChempropModel, self).__init__()
        self.mpnn = self.load_pretrained_model(checkpoints_path)
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        self.batch_size = batch_size
        
    def forward(self, smiles):
        # Prepare the data in order to generate embeddings from modulators SMILES
        self.smiles_data = [data.MoleculeDatapoint.from_smi(smi) for smi in smiles]
        self.smiles_dset = data.MoleculeDataset(self.smiles_data, featurizer=self.featurizer)
        self.smiles_loader = data.build_dataloader(self.smiles_dset, batch_size=batch_size, shuffle=False)
        
        embeddings = [
            # Etract the embedding from the last FFN layer, i.e., before the final prediction(thus i=-1)
            self.mpnn.predictor.encode(self.fingerprints_from_batch_molecular_graph(batch, self.mpnn), i=-1) 
            for batch in self.smiles_loader
        ]
        #print(embeddings)
        if not embeddings:
             return torch.empty(0, device=device)
        embeddings = torch.cat(embeddings, 0).to(device)
        return embeddings

    def fingerprints_from_batch_molecular_graph(self, batch, mpnn):
        batch.bmg.to(device)
        H_v = mpnn.message_passing(batch.bmg, batch.V_d)
        H = mpnn.agg(H_v, batch.bmg.batch)
        H = mpnn.bn(H)
        fingerprints = H if batch.X_d is None else torch.cat((H, mpnn.batch.X_d_transform(X_d)), 1)
        return fingerprints

    def load_pretrained_model(self, checkpoints_path):
        mpnn = models.MPNN.load_from_checkpoint(checkpoints_path).to(device)
        return mpnn


# In[24]:


class ChemBERTaPT(nn.Module):
    def __init__(self):
        super(ChemBERTaPT, self).__init__()
        self.model_name = "DeepChem/ChemBERTa-77M-MTR"
        self.chemberta = RobertaModel.from_pretrained(self.model_name)

    def forward(self, input_ids, attention_mask):
        bert_output = self.chemberta(input_ids=input_ids, attention_mask=attention_mask)
        return bert_output[1]


# In[53]:


class AUVG_PPI(nn.Module):
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
        
        return fold_results


# In[54]:


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


# ## Train the Model on Finetuned Datasets (multi_ppim_fold_2_0.8)  ##

# ### Load & Prepare the Dataset ###

# In[44]:


ds_folder_path = os.path.join('datasets', 'finetune_dataset', 'multi_ppim_folds_2_0.8')
all_files = os.listdir(ds_folder_path)

PRINTM(f'Folder content:\n\n{all_files}')


# In[45]:


dataframes = {}

# Read each CSV file into a dataframe and store it in the dictionary
for file in all_files:
    file_path = os.path.join(ds_folder_path, file)
    df = pd.read_csv(file_path)
    df_name = file.replace('.csv', '_df')
    dataframes[df_name] = df


# #### Add PPI Features to Train Dataframe ####
# 
# Before starting the training process on the training dataset, we'll add PPI features to the train and test datasets to utilize the model we built earlier. For this, we'll use the `esm_features.csv` dataset, which contains UniProt IDs with their features generated by Facebook's LLM algorithm called *ESM*. We'll also use the `idmapping_unip.tsv` file to map the correct features to the corresponding UniProt IDs in our training & testing datasets, using two helper functions. This process is identical to the one we used when we initially trained the model.

# In[47]:


uniprot_mapping = pd.read_csv(os.path.join('datasets', 'idmapping_unip.tsv'), delimiter = "\t")
ppi_features_df = pd.read_csv(os.path.join('datasets', 'merged_ppi_features.csv'))
PRINT()


# In[48]:


for df_name in dataframes.keys():
    dataframes[df_name] = convert_uniprot_ids(dataframes[df_name], uniprot_mapping)
    dataframes[df_name] = merge_datasets(dataframes[df_name], ppi_features_df)

# Access each dataframe using its name
train_fold1_df = dataframes['train_fold1_df']
train_fold2_df = dataframes['train_fold2_df']
train_fold3_df = dataframes['train_fold3_df']
train_fold4_df = dataframes['train_fold4_df']
train_fold5_df = dataframes['train_fold5_df']
test_fold1_df = dataframes['test_fold1_df']
test_fold2_df = dataframes['test_fold2_df']
test_fold3_df = dataframes['test_fold3_df']
test_fold4_df = dataframes['test_fold4_df']
test_fold5_df = dataframes['test_fold5_df']

PRINTM(f'Done inverse mapping & merging successfully !')


# In[50]:


test_fold2_df.head()


# #### Data Cleaning ####

# In[ ]:


for df_name, df in dataframes.items():
    null_counts = df.isnull().sum().sum()
    PRINTM(f'Number of nan values in {df_name} is -> {null_counts}')


# ### Train a Model on Each Fold ###

# In[55]:


def generate_train_val_datasets(df, portion) -> (pd.DataFrame, pd.DataFrame):
    train_size = int(portion * len(df))
    val_size = len(df) - train_size
    train_data, val_data = train_test_split(df, test_size=val_size)
    return (train_data, val_data)

def generate_dataloaders(train_data, val_data,
                         test_data, batch_size, shuffle):
    train_dataset = MoleculeDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    val_dataset = MoleculeDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    
    test_dataset = MoleculeDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return (train_loader, val_loader, test_loader)

def generate_model(checkpoint_path,
                   batch_size,
                  dropout) -> nn.Module:
    pretrained_chemprop_model = PretrainedChempropModel(checkpoints_path, batch_size)
    chemberta_model = ChemBERTaPT()
    ft_model = AUVG_PPI(pretrained_chemprop_model, chemberta_model, dropout).to(device)

    PRINTM('Generated combined model for fine-tuning successfully !')
    return ft_model


# In[56]:


checkpoints_path = os.path.join('pt_chemprop_checkpoint', 'fold_0', 'model_0', 'checkpoints', 'best-epoch=49-val_loss=0.12.ckpt')
batch_size = 32


# #### Fold number 1 #####

# In[ ]:


ft_model_f1 = generate_model(checkpoints_path, batch_size=32, dropout=0.1)
f1_res = ft_model_f1.cross_validate(train_fold1_df, num_folds=10, num_epochs=3,
                     batch_size=32, learning_rate=0.0001, weight_decay=1e-5,
                     shuffle=True, device=device)


# In[58]:


ft_model_f1.test_model(test_fold1_df,
                         criterion= nn.BCELoss() ,batch_size=32,
                         shuffle=True, device=device)


# In[ ]:


ft_model_f2 = generate_model(checkpoints_path, batch_size=32, dropout=0.8)
f2_res = ft_model_f2.cross_validate(train_fold2_df, num_folds=10, num_epochs=3,
                     batch_size=32, learning_rate=0.0001, weight_decay=1e-3,
                     shuffle=True, device=device)


# In[39]:


ft_model_f2.test_model(test_fold2_df,
                         criterion= nn.BCELoss() ,batch_size=32,
                         shuffle=True, device=device)


# #### Fold number 3 ####

# In[ ]:


ft_model_f3 = generate_model(checkpoints_path, batch_size=32, dropout=0.8)
f3_res = ft_model_f3.cross_validate(train_fold3_df, num_folds=10, num_epochs=3,
                     batch_size=32, learning_rate=0.0001, weight_decay=1e-3,
                     shuffle=True, device=device)


# In[ ]:


ft_model_f3.test_model(test_fold3_df,
                         criterion= nn.BCELoss() ,batch_size=32,
                         shuffle=True, device=device)


# #### Folder number 4 ####

# In[ ]:


ft_model_f4 = generate_model(checkpoints_path, batch_size=32, dropout=0.1)
f4_res = ft_model_f4.cross_validate(train_fold4_df, num_folds=10, num_epochs=3,
                     batch_size=32, learning_rate=0.0001, weight_decay=1e-5,
                     shuffle=True, device=device)


# In[ ]:


ft_model_f4.test_model(test_fold4_df,
                         criterion= nn.BCELoss() ,batch_size=32,
                         shuffle=True, device=device)


# #### Folder number 5 ####

# In[ ]:


ft_model_f5 = generate_model(checkpoints_path, batch_size=32, dropout=0.8)
f5_res = ft_model_f5.cross_validate(train_fold5_df, num_folds=10, num_epochs=3,
                     batch_size=32, learning_rate=0.0001, weight_decay=1e-3,
                     shuffle=True, device=device)


# In[ ]:


ft_model_f5.test_model(test_fold5_df,
                         criterion= nn.BCELoss() ,batch_size=32,
                         shuffle=True, device=device)


# ## Train the Model on Finetuned Datasets (DLIP_folds_2_0.8)  ##

# In[62]:


ds_folder_path = os.path.join('datasets', 'finetune_dataset', 'DLIP_folds_2_0.8')
all_files = os.listdir(ds_folder_path)

PRINTM(f'Folder content:\n\n{all_files}')


# In[63]:


dataframes = {}

# Read each CSV file into a dataframe and store it in the dictionary
for file in all_files:
    file_path = os.path.join(ds_folder_path, file)
    df = pd.read_csv(file_path)
    df_name = file.replace('.csv', '_df')
    dataframes[df_name] = df


# In[65]:


for df_name in dataframes.keys():
    dataframes[df_name] = convert_uniprot_ids(dataframes[df_name], uniprot_mapping)
    dataframes[df_name] = merge_datasets(dataframes[df_name], ppi_features_df)

# Access each dataframe using its name
train_fold1_df = dataframes['train_fold1_df']
train_fold2_df = dataframes['train_fold2_df']
train_fold3_df = dataframes['train_fold3_df']
train_fold4_df = dataframes['train_fold4_df']
train_fold5_df = dataframes['train_fold5_df']
test_fold1_df = dataframes['test_fold1_df']
test_fold2_df = dataframes['test_fold2_df']
test_fold3_df = dataframes['test_fold3_df']
test_fold4_df = dataframes['test_fold4_df']
test_fold5_df = dataframes['test_fold5_df']

PRINTM(f'Done inverse mapping & merging successfully !')


# #### Fold number 1 ####

# In[30]:


batch_size = 32


# In[ ]:


ft_model_f1_dlip = generate_model(checkpoints_path, batch_size=32, dropout=0.1)
f1_dlip_res = ft_model_f1_dlip.cross_validate(train_fold1_df, num_folds=10, num_epochs=3,
                     batch_size=32, learning_rate=0.0001, weight_decay=1e-5,
                     shuffle=True, device=device)


# In[68]:


ft_model_f1_dlip.test_model(test_fold1_df,
                         criterion= nn.BCELoss() ,batch_size=32,
                         shuffle=True, device=device)


# #### Fold number 2 ####

# In[29]:


train_fold2_df.head()


# In[ ]:


ft_model_f2_dlip = generate_model(checkpoints_path, batch_size=32, dropout=0.1)
f2_dlip_res = ft_model_f2_dlip.cross_validate(train_fold2_df, num_folds=10, num_epochs=5,
                     batch_size=32, learning_rate=0.0001, weight_decay=1e-5,
                     shuffle=True, device=device)


# In[ ]:


ft_model_f2_dlip.test_model(test_fold2_df,
                         criterion= nn.BCELoss() ,batch_size=32,
                         shuffle=True, device=device)


# #### Fold number 3 ####

# In[ ]:


ft_model_f3_dlip = generate_model(checkpoints_path, batch_size=32, dropout=0.1)
f3_dlip_res = ft_model_f3_dlip.cross_validate(train_fold3_df, num_folds=10, num_epochs=5,
                     batch_size=32, learning_rate=0.0001, weight_decay=1e-5,
                     shuffle=True, device=device)


# In[36]:


ft_model_f3_dlip.test_model(test_fold3_df,
                         criterion= nn.BCELoss() ,batch_size=32,
                         shuffle=True, device=device)


# #### Fold number 4 ####

# In[ ]:


ft_model_f4_dlip = generate_model(checkpoints_path, batch_size=32, dropout=0.1)
f4_dlip_res = ft_model_f4_dlip.cross_validate(train_fold4_df, num_folds=10, num_epochs=5,
                     batch_size=32, learning_rate=0.0001, weight_decay=1e-5,
                     shuffle=True, device=device)


# In[38]:


ft_model_f4_dlip.test_model(test_fold4_df,
                         criterion= nn.BCELoss() ,batch_size=32,
                         shuffle=True, device=device)


# #### Fold number 5 ####

# In[ ]:


ft_model_f5_dlip = generate_model(checkpoints_path, batch_size=32, dropout=0.1)
f5_dlip_res = ft_model_f5_dlip.cross_validate(train_fold5_df, num_folds=10, num_epochs=5,
                     batch_size=32, learning_rate=0.0001, weight_decay=1e-5,
                     shuffle=True, device=device)


# In[37]:


ft_model_f5_dlip.test_model(test_fold5_df,
                         criterion= nn.BCELoss() ,batch_size=32,
                         shuffle=True, device=device)


# ## Train the Model on Finetuned Datasets (DLIP_folds_2_0.9)  ##

# In[21]:


ds_folder_path = os.path.join('datasets', 'finetune_dataset', 'DLIP_folds_3_0.9')
all_files = os.listdir(ds_folder_path)

PRINTM(f'Folder content:\n\n{all_files}')


# In[22]:


dataframes = {}

# Read each CSV file into a dataframe and store it in the dictionary
for file in all_files:
    file_path = os.path.join(ds_folder_path, file)
    df = pd.read_csv(file_path)
    df_name = file.replace('.csv', '_df')
    dataframes[df_name] = df


# In[23]:


for df_name in dataframes.keys():
    dataframes[df_name] = convert_uniprot_ids(dataframes[df_name], uniprot_mapping)
    dataframes[df_name] = merge_datasets(dataframes[df_name], ppi_features_df)

# Access each dataframe using its name
train_fold1_df = dataframes['train_fold1_df']
train_fold2_df = dataframes['train_fold2_df']
train_fold3_df = dataframes['train_fold3_df']
train_fold4_df = dataframes['train_fold4_df']
train_fold5_df = dataframes['train_fold5_df']
test_fold1_df = dataframes['test_fold1_df']
test_fold2_df = dataframes['test_fold2_df']
test_fold3_df = dataframes['test_fold3_df']
test_fold4_df = dataframes['test_fold4_df']
test_fold5_df = dataframes['test_fold5_df']

PRINTM(f'Done inverse mapping & merging successfully !')


# In[24]:


test_fold5_df.head()


# In[ ]:


for df_name, df in dataframes.items():
    null_counts = df.isnull().sum().sum()
    PRINTM(f'Number of nan values in {df_name} is -> {null_counts}')


# #### Fold number 1 #####

# In[34]:


batch_size=64


# In[ ]:


ft_model_f1_dlip_ = generate_model(checkpoints_path, batch_size=64, dropout=0.8)
f1_dlip_res = ft_model_f1_dlip_.cross_validate(train_fold1_df, num_folds=10, num_epochs=5,
                     batch_size=64, learning_rate=0.0001, weight_decay=1e-4,
                     shuffle=True, device=device)


# In[33]:


ft_model_f1_dlip_.test_model(test_fold1_df,
                         criterion= nn.BCELoss() ,batch_size=64,
                         shuffle=True, device=device)


# #### Fold number 2 #####

# In[ ]:


ft_model_f2_dlip_ = generate_model(checkpoints_path, batch_size=64, dropout=0.3)
f2_dlip_res_ = ft_model_f2_dlip_.cross_validate(train_fold2_df, num_folds=10, num_epochs=5,
                     batch_size=64, learning_rate=0.0001, weight_decay=1e-4,
                     shuffle=True, device=device)


# In[38]:


ft_model_f2_dlip_.test_model(test_fold2_df,
                         criterion= nn.BCELoss() ,batch_size=64,
                         shuffle=True, device=device)


# #### Fold number 3 #####

# In[ ]:


ft_model_f3_dlip_ = generate_model(checkpoints_path, batch_size=64, dropout=0.3)
f3_dlip_res_ = ft_model_f3_dlip_.cross_validate(train_fold3_df, num_folds=10, num_epochs=5,
                     batch_size=64, learning_rate=0.0001, weight_decay=1e-4,
                     shuffle=True, device=device)


# In[27]:


ft_model_f3_dlip_.test_model(test_fold3_df,
                         criterion= nn.BCELoss() ,batch_size=64,
                         shuffle=True, device=device)


# #### Fold number 4 #####

# In[ ]:


ft_model_f4_dlip_ = generate_model(checkpoints_path, batch_size=64, dropout=0.3)
f4_dlip_res_ = ft_model_f4_dlip_.cross_validate(train_fold4_df, num_folds=10, num_epochs=5,
                     batch_size=64, learning_rate=0.0001, weight_decay=1e-4,
                     shuffle=True, device=device)


# In[29]:


ft_model_f4_dlip_.test_model(test_fold4_df,
                         criterion= nn.BCELoss() ,batch_size=64,
                         shuffle=True, device=device)


# #### Fold number 5 #####

# In[ ]:


ft_model_f5_dlip_ = generate_model(checkpoints_path, batch_size=64, dropout=0.3)
f5_dlip_res_ = ft_model_f5_dlip_.cross_validate(train_fold5_df, num_folds=10, num_epochs=5,
                     batch_size=64, learning_rate=0.0001, weight_decay=1e-4,
                     shuffle=True, device=device)


# In[31]:


ft_model_f5_dlip_.test_model(test_fold5_df,
                         criterion= nn.BCELoss() ,batch_size=64,
                         shuffle=True, device=device)


# In[44]:


ds_folder_path = os.path.join('datasets', 'finetune_dataset', 'original_folds PPIMI')
all_files = os.listdir(ds_folder_path)

PRINTM(f'Folder content:\n\n{all_files}')


# In[45]:


dataframes = {}

# Read each CSV file into a dataframe and store it in the dictionary
for file in all_files:
    file_path = os.path.join(ds_folder_path, file)
    df = pd.read_csv(file_path)
    df_name = file.replace('.csv', '_df')
    dataframes[df_name] = df


# In[46]:


for df_name, df in dataframes.items():
    # Replace 'na' with np.nan if necessary
    df.replace('na', np.nan, inplace=True)
    
    # Identify rows where 'uniprot_id2' is NaN and replace them with 'uniprot_id1' values
    df.loc[df['uniprot_id2'].isna(), 'uniprot_id2'] = df['uniprot_id1']
    
    print(f'Updated DataFrame: {df_name}')


# In[49]:


test_fold5_df = dataframes['test_fold5_df']
test_fold5_df.tail()


# In[50]:


for df_name in dataframes.keys():
    dataframes[df_name] = convert_uniprot_ids(dataframes[df_name], uniprot_mapping)
    dataframes[df_name] = merge_datasets(dataframes[df_name], esm_df)

# Access each dataframe using its name
train_fold1_df = dataframes['train_val_fold1_df']
train_fold2_df = dataframes['train_val_fold2_df']
train_fold3_df = dataframes['train_val_fold3_df']
train_fold4_df = dataframes['train_val_fold4_df']
train_fold5_df = dataframes['train_val_fold5_df']
test_fold1_df = dataframes['test_fold1_df']
test_fold2_df = dataframes['test_fold2_df']
test_fold3_df = dataframes['test_fold3_df']
test_fold4_df = dataframes['test_fold4_df']
test_fold5_df = dataframes['test_fold5_df']

PRINTM(f'Done inverse mapping & merging successfully !')


# In[51]:


test_fold5_df.head()


# In[52]:


for df_name, df in dataframes.items():
    null_counts = df.isnull().sum().sum()
    PRINTM(f'Number of nan values in {df_name} is -> {null_counts}')


# In[53]:


batch_size = 32


# In[ ]:


ft_model_f1 = generate_model(checkpoints_path, batch_size=32, dropout=0.5)
f1_res = ft_model_f1.cross_validate(train_fold1_df, num_folds=5, num_epochs=10,
                     batch_size=32, learning_rate=0.0001, weight_decay=1e-5,
                     shuffle=True, device=device)


# In[59]:


ft_model_f1.test_model(test_fold1_df,
                         criterion= nn.BCELoss() ,batch_size=32,
                         shuffle=True, device=device)


# In[ ]:


ft_model_f2 = generate_model(checkpoints_path, batch_size=32, dropout=0.5)
f2_res = ft_model_f2.cross_validate(train_fold2_df, num_folds=5, num_epochs=10,
                     batch_size=32, learning_rate=0.0001, weight_decay=1e-5,
                     shuffle=True, device=device)


# In[61]:


ft_model_f2.test_model(test_fold2_df,
                         criterion= nn.BCELoss() ,batch_size=32,
                         shuffle=True, device=device)


# In[ ]:


ft_model_f3 = generate_model(checkpoints_path, batch_size=32, dropout=0.5)
f3_res = ft_model_f3.cross_validate(train_fold3_df, num_folds=5, num_epochs=10,
                     batch_size=32, learning_rate=0.0001, weight_decay=1e-5,
                     shuffle=True, device=device)


# In[ ]:


ft_model_f3.test_model(test_fold3_df,
                         criterion= nn.BCELoss() ,batch_size=32,
                         shuffle=True, device=device)


# In[ ]:


ft_model_f4 = generate_model(checkpoints_path, batch_size=32, dropout=0.5)
f4_res = ft_model_f4.cross_validate(train_fold4_df, num_folds=5, num_epochs=10,
                     batch_size=32, learning_rate=0.0001, weight_decay=1e-5,
                     shuffle=True, device=device)


# In[ ]:


ft_model_f4.test_model(test_fold4_df,
                         criterion= nn.BCELoss() ,batch_size=32,
                         shuffle=True, device=device)


# In[ ]:


ft_model_f5 = generate_model(checkpoints_path, batch_size=32, dropout=0.5)
f5_res = ft_model_f5.cross_validate(train_fold5_df, num_folds=5, num_epochs=10,
                     batch_size=32, learning_rate=0.0001, weight_decay=1e-5,
                     shuffle=True, device=device)


# In[ ]:


ft_model_f5.test_model(test_fold5_df,
                         criterion= nn.BCELoss() ,batch_size=32,
                         shuffle=True, device=device)


# In[ ]:


model_f3 = generate_model(batch_size=32, dropout=0.5)
f3_res = model_f3.cross_validate(train_fold3_df, num_folds=5, num_epochs=10,
                     batch_size=32, learning_rate=0.0001, weight_decay=1e-5,
                     shuffle=True, device=device)


# In[51]:


model_f3.test_model(test_fold3_df,
                         criterion= nn.BCELoss() ,batch_size=32,
                         shuffle=True, device=device)


# In[ ]:


model_f4 = generate_model(batch_size=32, dropout=0.5)
f4_res = model_f4.cross_validate(train_fold4_df, num_folds=5, num_epochs=10,
                     batch_size=32, learning_rate=0.0001, weight_decay=1e-5,
                     shuffle=True, device=device)


# In[53]:


model_f4.test_model(test_fold4_df,
                         criterion= nn.BCELoss() ,batch_size=32,
                         shuffle=True, device=device)


# In[ ]:


model_f5 = generate_model(batch_size=32, dropout=0.5)
f5_res = model_f5.cross_validate(train_fold5_df, num_folds=5, num_epochs=10,
                     batch_size=32, learning_rate=0.0001, weight_decay=1e-5,
                     shuffle=True, device=device)


# In[55]:


model_f5.test_model(test_fold5_df,
                         criterion= nn.BCELoss() ,batch_size=32,
                         shuffle=True, device=device)


# ## DLiP 0.8 ##

# In[ ]:


model_f1 = generate_model(checkpoints_path, batch_size=32, dropout=0.5)
f1_dlip_res = ft_model_f1_dlip.cross_validate(train_fold1_df, num_folds=5, num_epochs=10,
                     batch_size=32, learning_rate=0.0001, weight_decay=1e-5,
                     shuffle=True, device=device)


# In[33]:


model_f1


# In[ ]:




