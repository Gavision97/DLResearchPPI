import os
import time
import warnings
import sys
import logging

logging.basicConfig(filename='output.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
#logging.info("Initialized log file successfully ...")
sys.stderr.write("This is an error message.\n")

warnings.filterwarnings('ignore')

import chemprop
from chemprop import data, featurizers, models

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import RobertaTokenizer, RobertaModel

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score

from rdkit import Chem

# Function to log messages
def log_message(message, level=logging.INFO):
    logging.log(level, message)
    # Flush to ensure the message is written to the file immediately
    for handler in logging.root.handlers:
        handler.flush()

log_message("Initialized log file successfully ....")
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

sys.stderr.write("This is an error message #2.\n")
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

# Pretrained Chemprop Model
class PretrainedChempropModel(nn.Module):
    def __init__(self, checkpoints_path, batch_size):
        super(PretrainedChempropModel, self).__init__()
        self.mpnn = self.load_pretrained_model(checkpoints_path)
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        self.batch_size = batch_size
        self.device = check_device()

        
    def forward(self, smiles):
        smiles_data = [data.MoleculeDatapoint.from_smi(smi) for smi in smiles]
        smiles_dset = data.MoleculeDataset(smiles_data, featurizer=self.featurizer)
        smiles_loader = data.build_dataloader(smiles_dset, batch_size=self.batch_size, shuffle=False)
        
        embeddings = [
            self.mpnn.predictor.encode(self.fingerprints_from_batch_molecular_graph(batch, self.mpnn), i=-1) 
            for batch in smiles_loader
        ]
        if not embeddings:
            return torch.empty(0, device=self.device)
        embeddings = torch.cat(embeddings, 0).to(self.device)
        return embeddings

    def fingerprints_from_batch_molecular_graph(self, batch, mpnn):
        batch.bmg.to(self.device)
        H_v = mpnn.message_passing(batch.bmg, batch.V_d)
        H = mpnn.agg(H_v, batch.bmg.batch)
        H = mpnn.bn(H)
        fingerprints = H if batch.X_d is None else torch.cat((H, mpnn.batch.X_d_transform(X_d)), 1)
        return fingerprints

    def load_pretrained_model(self, checkpoints_path):
        device = check_device()
        mpnn = models.MPNN.load_from_checkpoint(checkpoints_path).to(device)
        return mpnn

# ChemBERTa Model
class ChemBERTaPT(nn.Module):
    def __init__(self):
        super(ChemBERTaPT, self).__init__()
        self.model_name = "DeepChem/ChemBERTa-77M-MTR"
        self.chemberta = RobertaModel.from_pretrained(self.model_name)

    def forward(self, input_ids, attention_mask):
        bert_output = self.chemberta(input_ids=input_ids, attention_mask=attention_mask)
        return bert_output[1]
sys.stderr.write("This is an error message #3.\n")
# AUVG_PPI Model
class AUVG_PPI(nn.Module):
    def __init__(self, pretrained_chemprop_model, chemberta_model, dropout):
        super(AUVG_PPI, self).__init__()
        self.pretrained_chemprop_model = pretrained_chemprop_model
        self.chemberta_model = chemberta_model
        self.dropout = dropout
        self.device = check_device()
        
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
        ppi_features = proteins.to(self.device)
        ppi_features = self.ppi_mlp(ppi_features)

        # Concatinate chemprop fingerprints embeddings, chemberta embeddings and PPI embeddings together into one tensor
        # Afterwards, pass them through MLP layer and make prediction
        combined_embeddings = torch.cat([fingerprints, chemberta_embeddings, ppi_features], dim=1).to(self.device)
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

sys.stderr.write("This is an error message #4.\n")   
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

def generate_model(checkpoints_path,
                   batch_size,
                  dropout) -> nn.Module:
    pretrained_chemprop_model = PretrainedChempropModel(checkpoints_path, batch_size)
    chemberta_model = ChemBERTaPT()
    device = check_device()
    ft_model = AUVG_PPI(pretrained_chemprop_model, chemberta_model, dropout).to(device)

    PRINTM('Generated combined model for fine-tuning successfully !')
    return ft_model




def run_script():
    device = check_device()

    # Write logs
    logging.info("Device details:")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    logging.info(f"CUDA version: {torch.version.cuda}")
    logging.info(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA'}")
    logging.info()

    checkpoints_path = os.path.join('pt_chemprop_checkpoint', 'fold_0', 'model_0', 'checkpoints', 'best-epoch=49-val_loss=0.12.ckpt')
    batch_size = 32
    
    data_sets = {
        'multi_ppim_folds_2_0.8': os.path.join('datasets', 'finetune_dataset', 'multi_ppim_folds_2_0.8'),
        'DLIP_folds_2_0.8': os.path.join('datasets', 'finetune_dataset', 'DLIP_folds_2_0.8'),
        'DLIP_folds_3_0.9': os.path.join('datasets', 'finetune_dataset', 'DLIP_folds_3_0.9')
    }
    
    uniprot_mapping = pd.read_csv(os.path.join('datasets', 'idmapping_unip.tsv'), delimiter="\t")
    ppi_features_df = pd.read_csv(os.path.join('datasets', 'merged_ppi_features.csv'))
    
    for ds_name, ds_folder_path in data_sets.items():
        dataframes = {}
        all_files = os.listdir(ds_folder_path)
        
        for file in all_files:
            file_path = os.path.join(ds_folder_path, file)
            df = pd.read_csv(file_path)
            df_name = file.replace('.csv', '_df')
            dataframes[df_name] = df

        for df_name in dataframes.keys():
            dataframes[df_name] = convert_uniprot_ids(dataframes[df_name], uniprot_mapping)
            dataframes[df_name] = merge_datasets(dataframes[df_name], ppi_features_df)

        # Define fold indices
        folds = [1, 2, 3, 4, 5]

        for fold in folds:
            train_df = dataframes[f'train_fold{fold}_df']
            test_df = dataframes[f'test_fold{fold}_df']
            
            logging.info(f"Processing {ds_name} Fold {fold}...")
            logging.info("Start training...")
            ft_model_f1 = generate_model(checkpoints_path, batch_size=batch_size, dropout=0.2)
            
            # Cross-validation and training
            res = ft_model_f1.cross_validate(train_df, num_folds=5, num_epochs=4,
                          batch_size=batch_size, learning_rate=0.0001, weight_decay=1e-5,
                          shuffle=True, device=device)

            # Testing the model
            test_res = ft_model_f1.test_model(test_df, criterion=nn.BCELoss(),
                          batch_size=batch_size, shuffle=True, device=device)
            
            logging.info(f"{ds_name} Fold {fold} Results: {res}")

    logging.info("All folds processed.")

run_script()
