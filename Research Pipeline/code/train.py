import os
import time
import warnings
import sys
import logging

logging.basicConfig(filename='output.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Initialized log file successfully ...")
logging.info("...")

warnings.filterwarnings('ignore')

from ResearchModels import *
from ResearchHelpers import *
from MoleculeDatasets import *

import chemprop
from chemprop import data, featurizers, models

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

logging.info("...")
from transformers import RobertaTokenizer, RobertaModel

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect


def run_script():
    device = check_device()

    # Write logs
    logging.info("Device details:")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    logging.info(f"CUDA version: {torch.version.cuda}")
    logging.info(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA'}")

    checkpoints_path = os.path.join('pt_chemprop_checkpoint', 'fold_0', 'model_0', 'checkpoints', 'best-epoch=49-val_loss=0.12.ckpt')
    batch_size = 32
    
    data_sets = {
        'multi_ppim_folds_2_0.8': os.path.join('datasets', 'finetune_dataset', 'multi_ppim_folds_2_0.8'),
        'DLIP_folds_2_0.8': os.path.join('datasets', 'finetune_dataset', 'DLIP_folds_2_0.8'),
        'DLIP_folds_3_0.9': os.path.join('datasets', 'finetune_dataset', 'DLIP_folds_3_0.9')
    }
    
    uniprot_mapping = pd.read_csv(os.path.join('datasets', 'idmapping_unip.tsv'), delimiter="\t")
    #ppi_features_df = pd.read_csv(os.path.join('datasets', 'merged_ppi_features.csv'))
    
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
            dataframes[df_name] =data_augmentation_with_uniprots_order_switchings(dataframes[df_name])
            #dataframes[df_name] = merge_datasets(dataframes[df_name], ppi_features_df)

        # Define fold indices
        folds = [1, 2, 3, 4, 5]

        for fold in folds:
            if(fold == 2 and ds_name == 'DLIP_folds_2_0.8'):
              continue # skip fold 2 of dlip_2_0.8 for now'
              
            if(ds_name == 'multi_ppim_folds_2_0.8'):
              batch_size = 64
              if(fold == 1 or fold == 4 or fold == 5):
                dropout = 0.1
                weight_decay=1e-5
              else:
                dropout = 0.8
                weight_decay=1e-3
                
            if(ds_name == 'DLIP_folds_2_0.8'):
              dropout = 0.1
              weight_decay = 1e-5
              batch_size = 64

              
            if(ds_name == 'DLIP_fold_3_0,9'):
              batch_size = 124
              if(fold == 1):
                dropout = 0.5
                weight_decay = 1e-4
              elif(fold == 2 or fold == 5):
                dropout = 0.3
                weight_decay = 1e-4
              else:
                dropout = 0.1
                weight_decay = 1e-5
                
            train_df = dataframes[f'train_fold{fold}_df']
            test_df = dataframes[f'test_fold{fold}_df']
            
            logging.info(f"Processing {ds_name} Fold {fold}...")
            model = generate_model(checkpoints_path, batch_size=batch_size, dropout=dropout)
            
            # Cross-validation and training
            res = model.cross_validate(train_df, num_folds=10, num_epochs=3,
                          batch_size=batch_size, learning_rate=0.0001, weight_decay=weight_decay,
                          shuffle=True, device=device)

            # Testing the model
            _, test_auc = model.test_model(test_df, criterion=nn.BCELoss(),
                          batch_size=batch_size, shuffle=True, device=device)
            
            logging.info(f"{ds_name} Fold {fold} ,AUC: {test_auc}")

    logging.info("All folds processed.")

logging.info("Start run_script() ...")
run_script()
