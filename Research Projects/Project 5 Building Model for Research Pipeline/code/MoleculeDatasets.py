######## MoleculeDatasets ########
## Use MoleculeDataset_ if training with data augmentation & using prob in testing
## Else, use MoleculeDataset

import os
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset

from transformers import RobertaTokenizer

import pandas as pd
import numpy as np



# for training with structure features
class MoleculeDataset_v2_2(Dataset):
    def __init__(self, ds_):
        self.data = ds_
        self.mapping_df = pd.read_csv(os.path.join('datasets', 'idmapping_unip.tsv'), delimiter="\t")
        self.esm = pd.read_csv(os.path.join('datasets', 'MolDatasets', 'esm_features.csv'))
        self.custom = pd.read_csv(os.path.join('datasets', 'MolDatasets', 'custom_features.csv'))
        self.fegs = pd.read_csv(os.path.join('datasets', 'MolDatasets', 'fegs_features.csv'))

        gae_path = f'GAE_FEATURES_WITH_PREDICTED_alpha_0.25.csv'
        self.gae = pd.read_csv(os.path.join('datasets', 'GAE', gae_path))
        gae_features_columns = self.gae.iloc[:, 9:509]
        gae_uniprot_column = self.gae[['From']].rename(columns={'From': 'UniProt_ID'})
        self.gae = pd.concat([gae_uniprot_column, gae_features_columns], axis=1)

        # Merge datasets
        self.esm_features_ppi = self.merge_datasets(self.data, self.esm).drop(columns=['smiles', 'label']).astype(np.float32)
        self.custom_features_ppi = self.merge_datasets(self.data, self.custom).drop(columns=['smiles', 'label']).astype(np.float32)
        self.fegs_features_ppi = self.merge_datasets(self.data, self.fegs).drop(columns=['smiles', 'label']).astype(np.float32)
        self.gae_features_ppi = self.merge_datasets(self.data, self.gae).drop(columns=['smiles', 'label']).astype(np.float32)

        # SMILES RDKit features - Morgan Fingerprints (r=4, nbits=1024) & chemical descriptors
        self.smiles_morgan_fingerprints = pd.read_csv(os.path.join('datasets', 'MolDatasets', 'smiles_morgan_fingerprints_dataset.csv'))
        self.smiles_chemical_descriptors = pd.read_csv(os.path.join('datasets', 'MolDatasets', 'smiles_chem_descriptors_mapping_dataset.csv'))

        # Necessary features for ChemBERTa model
        self.smiles_list = self.data['smiles'].tolist()
        self.tokenizer = RobertaTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
        self.encoded_smiles = self.tokenizer(self.smiles_list, truncation=True, padding=True, return_tensors="pt")

        # Protein structure feature extraction
        self.uniprots = ds_[['uniprot_id1', 'uniprot_id2']]

        # Preload all protein structure feature files into a dictionary
        self.protein_structure_dict = self.load_protein_structure_features()

    def merge_datasets(self, dataset, features_df):
        # Existing merging logic
        dataset = dataset.merge(features_df, how='left', left_on='uniprot_id1', right_on='UniProt_ID', suffixes=('', '_id1'))
        dataset = dataset.drop(columns=['UniProt_ID'])
        
        features_df_renamed = features_df.add_suffix('_id2')
        features_df_renamed = features_df_renamed.rename(columns={'UniProt_ID_id2': 'UniProt_ID'})
        dataset = dataset.merge(features_df_renamed, how='left', left_on='uniprot_id2', right_on='UniProt_ID', suffixes=('', '_id2'))
        dataset = dataset.drop(columns=['UniProt_ID', 'uniprot_id1', 'uniprot_id2'])
        
        return dataset.drop_duplicates()
        
    def __len__(self):
        return len(self.data)

    def load_protein_structure_features(self):
        """Preload all protein structure feature CSVs into a dictionary."""
        protein_structure_dir = os.path.join('datasets', 'MolDatasets', 'ProteinStructureFeatures')
        protein_structure_files = os.listdir(protein_structure_dir)
        protein_structure_dict = {}
        
        for file in protein_structure_files:
            uniprot_id_key = file.replace('_ifeature_omega.csv', '')
            protein_structure_dict[uniprot_id_key] = pd.read_csv(os.path.join(protein_structure_dir, file)).iloc[:, 1:].astype(np.float32)
        return protein_structure_dict

    def __getitem__(self, idx):
        smiles = self.data.iloc[idx, 0]
        label = np.array(self.data.iloc[idx, -1], dtype=np.float32)  
        esm_features = np.array(self.esm_features_ppi.iloc[idx].values, dtype=np.float32)
        custom_features = np.array(self.custom_features_ppi.iloc[idx].values, dtype=np.float32)
        fegs_features = np.array(self.fegs_features_ppi.iloc[idx].values, dtype=np.float32)
        gae_features = np.array(self.gae_features_ppi.iloc[idx].values, dtype=np.float32)

        input_ids = self.encoded_smiles["input_ids"][idx]
        attention_mask = self.encoded_smiles["attention_mask"][idx]

        # Retrieve precomputed RDKit Morgan fingerprints
        morgan_fingerprint = self.smiles_morgan_fingerprints.loc[self.smiles_morgan_fingerprints['SMILES'] == smiles].iloc[0, 1:].values.astype(np.float32)
        chemical_descriptors = self.smiles_chemical_descriptors.loc[self.smiles_chemical_descriptors['SMILES'] == smiles].iloc[0, 1:].values.astype(np.float32)

        # Protein structure feature extraction
        prot1_sfp, prot2_sfp = self.uniprots.iloc[idx, 0], self.uniprots.iloc[idx, 1]
        ans = self.checkProteins(prot1_sfp, prot2_sfp)
        
        if ans:
            prot1_sf = self.protein_structure_dict[ans]
            prot2_sf = self.protein_structure_dict[f'Q03164_with_{ans}']
        else:
            prot1_sf = self.protein_structure_dict[prot1_sfp]
            prot2_sf = self.protein_structure_dict[prot2_sfp]

        return (smiles, prot1_sf, prot2_sf, esm_features, custom_features, fegs_features, gae_features, 
                input_ids, attention_mask, morgan_fingerprint, chemical_descriptors, label)

    def checkProteins(self, unip1, unip2):
        if unip1 == 'Q03164':
            return unip2
        elif unip2 == 'Q03164':
            return unip1
        return None

    def collate_fn(self, batch):
        smiles, prot1_sfs, prot2_sfs, esm_features, custom_features, fegs_features, gae_features, input_ids, attention_masks, morgan_fingerprints, chemical_descriptors, labels = zip(*batch)

        # Convert lists of numpy arrays into tensors for protein structure features
        prot1_sfs = [torch.tensor(p.values) for p in prot1_sfs]
        prot2_sfs = [torch.tensor(p.values) for p in prot2_sfs]
        
        # Find the maximum number of rows (i.e., 0th dimension) for prot1 and prot2 individually
        max_len_prot1 = max([p.shape[0] for p in prot1_sfs])
        max_len_prot2 = max([p.shape[0] for p in prot2_sfs])

        # Pad the protein structure features along the 0th dimension (rows) for each protein individually
        prot1_sfs_padded = torch.stack([torch.nn.functional.pad(p, (0, 0, 0, max_len_prot1 - p.shape[0]), "constant", 0) for p in prot1_sfs])
        prot2_sfs_padded = torch.stack([torch.nn.functional.pad(p, (0, 0, 0, max_len_prot2 - p.shape[0]), "constant", 0) for p in prot2_sfs])
    
        esm_features = torch.tensor(esm_features, dtype=torch.float32)
        custom_features = torch.tensor(custom_features, dtype=torch.float32)
        fegs_features = torch.tensor(fegs_features, dtype=torch.float32)
        gae_features = torch.tensor(gae_features, dtype=torch.float32)
        
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        
        morgan_fingerprints = torch.tensor(morgan_fingerprints, dtype=torch.float32)
        chemical_descriptors = torch.tensor(chemical_descriptors, dtype=torch.float32)
        
        # Convert labels to a flat list of scalars and then to a tensor
        flattened_labels = [label.item() for label in labels]
        labels_tensor = torch.tensor(flattened_labels, dtype=torch.float32)
        
        return (smiles, prot1_sfs_padded, prot2_sfs_padded, esm_features, custom_features, fegs_features, gae_features, 
                input_ids, attention_masks, morgan_fingerprints, chemical_descriptors, labels_tensor)


##########################################################################################################################################3

# for training without prob
class MoleculeDataset_v2_1(Dataset):
    def __init__(self, ds_):
        # Initialize data and load other features
        self.data = ds_
        self.mapping_df = pd.read_csv(os.path.join('datasets', 'idmapping_unip.tsv'), delimiter = "\t")
        self.esm = pd.read_csv(os.path.join('datasets', 'MolDatasets', 'esm_features.csv'))
        self.custom = pd.read_csv(os.path.join('datasets', 'MolDatasets', 'custom_features.csv'))
        self.fegs = pd.read_csv(os.path.join('datasets', 'MolDatasets', 'fegs_features.csv'))

        gae_path = f'GAE_FEATURES_WITH_PREDICTED_alpha_0.25.csv'
        self.gae = pd.read_csv(os.path.join('datasets', 'GAE', gae_path))
        gae_features_columns = self.gae.iloc[:, 9:509]
        gae_uniprot_column = self.gae[['From']].rename(columns={'From': 'UniProt_ID'})
        self.gae = pd.concat([gae_uniprot_column, gae_features_columns], axis=1)
        

        # Merge datasets
        self.esm_features_ppi = self.merge_datasets(self.data, self.esm).drop(columns=['smiles', 'label']).astype(np.float32)
        self.custom_features_ppi = self.merge_datasets(self.data, self.custom).drop(columns=['smiles', 'label']).astype(np.float32)
        self.fegs_features_ppi = self.merge_datasets(self.data, self.fegs).drop(columns=['smiles', 'label']).astype(np.float32)
        self.gae_features_ppi = self.merge_datasets(self.data, self.gae).drop(columns=['smiles', 'label']).astype(np.float32)

        # SMILES RDKit features - Morgan Fingerprints (r=4, nbits=1024) & chemical descriptors
        self.smiles_morgan_fingerprints = pd.read_csv(os.path.join('datasets', 'MolDatasets', 'smiles_morgan_fingerprints_dataset.csv'))
        self.smiles_chemical_descriptors = pd.read_csv(os.path.join('datasets', 'MolDatasets', 'smiles_chem_descriptors_mapping_dataset.csv'))

        # Necessary features for ChemBERTa model
        self.smiles_list = self.data['smiles'].tolist()
        self.tokenizer = RobertaTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
        self.encoded_smiles = self.tokenizer(self.smiles_list, truncation=True, padding=True, return_tensors="pt")

    def merge_datasets(self, dataset, features_df):
        # Existing merging logic
        dataset = dataset.merge(features_df, how='left', left_on='uniprot_id1', right_on='UniProt_ID', suffixes=('', '_id1'))
        dataset = dataset.drop(columns=['UniProt_ID'])
        
        features_df_renamed = features_df.add_suffix('_id2')
        features_df_renamed = features_df_renamed.rename(columns={'UniProt_ID_id2': 'UniProt_ID'})
        dataset = dataset.merge(features_df_renamed, how='left', left_on='uniprot_id2', right_on='UniProt_ID', suffixes=('', '_id2'))
        dataset = dataset.drop(columns=['UniProt_ID', 'uniprot_id1', 'uniprot_id2'])
        
        return dataset.drop_duplicates()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        smiles = self.data.iloc[idx, 0]
        label = np.array(self.data.iloc[idx, -1], dtype=np.float32)  
        esm_features = np.array(self.esm_features_ppi.iloc[idx].values, dtype=np.float32)
        custom_features = np.array(self.custom_features_ppi.iloc[idx].values, dtype=np.float32)
        fegs_features = np.array(self.fegs_features_ppi.iloc[idx].values, dtype=np.float32)
        gae_features = np.array(self.gae_features_ppi.iloc[idx].values, dtype=np.float32)

        input_ids = self.encoded_smiles["input_ids"][idx]
        attention_mask = self.encoded_smiles["attention_mask"][idx]

        # Retrieve precomputed RDKit Morgan fingerprints
        morgan_fingerprint = self.smiles_morgan_fingerprints.loc[self.smiles_morgan_fingerprints['SMILES'] == smiles].iloc[0, 1:].values.astype(np.float32)
        chemical_descriptors = self.smiles_chemical_descriptors.loc[self.smiles_chemical_descriptors['SMILES'] == smiles].iloc[0, 1:].values.astype(np.float32)
        
        return (smiles, esm_features, custom_features, fegs_features, gae_features, 
                input_ids, attention_mask, morgan_fingerprint, chemical_descriptors, label)
                
################################################################################################################################################################


# for training with prob
class MoleculeDataset_v2_1_1(Dataset):
    def __init__(self, ds_):
        # Initialize data and load other features
        self.data = ds_
        self.mapping_df = pd.read_csv(os.path.join('datasets', 'idmapping_unip.tsv'), delimiter = "\t")
        self.esm = pd.read_csv(os.path.join('datasets', 'MolDatasets', 'esm_features.csv'))
        self.custom = pd.read_csv(os.path.join('datasets', 'MolDatasets', 'custom_features.csv'))
        self.fegs = pd.read_csv(os.path.join('datasets', 'MolDatasets', 'fegs_features.csv'))
        self.uniprots = self.data.drop(columns=['smiles', 'label'])

        gae_path = f'GAE_FEATURES_WITH_PREDICTED_alpha_0.25.csv'
        self.gae = pd.read_csv(os.path.join('datasets', 'GAE', gae_path))
        gae_features_columns = self.gae.iloc[:, 9:509]
        gae_uniprot_column = self.gae[['From']].rename(columns={'From': 'UniProt_ID'})
        self.gae = pd.concat([gae_uniprot_column, gae_features_columns], axis=1)
        
        self.uniprots = self.data.drop(columns=['smiles', 'label'])

        # Merge datasets
        self.esm_features_ppi = self.merge_datasets(self.data, self.esm).drop(columns=['smiles', 'label']).astype(np.float32)
        self.custom_features_ppi = self.merge_datasets(self.data, self.custom).drop(columns=['smiles', 'label']).astype(np.float32)
        self.fegs_features_ppi = self.merge_datasets(self.data, self.fegs).drop(columns=['smiles', 'label']).astype(np.float32)
        self.gae_features_ppi = self.merge_datasets(self.data, self.gae).drop(columns=['smiles', 'label']).astype(np.float32)

        # SMILES RDKit features - Morgan Fingerprints (r=4, nbits=1024) & chemical descriptors
        self.smiles_morgan_fingerprints = pd.read_csv(os.path.join('datasets', 'MolDatasets', 'smiles_morgan_fingerprints_dataset.csv'))
        self.smiles_chemical_descriptors = pd.read_csv(os.path.join('datasets', 'MolDatasets', 'smiles_chem_descriptors_mapping_dataset.csv'))

        # Necessary features for ChemBERTa model
        self.smiles_list = self.data['smiles'].tolist()
        self.tokenizer = RobertaTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
        self.encoded_smiles = self.tokenizer(self.smiles_list, truncation=True, padding=True, return_tensors="pt")

    def merge_datasets(self, dataset, features_df):
        # Existing merging logic
        dataset = dataset.merge(features_df, how='left', left_on='uniprot_id1', right_on='UniProt_ID', suffixes=('', '_id1'))
        dataset = dataset.drop(columns=['UniProt_ID'])
        
        features_df_renamed = features_df.add_suffix('_id2')
        features_df_renamed = features_df_renamed.rename(columns={'UniProt_ID_id2': 'UniProt_ID'})
        dataset = dataset.merge(features_df_renamed, how='left', left_on='uniprot_id2', right_on='UniProt_ID', suffixes=('', '_id2'))
        dataset = dataset.drop(columns=['UniProt_ID', 'uniprot_id1', 'uniprot_id2'])
        
        return dataset.drop_duplicates()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        smiles = self.data.iloc[idx, 0]
        label = np.array(self.data.iloc[idx, -1], dtype=np.float32)  
        uniprots_tuple = (self.data.iloc[idx, 1], self.data.iloc[idx, 2]) # tuple that hold uniprot_id1 and uniprots_id2 -> for prob in testing phase
        esm_features = np.array(self.esm_features_ppi.iloc[idx].values, dtype=np.float32)
        custom_features = np.array(self.custom_features_ppi.iloc[idx].values, dtype=np.float32)
        fegs_features = np.array(self.fegs_features_ppi.iloc[idx].values, dtype=np.float32)
        gae_features = np.array(self.gae_features_ppi.iloc[idx].values, dtype=np.float32)

        input_ids = self.encoded_smiles["input_ids"][idx]
        attention_mask = self.encoded_smiles["attention_mask"][idx]

        # Retrieve precomputed RDKit Morgan fingerprints
        morgan_fingerprint = self.smiles_morgan_fingerprints.loc[self.smiles_morgan_fingerprints['SMILES'] == smiles].iloc[0, 1:].values.astype(np.float32)
        chemical_descriptors = self.smiles_chemical_descriptors.loc[self.smiles_chemical_descriptors['SMILES'] == smiles].iloc[0, 1:].values.astype(np.float32)
        
        return (smiles, uniprots_tuple, esm_features, custom_features, fegs_features, gae_features, 
                input_ids, attention_mask, morgan_fingerprint, chemical_descriptors, label)

