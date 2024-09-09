######## MoleculeDatasets ########
## Use MoleculeDataset_ if training with data augmentation & using prob in testing
## Else, use MoleculeDataset


# for training with prob
class MoleculeDataset_(Dataset):
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


##########################################################################################################################################3

# for training without prob
class MoleculeDataset(Dataset):
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
