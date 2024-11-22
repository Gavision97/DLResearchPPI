from imports import *

class CustomButinaSplitter:
    def __init__(self, smiles_col='smiles', label_col='label', cutoff=0.6):
        self.smiles_col = smiles_col
        self.label_col = label_col
        self.cutoff = cutoff

    def split_dataset(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        unique_smiles = df[self.smiles_col].drop_duplicates().reset_index(drop=True)
        print(f"Number of unique SMILES: {len(unique_smiles)}")

        # Compute fingerprints for unique SMILES (only once)
        unique_smiles_list = unique_smiles.tolist()
        fingerprints_dict = self._compute_fingerprints(unique_smiles_list)

        # Cluster the fingerprints (only once)
        clusters = self._cluster_fingerprints(fingerprints_dict, cutoff=self.cutoff)

        # Prepare for splitting
        splits = []
        for random_seed in [11, 7, 8, 1, 4]:
            print(f"Random Seed: {random_seed}")

            # Perform initial train/test split using Butina clustering
            train, test_combined = self._custom_butina_splitter(
                df,
                clusters,
                unique_smiles_list,
                smiles_col=self.smiles_col,
                frac_train=0.78,
                frac_test=0.22,
                random_state=random_seed
            )

            # Now split test_combined into validation and test sets using stratified split
            labels = test_combined[self.label_col]

            # Split test_combined into validation and test sets (each 50% of test_combined)
            valid, test = train_test_split(
                test_combined,
                test_size=0.5,
                random_state=random_seed,  # Use the same random seed for reproducibility
                stratify=labels
            )

            splits.append((train, valid, test))

            # Verify that the splits do not overlap
            train_indices = set(train.index)
            valid_indices = set(valid.index)
            test_indices = set(test.index)
            assert len(train_indices & valid_indices) == 0
            assert len(train_indices & test_indices) == 0
            assert len(valid_indices & test_indices) == 0

            # Calculate actual split sizes
            total_molecules = len(df)
            actual_train_frac = len(train) / total_molecules
            actual_valid_frac = len(valid) / total_molecules
            actual_test_frac = len(test) / total_molecules

            print(f"Train size: {len(train)} ({actual_train_frac:.2%}), "
                  f"Valid size: {len(valid)} ({actual_valid_frac:.2%}), "
                  f"Test size: {len(test)} ({actual_test_frac:.2%})")

        return splits

    def _compute_fingerprints(self, unique_smiles_list: List[str], radius: int = 2, n_bits: int = 1024) -> Dict[str, AllChem.rdchem.Mol]:
        """Private method to compute Morgan fingerprints for a list of unique SMILES strings."""
        fingerprints = {}
        for smi in unique_smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            else:
                # Handle invalid SMILES strings by adding a zeroed fingerprint
                zero_mol = Chem.MolFromSmiles('')
                fp = AllChem.GetMorganFingerprintAsBitVect(zero_mol, radius, nBits=n_bits)
            fingerprints[smi] = fp
        return fingerprints

    def _cluster_fingerprints(self, fingerprints: Dict[str, AllChem.rdchem.Mol], cutoff: float) -> List[Tuple[int]]:
        """Private method to cluster fingerprints using the Butina algorithm."""
        fps = list(fingerprints.values())
        n_fps = len(fps)
        dists = []
        for i in range(1, n_fps):
            sims = AllChem.DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend([1 - x for x in sims])
        clusters = Butina.ClusterData(dists, n_fps, cutoff, isDistData=True)
        return clusters

    def _custom_butina_splitter(
        self,
        df: pd.DataFrame,
        clusters: List[Tuple[int]],
        unique_smiles_list: List[str],
        smiles_col: str = 'smiles',
        frac_train: float = 0.78,
        frac_test: float = 0.22,
        random_state: int = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Private method to perform a custom Butina split into train and test sets."""
        rng = np.random.default_rng(random_state)

        # Shuffle clusters to introduce randomness
        cluster_indices = list(range(len(clusters)))
        rng.shuffle(cluster_indices)

        total_molecules = len(df)
        desired_train_size = int(frac_train * total_molecules)
        desired_test_size = total_molecules - desired_train_size

        train_smiles = set()
        test_smiles = set()
        train_count = 0
        test_count = 0

        # Iterate over clusters and assign to splits
        for idx in cluster_indices:
            cluster = clusters[idx]
            cluster_smiles = [unique_smiles_list[i] for i in cluster]
            cluster_molecule_count = df[df[smiles_col].isin(cluster_smiles)].shape[0]

            remaining_train = desired_train_size - train_count
            remaining_test = desired_test_size - test_count

            remaining = {'train': remaining_train, 'test': remaining_test}
            possible_splits = {k: v for k, v in remaining.items() if v > 0}

            if not possible_splits:
                total_counts = {'train': train_count, 'test': test_count}
                split = min(total_counts, key=total_counts.get)
            else:
                split = max(possible_splits, key=possible_splits.get)

            if split == 'train':
                train_smiles.update(cluster_smiles)
                train_count += cluster_molecule_count
            else:
                test_smiles.update(cluster_smiles)
                test_count += cluster_molecule_count

        train = df[df[smiles_col].isin(train_smiles)].copy()
        test_combined = df[df[smiles_col].isin(test_smiles)].copy()

        return train, test_combined
