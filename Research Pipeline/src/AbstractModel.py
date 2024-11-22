from imports import *
from MoleculeDataset import MoleculeDataset

class AbstractModel(ABC, nn.Module):
    def __init__(self):
        super(AbstractModel, self).__init__()
        self.early_stopping_patience = 5
        self.delta = 0.0001

    @abstractmethod
    def forward(self, cpe, esm, fegs, gae, cbae, morgan_fingerprints, chemical_descriptors):
        pass
        
    def train_model(self, fold, num_epochs, dataset, optimizer, criterion, 
                    batch_size=32, device='cuda', num_workers=5):
        
        train_dataset = MoleculeDataset(dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        print(f'Start training {fold} for {num_epochs} epochs !')
        for epoch in range(num_epochs):
            start_time = time.time()
            self.train()
            train_loss = 0.0
            all_labels = []
            all_outputs = []
            for (batch_chemprop_features, batch_esm_features, batch_fegs_features, batch_gae_features,
                 batch_chemberta_features, batch_morgan, batch_chem_desc ,batch_labels) in train_loader:
                # Move tensors to the configured device
                batch_chemprop_features = batch_chemprop_features.to(device)
                batch_chemberta_features = batch_chemberta_features.to(device)
                batch_esm_features = batch_esm_features.to(device)
                batch_fegs_features = batch_fegs_features.to(device)
                batch_gae_features = batch_gae_features.to(device)
                batch_morgan = batch_morgan.to(device)
                batch_chem_desc = batch_chem_desc.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()
                outputs = self(batch_chemprop_features , batch_esm_features, batch_fegs_features,
                               batch_gae_features, batch_chemberta_features, batch_morgan, batch_chem_desc) 
                loss = criterion(outputs.squeeze(), batch_labels)
                
                train_loss += loss.item()
                all_labels.extend(batch_labels.cpu().numpy())
                all_outputs.extend(outputs.squeeze().detach().cpu().numpy())
        
                loss.backward()
                optimizer.step()
                        
            # Calculate metrics
            train_loss /= len(train_loader)
            train_auc = roc_auc_score(all_labels, all_outputs)
            train_aupr = average_precision_score(all_labels, all_outputs)
            precision = precision_score(all_labels, (np.array(all_outputs) > 0.8).astype(int))
            sensitivity = recall_score(all_labels, (np.array(all_outputs) > 0.8).astype(int))
            
            tn, fp, fn, tp = confusion_matrix(all_labels, (np.array(all_outputs) > 0.8).astype(int)).ravel()
            specificity = tn / (tn + fp)
            
            end_time = time.time()
            epoch_time = (end_time - start_time) / 60
            print(f"Epoch {epoch+1} Time: {epoch_time:.2f} min, Train Loss: {train_loss:.5f}, Train AUC: {train_auc:.5f}, "
                  f"Train AUPR: {train_aupr:.5f}, Precision: {precision:.5f}, Sensitivity: {sensitivity:.5f}, Specificity: {specificity:.5f}")

        
    def train_val_model(self, fold, num_epochs, dataset, optimizer, criterion, 
                    batch_size=32, device='cuda', num_workers=5):
        best_val_auc = float('-inf')
        no_improve_epochs = 0
        smiles_col = 'smiles'
        smiles_df = dataset[["smiles"]].drop_duplicates()
        
        dc_dataset = dc.data.DiskDataset.from_dataframe(smiles_df, ids=smiles_col)
        scaffoldsplitter = dc.splits.ScaffoldSplitter()

        frac_train = 0.8
        frac_test = 0.2
    
        train_dataset, val_dataset = scaffoldsplitter.train_test_split(dc_dataset, frac_train=frac_train)
    
        train_indices = train_dataset.ids
        val_indices = val_dataset.ids
    
        train_subset = dataset[dataset[smiles_col].isin(train_indices)].copy()
        val_subset = dataset[dataset[smiles_col].isin(val_indices)].copy()

        train_dataset = MoleculeDataset(train_subset)
        val_dataset = MoleculeDataset(val_subset)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        for epoch in range(num_epochs):
            all_preds = []
            all_labels = []
            start_time = time.time()
            self.train()
            running_loss = 0.0
            for (batch_chemprop_features, batch_esm_features, batch_fegs_features, batch_gae_features,
                 batch_chemberta_features, batch_morgan, batch_chem_desc ,batch_labels) in train_loader:
                batch_chemprop_features = batch_chemprop_features.to(device)
                batch_chemberta_features = batch_chemberta_features.to(device)
                batch_esm_features = batch_esm_features.to(device)
                batch_fegs_features = batch_fegs_features.to(device)
                batch_gae_features = batch_gae_features.to(device)
                batch_morgan = batch_morgan.to(device)
                batch_chem_desc = batch_chem_desc.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()
                outputs = self(batch_chemprop_features , batch_esm_features, batch_fegs_features,
                               batch_gae_features, batch_chemberta_features, batch_morgan, batch_chem_desc)                 
                loss = criterion(outputs.squeeze(), batch_labels)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                all_labels.extend(batch_labels.cpu().numpy())
                all_preds.extend(outputs.squeeze().detach().cpu().numpy())
            
            train_auc = roc_auc_score(all_labels, all_preds)
            val_loss, val_accuracy, val_auc = self.validate_model(val_loader, criterion, device)
            end_time = time.time()
            epoch_time = (end_time - start_time) / 60
            
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {(running_loss/len(train_loader)):.4f}, Train AUC: {train_auc:.4f}, Validation AUC: {val_auc:.4f}, Epoch Time: {epoch_time:.4f}')
            # Check whether val_auc > best_val_auc + delta
            if val_auc > best_val_auc + self.delta:
                best_val_auc = val_auc
                train_epoch = epoch+1
                no_improve_epochs = 0 
                print(f"Current best val_auc -> {val_auc:.5f}, at epoch {epoch+1}")
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= self.early_stopping_patience:
                    print(f"Stopping early at epoch {epoch+1}")
                    break

        print(f'Train the model for -> {train_epoch}, best validation auc: {best_val_auc:.5f}')
                

    def test_model(self, test_dataset, criterion, batch_size, device, num_workers):
        test_dataset = MoleculeDataset(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.eval()
    
        test_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_outputs = []
    
        with torch.no_grad():
            for (batch_chemprop_features, batch_esm_features, batch_fegs_features, batch_gae_features,
                 batch_chemberta_features, batch_morgan, batch_chem_desc, batch_labels) in test_loader:
                
                batch_chemprop_features = batch_chemprop_features.to(device)
                batch_chemberta_features = batch_chemberta_features.to(device)
                batch_esm_features = batch_esm_features.to(device)
                batch_fegs_features = batch_fegs_features.to(device)
                batch_gae_features = batch_gae_features.to(device)
                batch_morgan = batch_morgan.to(device)
                batch_chem_desc = batch_chem_desc.to(device)
                batch_labels = batch_labels.to(device)
    
                outputs = self(batch_chemprop_features, batch_esm_features, batch_fegs_features,
                               batch_gae_features, batch_chemberta_features, batch_morgan, batch_chem_desc)
                loss = criterion(outputs.squeeze(), batch_labels)
                test_loss += loss.item()
    
                all_labels.extend(batch_labels.cpu().numpy())
                all_outputs.extend(outputs.squeeze().cpu().numpy())
    
                predicted = (outputs.squeeze() > 0.8).float()
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
    
        # Compute metrics
        test_loss /= len(test_loader)
        accuracy = correct / total
        test_auc = roc_auc_score(all_labels, all_outputs)
        test_aupr = average_precision_score(all_labels, all_outputs)
        precision = precision_score(all_labels, (np.array(all_outputs) > 0.8).astype(int))
        sensitivity = recall_score(all_labels, (np.array(all_outputs) > 0.8).astype(int))
        
        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(all_labels, (np.array(all_outputs) > 0.8).astype(int)).ravel()
        specificity = tn / (tn + fp)
    
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        print(f"Test AUPR: {test_aupr:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Test Specificity: {specificity:.4f}")

        return round(test_loss, 5), round(test_auc, 5), round(test_aupr, 5), round(precision, 5), round(sensitivity, 5), round(specificity, 5)

    def validate_model(self, val_loader, criterion, device):
        self.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_outputs = []

        with torch.no_grad():
            for (batch_chemprop_features, batch_esm_features, batch_fegs_features, batch_gae_features,
                 batch_chemberta_features, batch_morgan, batch_chem_desc ,batch_labels) in val_loader:

                batch_chemprop_features = batch_chemprop_features.to(device)
                batch_chemberta_features = batch_chemberta_features.to(device)
                batch_esm_features = batch_esm_features.to(device)
                batch_fegs_features = batch_fegs_features.to(device)
                batch_gae_features = batch_gae_features.to(device)
                batch_morgan = batch_morgan.to(device)
                batch_chem_desc = batch_chem_desc.to(device)
                batch_labels = batch_labels.to(device)

                outputs = self(batch_chemprop_features , batch_esm_features, batch_fegs_features,
                               batch_gae_features, batch_chemberta_features, batch_morgan, batch_chem_desc)                
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