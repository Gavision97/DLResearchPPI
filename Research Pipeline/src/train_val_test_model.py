from imports import *
from CustomButinaSplitter import *
from MoleculeDataset import MoleculeDataset
from helpers import *

def train_val_test_model(dataset, num_epochs, dropout, lr, weight_decay, criterion, 
                         batch_size=32, device='cuda', num_workers=5):
    all_folds_valid_metrics = []
    all_folds_test_metrics = []
    splits = []
    smiles_df = dataset[["smiles"]].drop_duplicates()
    smiles_col = 'smiles'
    
    # split dataset into 5 folds of (train, val, test) dataframes using custom butina splitter obj.
    butinaSplitter = CustomButinaSplitter()
    splits = butinaSplitter.split_dataset(dataset)
            
    for fold_number, (train_subset, val_subset, test_subset) in enumerate(splits, 1):
        PRINTC()
        logging.info(f"fold number {fold_number}")
        PRINTC()
        train_df, val_df, test_df = train_subset, val_subset, test_subset
        test_dataset = MoleculeDataset(test_df)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        bootstrap_valid_metrics = []
        bootstrap_test_metrics = []
                        
        for bootsrap in range(5):
            best_val_auc = float('-inf')
            epochs_without_improvement = 0
            early_stopping_patience = 5
            best_model_state_dict = None
            
            logging.info(f"bootsrap number: {bootsrap + 1}")
            model = generate_model(batch_size=batch_size, dropout=dropout)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            PRINTC()
            
            seed_train = fold_number*1000 + bootsrap + 1
            labels_list = train_df['label'].values
            train_b = resample(train_df, random_state=seed_train, stratify=labels_list)

            train_dataset = MoleculeDataset(train_b)                    
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            val_dataset = MoleculeDataset(val_df)                    
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            last_test_auc = 0  # Initialize the last test AUC for this fold
            for epoch in range(num_epochs):
                start_time = time.time()
                model.train()
                epoch_loss = 0
                all_preds = []
                all_labels = []
                running_loss = 0.0
                for (batch_chemprop_features, batch_esm_features, batch_fegs_features, batch_gae_features,
                     batch_chemberta_features, batch_morgan, batch_chem_desc ,batch_labels) in train_loader:

                    # Move all tensors to device
                    batches = [batch_chemprop_features, batch_esm_features, batch_fegs_features, batch_gae_features, batch_chemberta_features, batch_morgan, batch_chem_desc, batch_labels]
                    batches = [batch.to(device) for batch in batches]
                    batch_chemprop_features, batch_esm_features, batch_fegs_features, batch_gae_features, batch_chemberta_features, batch_morgan, batch_chem_desc, batch_labels = batches
                        
                    optimizer.zero_grad()
                    outputs = model(batch_chemprop_features , batch_esm_features, batch_fegs_features,
                                   batch_gae_features, batch_chemberta_features, batch_morgan, batch_chem_desc) 
        
                    loss = criterion(outputs.squeeze(), batch_labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
        
                    all_labels.extend(batch_labels.cpu().numpy())
                    all_preds.extend(outputs.squeeze().detach().cpu().numpy())
                    
                # Compute training metrics
                train_auc = roc_auc_score(all_labels, all_preds)
                train_aupr = average_precision_score(all_labels, all_preds)
                predicted_classes = (np.array(all_preds) >= 0.5).astype(int)
                cm = confusion_matrix(all_labels, predicted_classes)
                TN, FP, FN, TP = cm.ravel()
                train_sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
                train_specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
                train_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                        
                # Evaluate the model on the validation set
                all_val_labels = []
                all_val_outputs = []
                model.eval()
                with torch.no_grad():
                    for (batch_chemprop_features, batch_esm_features, batch_fegs_features, batch_gae_features,
                         batch_chemberta_features, batch_morgan, batch_chem_desc ,batch_labels) in val_loader:
                        
                        # Move all tensors to device
                        batches = [batch_chemprop_features, batch_esm_features, batch_fegs_features, batch_gae_features, batch_chemberta_features, batch_morgan, batch_chem_desc, batch_labels]
                        batches = [batch.to(device) for batch in batches]
                        batch_chemprop_features, batch_esm_features, batch_fegs_features, batch_gae_features, batch_chemberta_features, batch_morgan, batch_chem_desc, batch_labels = batches
                                        
                        outputs = model(batch_chemprop_features , batch_esm_features, batch_fegs_features,
                                       batch_gae_features, batch_chemberta_features, batch_morgan, batch_chem_desc)               
                        
                        all_val_labels.extend(batch_labels.cpu().numpy())
                        all_val_outputs.extend(outputs.squeeze().detach().cpu().numpy())
                        
                all_val_labels = np.array(all_val_labels)
                all_val_outputs = np.array(all_val_outputs)
                
                # Perform bootstrapping on predictions and labels (validation phase)
                current_b_aucs = []
                current_b_auprs = []
                current_b_sensitivities = []
                current_b_specificities = []
                current_b_precisions = []
                N_val = all_val_labels.shape[0]
                for b in range(1000):
                    seed_value = epoch * 1000 + b + (bootsrap+1)*1000  # or any function of your parameters
                    np.random.seed(seed_value)
                    indices = np.random.randint(0, N_val, size=N_val)
                    y_valid_pred_b = all_val_outputs[indices]
                    y_valid_b = all_val_labels[indices]
                    valid_auc = roc_auc_score(y_valid_b, y_valid_pred_b)
                    valid_aupr = average_precision_score(y_valid_b, y_valid_pred_b)
                    predicted_classes = (y_valid_pred_b >= 0.5).astype(int)
                    cm = confusion_matrix(y_valid_b, predicted_classes)
                    TN, FP, FN, TP = cm.ravel()
                    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
                    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
                    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                    current_b_aucs.append(valid_auc)
                    current_b_auprs.append(valid_aupr)
                    current_b_sensitivities.append(sensitivity)
                    current_b_specificities.append(specificity)
                    current_b_precisions.append(precision)
        
                mean_val_auc = np.mean(current_b_aucs)
                mean_val_aupr = np.mean(current_b_auprs)
                mean_val_sensitivity = np.mean(current_b_sensitivities)
                mean_val_specificity = np.mean(current_b_specificities)
                mean_val_precision = np.mean(current_b_precisions)
                end_time = time.time()
                epoch_time = (end_time - start_time) / 60
                logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {(epoch_loss/len(train_loader)):.4f}, '
                            f'Train AUC: {train_auc:.4f}, Train AUPR: {train_aupr:.4f}, '
                            f'Train Sensitivity: {train_sensitivity:.4f}, Train Specificity: {train_specificity:.4f}, '
                            f'Train Precision: {train_precision:.4f}, '
                            f'Mean Validation AUC: {mean_val_auc:.4f}, Mean Validation AUPR: {mean_val_aupr:.4f}, '
                            f'Mean Validation Sensitivity: {mean_val_sensitivity:.4f}, '
                            f'Mean Validation Specificity: {mean_val_specificity:.4f}, '
                            f'Mean Validation Precision: {mean_val_precision:.4f}, Epoch Time: {epoch_time:.4f}')

                # Early stopping logic based on validation AUC
                if mean_val_auc > best_val_auc:
                    best_val_auc = mean_val_auc
                    epochs_without_improvement = 0
                    # Save the best model state dict
                    best_model_state_dict = copy.deepcopy(model.state_dict())
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= early_stopping_patience:
                    logging.info("Early stopping triggered")
                    break
                    
            # Load the best model in order to evaluate it on the test set
            model.load_state_dict(best_model_state_dict)
            
            all_test_labels = []
            all_test_outputs = []
            model.eval()
            with torch.no_grad():
                for (batch_chemprop_features, batch_esm_features, batch_fegs_features, batch_gae_features,
                        batch_chemberta_features, batch_morgan, batch_chem_desc ,batch_labels) in test_loader:
                        
                    # Move all tensors to device
                    batches = [batch_chemprop_features, batch_esm_features, batch_fegs_features, batch_gae_features, batch_chemberta_features, batch_morgan, batch_chem_desc, batch_labels]
                    batches = [batch.to(device) for batch in batches]
                    batch_chemprop_features, batch_esm_features, batch_fegs_features, batch_gae_features, batch_chemberta_features, batch_morgan, batch_chem_desc, batch_labels = batches
                                        
                    outputs = model(batch_chemprop_features , batch_esm_features, batch_fegs_features,
                                   batch_gae_features, batch_chemberta_features, batch_morgan, batch_chem_desc)               
                        
                    all_test_labels.extend(batch_labels.cpu().numpy())
                    all_test_outputs.extend(outputs.squeeze().detach().cpu().numpy())
                        
            all_test_labels = np.array(all_test_labels)
            all_test_outputs = np.array(all_test_outputs)                 

            # Perform bootstrapping on predictions and labels (test phase)
            current_b_aucs = []
            current_b_auprs = []
            current_b_sensitivities = []
            current_b_specificities = []
            current_b_precisions = []
            N_test = all_test_labels.shape[0]
            for b in range(1000):
                seed_value = epoch * 1000 + b + (bootsrap+1)*1000  # or any function of your parameters
                np.random.seed(seed_value)
                indices = np.random.randint(0, N_test, size=N_test)
                y_test_pred_b = all_test_outputs[indices]
                y_test_b = all_test_labels[indices]
                test_auc = roc_auc_score(y_test_b, y_test_pred_b)
                test_aupr = average_precision_score(y_test_b, y_test_pred_b)
                predicted_classes = (y_test_pred_b >= 0.5).astype(int)
                cm = confusion_matrix(y_test_b, predicted_classes)
                TN, FP, FN, TP = cm.ravel()
                sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
                specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                current_b_aucs.append(test_auc)
                current_b_auprs.append(test_aupr)
                current_b_sensitivities.append(sensitivity)
                current_b_specificities.append(specificity)
                current_b_precisions.append(precision)
        
            mean_test_auc = np.mean(current_b_aucs)
            mean_test_aupr = np.mean(current_b_auprs)
            mean_test_sensitivity = np.mean(current_b_sensitivities)
            mean_test_specificity = np.mean(current_b_specificities)
            mean_test_precision = np.mean(current_b_precisions)
            logging.info(f'Bootstrap {bootsrap}, Mean Test AUC: {mean_test_auc:.4f}, Mean Test AUPR: {mean_test_aupr:.4f}, '
                        f'Mean Test Sensitivity: {mean_test_sensitivity:.4f}, Mean Test Specificity: {mean_test_specificity:.4f}, '
                        f'Mean Test Precision: {mean_test_precision:.4f}')

            # Store the best validation and test metrics for this bootstrap
            bootstrap_valid_metrics.append({
                'auc': best_val_auc,
                'aupr': mean_val_aupr,
                'sensitivity': mean_val_sensitivity,
                'specificity': mean_val_specificity,
                'precision': mean_val_precision
            })
            bootstrap_test_metrics.append({
                'auc': mean_test_auc,
                'aupr': mean_test_aupr,
                'sensitivity': mean_test_sensitivity,
                'specificity': mean_test_specificity,
                'precision': mean_test_precision
            })
        
        # Compute mean validation and test metrics for the current fold
        current_fold_mean_valid_metrics = {
            'auc': np.mean([m['auc'] for m in bootstrap_valid_metrics]),
            'aupr': np.mean([m['aupr'] for m in bootstrap_valid_metrics]),
            'sensitivity': np.mean([m['sensitivity'] for m in bootstrap_valid_metrics]),
            'specificity': np.mean([m['specificity'] for m in bootstrap_valid_metrics]),
            'precision': np.mean([m['precision'] for m in bootstrap_valid_metrics])
        }
        current_fold_mean_test_metrics = {
            'auc': np.mean([m['auc'] for m in bootstrap_test_metrics]),
            'aupr': np.mean([m['aupr'] for m in bootstrap_test_metrics]),
            'sensitivity': np.mean([m['sensitivity'] for m in bootstrap_test_metrics]),
            'specificity': np.mean([m['specificity'] for m in bootstrap_test_metrics]),
            'precision': np.mean([m['precision'] for m in bootstrap_test_metrics])
        }
        logging.info(f"Fold {fold_number} Mean Validation Metrics: {current_fold_mean_valid_metrics}")
        logging.info(f"Fold {fold_number} Mean Test Metrics: {current_fold_mean_test_metrics}")
        all_folds_valid_metrics.append(current_fold_mean_valid_metrics)
        all_folds_test_metrics.append(current_fold_mean_test_metrics)
    
    # Compute final mean metrics across all folds
    final_mean_valid_metrics = {
        'auc': np.mean([m['auc'] for m in all_folds_valid_metrics]),
        'aupr': np.mean([m['aupr'] for m in all_folds_valid_metrics]),
        'sensitivity': np.mean([m['sensitivity'] for m in all_folds_valid_metrics]),
        'specificity': np.mean([m['specificity'] for m in all_folds_valid_metrics]),
        'precision': np.mean([m['precision'] for m in all_folds_valid_metrics])
    }
    final_mean_test_metrics = {
        'auc': np.mean([m['auc'] for m in all_folds_test_metrics]),
        'aupr': np.mean([m['aupr'] for m in all_folds_test_metrics]),
        'sensitivity': np.mean([m['sensitivity'] for m in all_folds_test_metrics]),
        'specificity': np.mean([m['specificity'] for m in all_folds_test_metrics]),
        'precision': np.mean([m['precision'] for m in all_folds_test_metrics])
    }

    PRINTC()               
    logging.info(f"Final Mean Validation Metrics across all folds: {final_mean_valid_metrics}")
    logging.info(f"Validation Metrics for all folds: {all_folds_valid_metrics}")
    logging.info(f"Final Mean Test Metrics across all folds: {final_mean_test_metrics}")
    logging.info(f"Test Metrics for all folds: {all_folds_test_metrics}")