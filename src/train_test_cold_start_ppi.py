from imports import * 
from helpers import *
from AUVG_PPI import AUVG_PPI
from MoleculeDataset import MoleculeDataset

logging.basicConfig(filename='output_train_test_bilinear_exp1.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("--- exp1 : with self and bilinear - with mlps after self - same hyperparameters as described in the paper ---")

if torch.cuda.is_available():
    logging.info(f"GPU is available.")
    device = "cuda"
else:
    logging.info(f"GPU is not available. Using CPU instead.")
    device = "cpu"


# Train & test on cold start data
uniprot_mapping = pd.read_csv(os.path.join('datasets', 'idmapping_unip.tsv'), delimiter = "\t")
ds_folder_path = os.path.join('datasets', 'test_dataset', 'train_test_5_0.75')
all_files = os.listdir(ds_folder_path)
dataframes = {}

# Read each CSV file into a dataframe and store it in the dictionary
for file in all_files:
    file_path = os.path.join(ds_folder_path, file)
    df = pd.read_csv(file_path)
    df_name = file.replace('_5_0.75.csv', '_df')
    dataframes[df_name] = df

for df_name in dataframes.keys():
    dataframes[df_name] = convert_uniprot_ids(dataframes[df_name], uniprot_mapping)

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



def avg_expirements_auc(num_epochs_list, n):
    res_dict = {f'exp{i+1}': [] for i in range(n)}
    for exp_num in range(1, n+1):
        logging.info(f"Starting Experiment {exp_num}")
        
        for fold_num in range(1, 6): 
            fold_name = f'fold{fold_num}'
            train_fold, test_fold = dataframes[f'train_{fold_name}_df'], dataframes[f'test_{fold_name}_df'] 
            num_epochs = num_epochs_list[fold_num - 1] 
            model = generate_model(batch_size=64, dropout=0.3)
            model.train_model(fold_name, num_epochs=num_epochs, dataset=train_fold,
                                          optimizer=optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-3),
                                          criterion=nn.BCEWithLogitsLoss(),
                                          batch_size=64, device=device, num_workers=16)
    
            res_tuple = model.test_model(test_fold, 
                                      criterion=nn.BCEWithLogitsLoss(), batch_size=64, 
                                      device=device, num_workers=16)
    
            res_dict[f'exp{exp_num}'].append(res_tuple)
    return res_dict
    
nel = [23, 32, 23, 54, 30]
n = 10
ten_exp_res_dict = avg_expirements_auc(num_epochs_list = nel,
                                       n=n)

for exp, res_tuple in ten_exp_res_dict.items():
    logging.info(f"{exp}: {res_tuple}")