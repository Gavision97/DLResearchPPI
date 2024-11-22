from imports import *
from AUVG_PPI import AUVG_PPI

def convert_uniprot_ids(dataset, mapping_df):
    # Create a dictionary from the mapping dataframe
    mapping_dict = mapping_df.set_index('From')['Entry'].to_dict()

    # Map the uniprot_id1 and uniprot_id2 columns to their respective Entry values
    dataset['uniprot_id1'] = dataset['uniprot_id1'].map(mapping_dict)
    dataset['uniprot_id2'] = dataset['uniprot_id2'].map(mapping_dict)
    return dataset.drop_duplicates()

def generate_model(batch_size,
                  dropout) -> nn.Module:
    model = AUVG_PPI(dropout).to(device)

    PRINTM('Generated model successfully !')
    return model


def generate_model_(batch_size,
                  dropout) -> nn.Module:
    ## Use that option when adding fine-tune option (both chemport & chemBERTa)
    checkpoints_path = os.path.join('pt_chemprop_checkpoint_wmfp', 'model_0', 'checkpoints', 'best-epoch=20-val_loss=0.06.ckpt')
    pretrained_chemprop_model = PretrainedChempropModel(checkpoints_path, batch_size).to(device)
    chemberta_model = ChemBERTaPT().to(device)
    model = AUVG_PPI(pretrained_chemprop_model, chemberta_model, dropout).to(device)

    PRINTM('Generated model successfully !')
    return model