from imports import *

## Helpers (e.g., pre-trained chemprop & chemBERTa for fine-tune tasks...

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
        self.smiles_loader = data.build_dataloader(self.smiles_dset, batch_size=self.batch_size, shuffle=False)
        
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

class ChemBERTaPT(nn.Module):
    def __init__(self):
        super(ChemBERTaPT, self).__init__()
        self.model_name = "DeepChem/ChemBERTa-77M-MTR"
        self.chemberta = RobertaModel.from_pretrained(self.model_name)

    def forward(self, input_ids, attention_mask):
        bert_output = self.chemberta(input_ids=input_ids, attention_mask=attention_mask)
        return bert_output[1]