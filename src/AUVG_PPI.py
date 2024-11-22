from imports import *
from DTF import DTF
from custom_self_attention import custom_self_attention
from AbstractModel import AbstractModel

class AUVG_PPI(AbstractModel):
    def __init__(self ,dropout):
        super(AUVG_PPI, self).__init__()
        self.dropout = dropout
        PRINTM('21/11 - exp4 -> with self & DTF')
        self.ppi_self_attention = custom_self_attention(embed_dim=256, num_heads=4, dropout=0.1)
        self.smiles_self_attention = custom_self_attention(embed_dim=256, num_heads=4, dropout=0.1)
        self.esm_dtf = DTF(channels=1280, r=8)
        self.gae_dtf = DTF(channels=500, r=4)
        self.fegs_dtf = DTF(channels=578, r=4)

        self.esm_mlp = nn.Sequential(
            nn.Linear(in_features=1280, out_features=640),
            nn.ReLU(),
            nn.BatchNorm1d(640),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=640, out_features=320),
            nn.ReLU(),
            nn.BatchNorm1d(320),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=320, out_features=256)
        )

        self.fegs_mlp = nn.Sequential(
            nn.Linear(in_features=578, out_features=256)
        )  

        self.gae_mlp = nn.Sequential(
            nn.Linear(in_features=500, out_features=256)
        )

        # MLP for ppi_features
        self.ppi_mlp = nn.Sequential(
            nn.Linear(in_features=256 * 3, out_features=512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=512, out_features=256)
        )
        
        self.fp_mlp = nn.Sequential(
            nn.Linear(in_features=1200, out_features=600), 
            nn.ReLU(),
            nn.BatchNorm1d(600),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=600, out_features=300),
            nn.ReLU(),
            nn.BatchNorm1d(300),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=300, out_features=256)
        )

        self.mfp_cd_mlp = nn.Sequential(
            nn.Linear(in_features=1024 + 194, out_features= 609),
            nn.ReLU(),
            nn.BatchNorm1d(609),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=609, out_features=300),
            nn.ReLU(),
            nn.BatchNorm1d(300),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=300, out_features=256)
        )

        self.chemberta_mlp = nn.Sequential(
            nn.Linear(in_features=384, out_features= 256)
        )

        self.smiles_mlp = nn.Sequential(
            nn.Linear(in_features=256 * 3 , out_features= 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=512, out_features=256)
        )

        self.additional_layers = nn.Sequential(
            nn.Linear(in_features=256 + 256, out_features=256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=128, out_features=1)
        )
        
    def forward(self, cpe, esm, fegs, gae, cbae, morgan_fingerprints, chemical_descriptors):
        cp_fingerprints = self.fp_mlp(cpe)
        cbae = self.chemberta_mlp(cbae)
        
        mfp_chem_descriptors = torch.cat([morgan_fingerprints, chemical_descriptors], dim=1)
        mfp_chem_descriptors = self.mfp_cd_mlp(mfp_chem_descriptors)
        
        smiles_embeddings = torch.stack([cp_fingerprints, cbae, mfp_chem_descriptors], dim=1).to(device)  # shape ->> (batch_size, 3, 256)
        smiles_embeddings = self.smiles_self_attention(smiles_embeddings)

        # Pass all PPI features through their DTF module and then through MLP layer
        # in order to reduce shape to (batch_size, 256)
        esm_embedding_p1, esm_embedding_p2 = torch.split(esm, esm.shape[1] // 2, dim=1)
        esm_embeddings = self.esm_dtf(esm_embedding_p1, esm_embedding_p2)
        esm_embeddings = self.esm_mlp(esm_embeddings)
        
        fegs_embedding_p1, fegs_embedding_p2 = torch.split(fegs, fegs.shape[1] // 2, dim=1)
        fegs_embeddings = self.fegs_dtf(fegs_embedding_p1, fegs_embedding_p2)
        fegs_embeddings = self.fegs_mlp(fegs_embeddings)
        
        gae_embedding_p1, gae_embedding_p2 = torch.split(gae, gae.shape[1] // 2, dim=1)
        gae_embeddings = self.gae_dtf(gae_embedding_p1, gae_embedding_p2)
        gae_embeddings = self.gae_mlp(gae_embeddings)
        
        # Stack all 3 ppi embeddings along a new dimension (3x256) 
        ppi_embeddings = torch.stack([esm_embeddings, fegs_embeddings, gae_embeddings], dim=1).to(device)  # shape ->> (batch_size, 3, 256)
        ppi_embeddings = self.ppi_self_attention(ppi_embeddings)

        flatten_smiles_embed = smiles_embeddings.flatten(start_dim=1)
        flatten_ppi_embed = ppi_embeddings.flatten(start_dim=1)

        smiles_embed = self.smiles_mlp(flatten_smiles_embed)
        ppi_embed = self.ppi_mlp(flatten_ppi_embed)
        combined_embeddings = torch.cat([smiles_embed, ppi_embed], dim=1)

        output = self.additional_layers(combined_embeddings)
        
        return output