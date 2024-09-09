
# import necessary libraries for Chemberta model
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, AdamW, get_linear_schedule_with_warmup , BertModel

class ChemBERTaPT(nn.Module):
    def __init__(self):
        super(ChemBERTaPT, self).__init__()
        self.model_name = "DeepChem/ChemBERTa-77M-MTR"
        self.chemberta = RobertaModel.from_pretrained(self.model_name)

    def forward(self, input_ids, attention_mask):
        bert_output = self.chemberta(input_ids=input_ids, attention_mask=attention_mask)
        return bert_output[1]