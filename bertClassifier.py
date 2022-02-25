from torch import nn
from transformers import BertModel, AutoModel

class BertClassifier(nn.Module):

    def __init__(self):

        super(BertClassifier, self).__init__()

        #self.bert = BertModel.from_pretrained('bert-base-cased')
        self.bert = AutoModel.from_pretrained("vinai/bertweet-base")
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

        """# Freeze the BERT model
        for param in self.bert.parameters():
            param.requires_grad = False"""

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        linear_output = self.linear(pooled_output)
        final_layer = self.relu(linear_output)

        return final_layer