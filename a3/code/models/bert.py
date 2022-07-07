import torch.nn as nn
import torch
from transformers import BertModel

class BertCased(nn.Module):

    def __init__(self, bert_dim, output_dim, device = 'cpu', dropout = 0.3):

        super(BertCased, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.bert = self.bert.to(device)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(bert_dim, output_dim, device = device)
        self.relu = nn.ReLU()

    def forward(self, inputs):

        _, pooled_output = self.bert(input_ids = inputs['ids'], attention_mask = inputs['mask'], return_dict = False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer