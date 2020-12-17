import json
import shutil
import torch
import torch.nn as nn

from fairseq import hub_utils


class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class RobertaClassifier(nn.Module):

    def __init__(self, config, task, model, path, id2label):
        super().__init__()
        self.path = path
        self.config = config
        self.task = task
        self.roberta = model
        self.config.id2label = id2label
        self.pooler = BertPooler(config.encoder_embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.encoder_embed_dim, len(id2label))

    @staticmethod
    def from_pretrained(path, id2label, **kwargs):
        x = hub_utils.from_pretrained(path)
        return RobertaClassifier(x['args'], x['task'], x['models'][0], path, id2label)

    def save_pretrained(self, path):
        torch.save({
            'args': self.config,
            'task': self.task,
            'models': [self.roberta.state_dict()]
        }, path + '/model.pt')
        shutil.copyfile(self.path + '/dict.txt', path + '/dict.txt')

    def forward(self, input_ids, attention_mask=None):
        hidden = self.roberta.encoder.extract_features(input_ids)[0]
        pooled_out = self.dropout(self.pooler(hidden))
        return self.classifier(pooled_out),  # NOTE: transformers==3.5
