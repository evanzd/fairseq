import torch.nn as nn

from fairseq import hub_utils

from transformers import RobertaTokenizer as RT


class RobertaClassifier(nn.Module):

    def __init__(self, config, task, model, id2label):
        super().__init__()
        self.config = config
        self.task = task
        self.model = model
        self.id2label = id2label
        self.fc = nn.Linear(config.encoder_embed_dim, len(id2label))

    @staticmethod
    def from_pretrained(path, id2label, **kwargs):
        x = hub_utils.from_pretrained(path)
        return RobertaClassifier(x['args'], x['task'], x['models'][0], id2label)

    def save_pretrained(path):
        torch.save({
            'args': self.config,
            'task': self.task,
            'models': [self.model.state_dict()]
        }, path)

    def forward(self, input_ids, attention_mask):
        hidden = self.model.encoder.extract_features(input_ids)[0]
        cls_hidden = hidden[:, 0]
        return self.fc(cls_hidden),  # NOTE: transformers==3.5


class RobertaTokenizer:

    @staticmethod
    def from_pretrained(path, **kwargs):
        tokenizer = RT(path + '/encoder.json', path + '/vocab.bpe',
                       model_max_length=512)
        # fairseq.data.dictionary
        # TODO: without overwriting original mappings
        for idx, token in enumerate(['<s>', '<pad>', '</s>', '<unk>']):
            tokenizer.encoder[token] = idx
            tokenizer.decoder[idx] = token
        return tokenizer
