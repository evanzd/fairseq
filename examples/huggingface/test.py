from transformers import RobertaTokenizer

from fairseq_model import RobertaClassifier
from fairseq.models.roberta import RobertaModel

model_path = './roberta.base'

sentence = 'hello world!'
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaClassifier.from_pretrained(
    model_path, {0: 'negative', 1: 'neutral', 2: 'positive'}).eval()
roberta = RobertaModel.from_pretrained(model_path)

out1 = tokenizer([sentence], return_tensors='pt')
out2 = roberta.encode(sentence)
assert (out1['input_ids'][0] == out2).all()

print(model(out1['input_ids'], out1['attention_mask']))
