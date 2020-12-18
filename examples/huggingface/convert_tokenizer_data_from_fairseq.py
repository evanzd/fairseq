import os
import shutil
import json

from transformers import RobertaTokenizer

input_path = 'input'
output_path = 'output'

os.makedirs(output_path, exist_ok=True)

# load
with open(input_path + '/encoder.json', 'r', encoding='utf8') as f:
    encoder = json.load(f)
decoder = {v:k for k,v in encoder.items()}

with open(input_path + '/dict.txt', 'r', encoding='utf8') as f:
    dictionary = [x.strip().split() for x in f]

# map fairseq encoder+dict to huggingface tokenizer
encoder_new = [
    ('<s>', 0),
    ('<pad>', 1),
    ('</s>', 2),
    ('<unk>', 3)
]
for key, count in dictionary:
    if not key.isnumeric():
        token = key
    else:
        token = decoder[int(key)]
    encoder_new.append((token, len(encoder_new)))
encoder_new.append(('<mask>', len(encoder_new)))
encoder_new = dict(sorted(encoder_new, key=lambda x: x[1]))

# save
with open(output_path + '/vocab.json', 'w', encoding='utf8') as f:
    json.dump(encoder_new, f)

shutil.copyfile(input_path + '/vocab.bpe', output_path + '/merges.txt')

# check
sentence = 'hello world!'
our = RobertaTokenizer.from_pretrained(output_path)
huggingface = RobertaTokenizer.from_pretrained('roberta-base')
assert (our(sentence, return_tensors='pt')['input_ids'] \
        == huggingface(sentence, return_tensors='pt')['input_ids']).all()

print('passed!')
