from fairseq_model import RobertaTokenizer, RobertaClassifier


sentence = 'hello world!'
tokenizer = RobertaTokenizer.from_pretrained('./roberta.base')
model = RobertaClassifier.from_pretrained(
    './roberta.base', {0: 'negative', 1: 'neutral', 2: 'positive'})

print(tokenizer.decode(tokenizer.encode(sentence)))

out = tokenizer([sentence], return_tensors='pt')
print(out)

print(model(out['input_ids'], out['attention_mask']))
