#!/bin/bash

# Test Model
wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz
tar -xzvf roberta.base.tar.gz && rm -f roberta.base.tar.gz

python test.py

# Test Convert
mkdir -p input
wget -P input 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
wget -P input 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -P input 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'

python convert_tokenizer_data_from_fairseq.py
