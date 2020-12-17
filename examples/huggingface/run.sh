#!/bin/bash
wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz
tar -xzvf roberta.base.tar.gz && rm -f roberta.base.tar.gz

wget -P roberta.base https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
wget -P roberta.base https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe

python test.py
