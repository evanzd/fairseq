#!/bin/bash
wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz
tar -xzvf roberta.base.tar.gz && rm -f roberta.base.tar.gz

python test.py
