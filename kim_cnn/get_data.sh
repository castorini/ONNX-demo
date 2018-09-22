#!/bin/bash
mkdir data
cd data
wget https://git.uwaterloo.ca/jimmylin/Castor-data/raw/master/embeddings/word2vec/GoogleNews-vectors-negative300.bin
python3 ../convert_bin2txt.py GoogleNews-vectors-negative300.bin word2vec.txt
wget https://git.uwaterloo.ca/jimmylin/Castor-data/raw/master/datasets/SST/stsa.fine.dev -O stsa.fine.dev.tsv
wget https://git.uwaterloo.ca/jimmylin/Castor-data/raw/master/datasets/SST/stsa.fine.phrases.train -O stsa.fine.phrases.train.tsv
wget https://git.uwaterloo.ca/jimmylin/Castor-data/raw/master/datasets/SST/stsa.fine.test -O stsa.fine.test.tsv