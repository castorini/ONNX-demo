#!/bin/bash
mkdir data
cd data
wget https://git.uwaterloo.ca/jimmylin/Castor-data/raw/master/embeddings/word2vec/GoogleNews-vectors-negative300.bin
python3 ../convert_bin2txt.py GoogleNews-vectors-negative300.bin word2vec.txt
wget https://raw.githubusercontent.com/Impavidity/kim_cnn/master/data/stsa.fine.dev.tsv
wget https://raw.githubusercontent.com/Impavidity/kim_cnn/master/data/stsa.fine.phrases.train.tsv
wget https://raw.githubusercontent.com/Impavidity/kim_cnn/master/data/stsa.fine.test.tsv
wget https://raw.githubusercontent.com/Impavidity/kim_cnn/master/vector_preprocess.py
