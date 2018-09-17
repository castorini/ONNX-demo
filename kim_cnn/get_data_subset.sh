#!/bin/bash
mkdir data
cd data
wget https://raw.githubusercontent.com/heliwang/SST1-dataset/master/word2vec.sst-1 -O word2vec.txt
wget https://raw.githubusercontent.com/Impavidity/kim_cnn/master/data/stsa.fine.dev.tsv
wget https://raw.githubusercontent.com/Impavidity/kim_cnn/master/data/stsa.fine.phrases.train.tsv
wget https://raw.githubusercontent.com/Impavidity/kim_cnn/master/data/stsa.fine.test.tsv
wget https://raw.githubusercontent.com/Impavidity/kim_cnn/master/vector_preprocess.py