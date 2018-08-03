#!/bin/bash
mkdir data
cd data
wget http://ocp59jkku.bkt.clouddn.com/sst-1.zip
unzip sst-1.zip
wget https://raw.githubusercontent.com/Impavidity/kim_cnn/master/data/stsa.fine.dev.tsv
wget https://raw.githubusercontent.com/Impavidity/kim_cnn/master/data/stsa.fine.phrases.train.tsv
wget https://raw.githubusercontent.com/Impavidity/kim_cnn/master/data/stsa.fine.test.tsv
wget https://github.com/Impavidity/kim_cnn/raw/master/data/word2vec.sst-1.pt
wget https://raw.githubusercontent.com/Impavidity/kim_cnn/master/vector_preprocess.py