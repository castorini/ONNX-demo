import numpy as np
import sys
import os
import random
import time
from collections import defaultdict

embedding_cache = dict()

def load_embedding(filename="word2vec.sst-1"):
    global embedding_cache
    header_found = False
    with open(filename, 'r', encoding="utf-8") as f:
        for line in f:
            parts = line.split(' ')
            if len(parts) < 50 and not header_found:
                print('Ignoring header row')
                header_found = True
                continue
            vec = list(map(float, parts[1:]))
            embedding_cache[parts[0]] = vec

def sentence_to_embedding(words, length, matrix, sentence_i):
    """
    if length > len(words), padding 0 to extend the sentence matrix
    """
    chr_i = 0
    for word in words:
        if chr_i == length: continue # exceed the limit of sentence length
        word_vec = embedding_cache.get(word, None)
        vec = None
        if word_vec == None:
            # random vector if word not in lookup
            # print('Not found in Cache: ' + word)
            vec = np.random.rand(300)
        else:
            vec = np.array(word_vec)
        matrix[sentence_i, 0, chr_i] = vec
        chr_i += 1
    
    return matrix # (num_sentencens, channel = 1, sentence_length, word_embdding_length = 300)

def embedding(sentences, length, single_sentence_length):
    """
    sentences is a 2d list, list<list<word>> for inference
    if length > len(sentences), padding 0 to extend the matrix
    """
    matrix = np.zeros((length, 1, single_sentence_length, 300))

    for i in range(len(sentences)):
        words = sentences[i]
        sentence_matrix = sentence_to_embedding(words, single_sentence_length, matrix, i)

    return matrix

# print(embedding(["a very well-made, funny and entertaining picture"], 100, 47))
