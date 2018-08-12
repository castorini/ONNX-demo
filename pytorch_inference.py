import numpy as np
import torch
import os
import sys
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from collections import namedtuple
from embedding import embedding
from embedding import load_embedding

# load model and initialization
batch_size, channel, single_sentence_length, embedding_size = 1, 1, 49, 300
cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cur_path + '/kim_cnn')
from kim_cnn import KimCNN
mod = KimCNN()


def mod_inference(matrix):
    Batch = namedtuple('Batch', ['data'])
    pytorch_input = torch.FloatTensor(matrix)
    output = mod.vector_inference(Variable(pytorch_input))
    return output


def inference(sentence):
    print("input sentence:")
    print(sentence)
    sentences = []
    words = sentence.split(' ')
    sentences.append(words)
    sentences_embedding = embedding(
        sentences, batch_size, single_sentence_length)
    print("input embedding:")
    print(sentences_embedding)
    output = mod_inference(sentences_embedding)
    print("output vector:")
    print(output)
    return output


load_embedding()
inference("in his first stab at the form , jacquot takes a slightly anarchic approach that works only sporadically .")
