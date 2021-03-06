import numpy as np
import mxnet as mx
import argparse
import mxnet.contrib.onnx as onnx_mxnet
from collections import namedtuple
from embedding import embedding
from embedding import load_embedding

# load model and initialization
batch_size, channel, single_sentence_length, embedding_size = 1, 1, 49, 300
random_input = np.random.rand(
    batch_size,
    channel,
    single_sentence_length,
    embedding_size)
sym, arg_params, aux_params = onnx_mxnet.import_model('kim_model.onnx')
mod = mx.mod.Module(
    symbol=sym,
    data_names=['1'],
    context=mx.cpu(),
    label_names=None)
mod.bind(for_training=False, data_shapes=[
         ('1', random_input.shape)], label_shapes=None)
mod.set_params(
    arg_params=arg_params,
    aux_params=aux_params,
    allow_missing=True)

def mod_inference(matrix):
    random_input = np.random.rand(
        batch_size,
        channel,
        single_sentence_length,
        embedding_size)
    sym, arg_params, aux_params = onnx_mxnet.import_model('kim_model.onnx')
    mod = mx.mod.Module(
        symbol=sym,
        data_names=['1'],
        context=mx.cpu(),
        label_names=None)
    mod.bind(for_training=False, data_shapes=[
             ('1', random_input.shape)], label_shapes=None)
    mod.set_params(
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True)

    Batch = namedtuple('Batch', ['data'])
    mod.forward(Batch([mx.nd.array(matrix)]))
    output = mod.get_outputs()[0]
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", dest="embedding",
                    help="Read embedding from the path", metavar="FILE", required=True)
    parser.add_argument('--seed', nargs='?', dest="seed", const=1, type=int) # set default random seed to 1
    args = parser.parse_args()
    
    load_embedding(args.embedding, args.seed)
    inference("in his first stab at the form , jacquot takes a slightly anarchic approach that works only sporadically .")