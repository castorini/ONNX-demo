import time

import boto3
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

# change this to use other models
from model import KimCNN

# You will need to setup your own dynamoDB and table
client = boto3.client('dynamodb')
# and replace this with your own table's name
table_name = 'word2vec300d'
model = None

def sublist(l, batch_size):
    """
    Helper function to divide list into sublists
    """
    return [l[i:i+batch_size] for i in range(0,len(l), batch_size)]


def build_matrix(words, lookup):
    """
    Helper to turn words into matrix consist of word vectors
    """
    matrix = None
    for word in words:
        if word in lookup:
            vec_raw = lookup[word]
            vec = np.array([float(f['N']) for f in vec_raw])
        else:
            # random vector if word not in lookup
            print('Not found in Dynamo: ' + word)
            vec = np.random.rand(300)
        vec.resize(1, 1, vec.shape[0])
        if matrix is None:
            matrix = vec
        else:
            matrix = np.append(matrix, vec, axis=1)

    matrix.resize(1, matrix.shape[0], matrix.shape[1], matrix.shape[2])
    return matrix


def sentence_to_matrix(sentence):
    """
    Get word vectors for word in sentence and build a matrix with the vectors
    """
    words = sentence.split(' ')

    # request cannot contain duplicate keys. remove duplicates
    words_no_dup = list(set(words))
    read_batch_size = 100
    batches = sublist(words_no_dup, read_batch_size)
    wordvec_a = []
    for batch in batches:
        request = [{'word':{'S':word}} for word in batch]

        response = client.batch_get_item(
            RequestItems = {
                table_name: {
                    'Keys': request
                }
            }
        )

        wordvec_a = wordvec_a + [(d['word']['S'], d['vector']['L']) for d in response['Responses'][table_name]]

    lookup = dict(wordvec_a)

    return build_matrix(words, lookup)


def handler(event, context):
    t_start = time.time()
    event = json.loads(event['body'])
    sentence = event['input']

    t_start_build_sentence_embedding = time.time()
    input_matrix = sentence_to_matrix(sentence)
    t_duration_sentence_embedding = time.time() - t_start_build_sentence_embedding

    # load and run model
    # you may need to modify this based on your model definition
    t_start_load_model = time.time()
    global model
    if model is None:
        model = torch.load('static_best_model_cpu.pt')
        model.eval()
    t_duration_load_model = time.time() - t_start_load_model

    t_start_inference = time.time()
    torchIn = torch.from_numpy(input_matrix.astype(np.float32))
    torchIn = Variable(torchIn)
    output = model(torchIn)

    prediction = torch.max(output, 1)[1].view(1).data.tolist()[0]
    t_duration_inference = time.time() - t_start_inference
    t_duration = time.time() - t_start

    result = {
        'input': sentence,
        'prediction': prediction,
        'output': output.data.tolist()[0],
        't_overall': t_duration,
        't_sent_embedding': t_duration_sentence_embedding,
        't_load_model': t_duration_load_model,
        't_inference': t_duration_inference
    }

    # return result
    return {
        'statusCode': 200,
        'headers': { 'Content-Type': 'application/json' },
        'body': json.dumps(result)
    }

if __name__ == '__main__':
    event = {
        'body': json.dumps({
            'input': 'this is very good',
        })
    }
    print(handler(event, None))
