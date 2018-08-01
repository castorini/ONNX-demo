import sys
import os
import random
import numpy as np
import torch
from torchtext import data
from args import get_args
from SST1 import SST1Dataset
from utils import clean_str_sst
import torch.onnx
from torch.autograd import Variable

class KimCNN():
    def __init__(self):
        # Get default args from args.py before load the model
        self.args = get_args()
        self.args.trained_model = "./kim_cnn/saves/static_best_model_cpu.pt"
        self.args.trained_model_gpu = "./kim_cnn/saves/static_best_model_gpu.pt"
        self.args.mode = "static"
        self.args.cuda = False

        torch.manual_seed(self.args.seed)
        if not self.args.cuda:
            self.args.gpu = -1
        if torch.cuda.is_available() and self.args.cuda:
            torch.cuda.set_device(self.args.gpu)
            torch.cuda.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)

        self.model = None
        if self.args.cuda:
            self.model = torch.load(self.args.trained_model_gpu, map_location=lambda storage, location: storage.cuda(self.args.gpu))
        else:
            self.model = torch.load(self.args.trained_model, map_location=lambda storage, location: storage)

        self.TEXT  = data.Field(batch_first=True, lower=True, tokenize=clean_str_sst)
        self.LABEL = data.Field(sequential=False)

        train, dev, test = SST1Dataset.splits(self.TEXT, self.LABEL)
        self.dev_iter = data.Iterator(dev, batch_size=self.args.batch_size, device=self.args.gpu, train=False, repeat=False,
                                   sort=False, shuffle=False)

        train = SST1Dataset.splits(self.TEXT, self.LABEL)[0] # using SST-1 training set to build vocab
        self.TEXT.build_vocab(train, min_freq=2)
        self.LABEL.build_vocab(train)

    def predict(self, features):
        dict_list = [{ 'text': feature, 'label': None } for feature in features]
        self.model.eval()
        test_fields_list = [('label', self.LABEL), ('text', self.TEXT)]
        test_fields_json = {'label': ('label', self.LABEL), 'text': ('text', self.TEXT)}
        test_examples = [data.Example.fromdict(data = dict_data, fields = test_fields_json) for dict_data in dict_list]
        test_dataset = data.Dataset(examples=test_examples, fields = test_fields_list)
        data_batch = data.Batch(test_examples, dataset=test_dataset, device=self.args.gpu, train=False)
        scores  = self.model(data_batch.text)
        predicted_results = torch.max(scores, 1)[1].view(data_batch.label.size()).data
        return list(predicted_results)

    def vector_inference(self, input_tensor):
        return self.model.forward(input_tensor)

    def onnx_export(self, export_file):
        print("Saving model to ONNX...")
        x = torch.rand(1, 1, 49, 300)
        dummy_input = Variable(x, requires_grad=True)
        torch.onnx.export(self.model, dummy_input, export_file, verbose=True)
