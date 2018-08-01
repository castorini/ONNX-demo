import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import mxnet as mx
import mxnet.contrib.onnx as onnx_mxnet
from collections import namedtuple
import random

cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cur_path + '/kim_cnn')
from kim_cnn import KimCNN

random_input = np.random.rand(1, 1, 49, 300) # the batch size is fixed to 1

# Step1: Export to kim_model.onnx
model = KimCNN()
model.onnx_export("kim_model.onnx") 

# Step2: Read onnx graph and convert to mxnet symbols
sym, arg_params, aux_params = onnx_mxnet.import_model('kim_model.onnx')

mod = mx.mod.Module(symbol=sym, data_names=['1'], context=mx.cpu(), label_names=None)
mod.bind(for_training=False, data_shapes=[('1', random_input.shape)], label_shapes=None)
mod.set_params(arg_params=arg_params, aux_params=aux_params, allow_missing=True)

# Step3: Testing - Same input to pytorch and mxnet. The outputs from pytorch and mxnet should be the same

print("-------------Testing-------------")
#print("ONNX_MXNET Iutput:")
mxnet_input = mx.nd.array(random_input)
#print(mxnet_input)

print("ONNX_MXNET Output:")
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])
mod.forward(Batch([mxnet_input]))

print(mod.get_outputs())

#print("PyTorch Iutput:")
pytorch_input = torch.FloatTensor(random_input)
#print(pytorch_input)

print("PyTorch Output:")
print(model.vector_inference(Variable(pytorch_input)).data.numpy())

print("-------------Export MXNet model artifacts -------------")

# Step4: MxNet Export (for inference in JVM)
# save a model to kim-symbol.json and kim.params
prefix = "kim"
mod._symbol.save("%s-symbol.json" % prefix)
mod.save_params("%s-0100.params" % prefix)
print("Successfully exported to %s-symbol.json and %s-0100.params" % (prefix, prefix))
