# Model conversion of Kim Model
### PyTorch -> ONNX -> MxNet (Python) -> MxNet (Scala)

----------
### Build the docker image and run a container
```
docker build --tag h379wang/kim-model-conversion .
docker run -it h379wang/kim-model-conversion:latest /bin/bash
```
----------
### PyTorch inference demo of Kim Model
```
python3 pytorch_inference.py
```
----------
### MXNet Python - Kim ONNX model importing and model inference
```
python3 mxnet_inference.py
```
----------
### MXNet Scala - Kim model artifacts importing and model inference
```
<will push later>
```
----------

### Current Issues
(1) ONNX doesnot support dynamic shape
