# Model conversion of Kim Model
### PyTorch -> ONNX -> MxNet (Python) -> MxNet (Scala)

----------
### Build the docker image and run a container
```
docker build --tag h379wang/kim-model-conversion .
docker run -it h379wang/kim-model-conversion:latest /bin/bash
```
----------
### Get Datasets
```
cd kim_cnn
./get_data.sh
```
----------
### PyTorch inference demo of Kim Model
```
python3 pytorch_inference.py
```
----------
### MXNet Python - Kim ONNX model importing and model inference
```
source /opt/intel/bin/compilervars.sh intel64
python3 mxnet_inference.py
```
----------
### MXNet Scala - Kim model artifacts importing and model inference
```
cp ./kim_cnn/data/word2vec.txt ./scala_inference
cd scala_inference
./prediction.sh
```
----------

### Current Issues
(1) ONNX doesnot support dynamic shape -- currently length of one sentence is limited to <= 100
