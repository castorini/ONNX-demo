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
chmod +x get_data.sh
./get_data.sh
```

Note that the convert_bin2txt.py process of converting the whole GoogleNews-vectors-negative300.bin embedding can take a long time. For testing purpose if you only need a small subset of the embedding, please execute the following commands:
```
cd kim_cnn
chmod +x get_data_subset.sh
./get_data_subset.sh
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
chmod +x prediction.sh
./prediction.sh
```
----------

### Current Issues
(1) ONNX doesnot support dynamic shape -- currently length of one sentence is limited to <= 100
