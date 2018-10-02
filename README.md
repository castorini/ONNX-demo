# Model Conversion of Kim Model
### PyTorch -> ONNX -> MxNet (Python) -> MxNet (Scala)

----------
### Docker Image Build
```
docker build --tag h379wang/kim-model-conversion .
```
### Run Docker Container
Assume that from `ONNX-demo`, [Castor-data](https://git.uwaterloo.ca/jimmylin/Castor-data) is checked out in `..` . Assume the current folder is `ONNX-demo`.
```
docker run -it -v $(pwd)/../Castor-data:/data h379wang/kim-model-conversion:latest /bin/bash
```
### Model Conversion
PyTorch -> ONNX -> MXNet Python -> MXNet Scala
```
source /opt/intel/bin/compilervars.sh intel64
python3 export_models.py
# Move the generated metadata of scala version of Kim Model
mv *.params scala_inference/
mv *.json scala_inference/
```
----------
### Inference Under Embedding Subset for SST-1 Dataset
##### PyTorch Inference of Kim Model
```
python3 pytorch_inference.py --embedding /data/embeddings/word2vec/word2vec.sst-1
```
##### MXNet Python Inference of Kim Model
```
source /opt/intel/bin/compilervars.sh intel64
python3 mxnet_inference.py --embedding /data/embeddings/word2vec/word2vec.sst-1
```
#####  MXNet Scala Inference of Kim Model
```
cd scala_inference
chmod +x prediction.sh
./prediction.sh
```
----------
### Inference Under Full Word2Vec Embedding
##### Create Full Word2Vec Embedding
Note that the convert_bin2txt.py process of converting the whole GoogleNews-vectors-negative300.bin embedding can take a long time.
```
python3 /app/kim_cnn/convert_bin2txt.py \
        /data/embeddings/word2vec/GoogleNews-vectors-negative300.bin \
        /data/embeddings/word2vec/word2vec.txt
```
##### PyTorch Inference of Kim Model
```
python3 pytorch_inference.py --embedding /data/embeddings/word2vec/word2vec.txt
```
##### MXNet Python Inference of Kim Model
```
source /opt/intel/bin/compilervars.sh intel64
python3 mxnet_inference.py --embedding /data/embeddings/word2vec/word2vec.txt
```
#####  MXNet Scala Inference of Kim Model
```
cd scala_inference
chmod +x prediction_full_embedding.sh
./prediction_full_embedding.sh
```
### Current Issues
(1) ONNX doesnot support dynamic shape -- currently length of one sentence is limited to <= 100
