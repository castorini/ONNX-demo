# Kim Model in ONNX
##### PyTorch -> ONNX -> MXNet (Python) -> MXNet (Scala) -> Spark Distribued Inference (Scala)
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
##### PyTorch Inference
```
python3 pytorch_inference.py --embedding /data/embeddings/word2vec/word2vec.sst-1 --seed 1
```
##### MXNet Python Inference
```
source /opt/intel/bin/compilervars.sh intel64
python3 mxnet_inference.py --embedding /data/embeddings/word2vec/word2vec.sst-1 --seed 1
```
#####  MXNet Scala Inference
```
cd /app/scala_inference/
export LD_LIBRARY_PATH="/app/incubator-mxnet/scala-package/native/linux-x86_64-cpu/target:$LD_LIBRARY_PATH"
export MXNET_JAR_PATH="/app/incubator-mxnet/scala-package/core/target/*.jar,/app/incubator-mxnet/scala-package/spark/target/*.jar"
export PATH="/app/spark-2.3.1-bin-hadoop2.7/bin:$PATH"
mvn package

spark-submit --conf spark.driver.extraLibraryPath="${LD_LIBRARY_PATH}" \
--conf spark.executorEnv.LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
--jars ${MXNET_JAR_PATH} \
--class scalakim.Inference \
target/kim-1.0-SNAPSHOT.jar \
/data/embeddings/word2vec/word2vec.sst-1 300 49 1
```

##### MXNet Spark Distributed Inference (standalone mode)
```
# Remove possible previous results
rm -r /app/scala_inference/spark_output/ 2> /dev/null

spark-submit --conf spark.driver.extraLibraryPath="${LD_LIBRARY_PATH}" \
--conf spark.executorEnv.LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
--jars /app/incubator-mxnet/scala-package/core/target/*.jar,/app/incubator-mxnet/scala-package/spark/target/*.jar \
--class scalakim.SparkInference \
target/kim-1.0-SNAPSHOT.jar \
/data/embeddings/word2vec/word2vec.sst-1 300 49 /data/⁨datasets⁩/⁨SST⁩/stsa.fine.dev /app/scala_inference/spark_output/

cat /app/scala_inference/spark_output/* | head -20
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

#####  MXNet Scala Inference
```
cd /app/scala_inference/
export LD_LIBRARY_PATH="/app/incubator-mxnet/scala-package/native/linux-x86_64-cpu/target:$LD_LIBRARY_PATH"
export MXNET_JAR_PATH="/app/incubator-mxnet/scala-package/core/target/*.jar,/app/incubator-mxnet/scala-package/spark/target/*.jar"
export PATH="/app/spark-2.3.1-bin-hadoop2.7/bin:$PATH"
mvn package

spark-submit --conf spark.driver.extraLibraryPath="${LD_LIBRARY_PATH}" \
--conf spark.executorEnv.LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
--jars ${MXNET_JAR_PATH} \
--class scalakim.Inference \
target/kim-1.0-SNAPSHOT.jar \
/data/embeddings/word2vec/word2vec.txt 300 49 1
```

##### MXNet Spark Distributed Inference (standalone mode)
```
# Remove possible previous results
rm -r /app/scala_inference/spark_output/ 2> /dev/null

spark-submit --conf spark.driver.extraLibraryPath="${LD_LIBRARY_PATH}" \
--conf spark.executorEnv.LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
--jars /app/incubator-mxnet/scala-package/core/target/*.jar,/app/incubator-mxnet/scala-package/spark/target/*.jar \
--class scalakim.SparkInference \
target/kim-1.0-SNAPSHOT.jar \
/data/embeddings/word2vec/word2vec.txt 300 49 /data/⁨datasets⁩/⁨SST⁩/stsa.fine.dev /app/scala_inference/spark_output/

cat /app/scala_inference/spark_output/* | head -20
```

### Current Issues
(1) ONNX doesnot support dynamic shape -- currently length of one sentence is limited to <= 100
