export MKL_NUM_THREADS=12
source /opt/intel/bin/compilervars.sh intel64
mvn package
mvn exec:java -Dexec.mainClass=scalakim.App  -Dexec.args="word2vec.txt 300 49"