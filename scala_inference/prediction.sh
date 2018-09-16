export MKL_NUM_THREADS=12
source /opt/intel/bin/compilervars.sh intel64
mvn package exec:java -Dexec.mainClass=scalakim.App

