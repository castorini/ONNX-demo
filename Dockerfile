FROM h379wang/pytorch-mxnet-jvm:latest
# https://github.com/HeliWang/pytorch-mxnet-jvm
RUN pip3 install gensim==3.5.0
RUN wget http://mirror.csclub.uwaterloo.ca/apache/spark/spark-2.3.1/spark-2.3.1-bin-hadoop2.7.tgz
RUN tar xvf spark-2.3.1-bin-hadoop2.7.tgz
RUN rm spark-2.3.1-bin-hadoop2.7.tgz l_mkl_2018.2.199.tgz
COPY . /app/
WORKDIR /app/
