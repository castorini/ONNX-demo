FROM h379wang/pytorch-mxnet-jvm:latest
# https://github.com/HeliWang/pytorch-mxnet-jvm
RUN pip3 install gensim==3.5.0
COPY . /app/
WORKDIR /app/