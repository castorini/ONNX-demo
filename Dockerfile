FROM h379wang/pytorch-mxnet-jvm:latest
# https://github.com/HeliWang/pytorch-mxnet-jvm
COPY . /app/
WORKDIR /app/