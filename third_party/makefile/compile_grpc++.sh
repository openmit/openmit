#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR=$SCRIPT_DIR/../..
GEN_DIR=$SCRIPT_DIR/gen 

# install reference address 
# https://github.com/grpc/grpc/blob/master/INSTALL.md

cd downloads 

#git clone -b $(curl -L https://grpc.io/release) https://github.com/grpc/grpc 
cd grpc
#git submodule update --init

mkdir -p $GEN_DIR/grpc || true
make && make install prefix=$GEN_DIR/grpc 

echo $GEN_DIR
