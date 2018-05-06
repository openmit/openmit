#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$SCRIPT_DIR/../.. 
GEN_DIR=$SCRIPT_DIR/gen


source $PROJECT_DIR/conf/base.conf
source $PROJECT_DIR/conf/deps_version.conf 

# protobuf
sh $SCRIPT_DIR/compile_linux_protobuf.sh

export PROTOBUF_HOME=$GEN_DIR/protobuf/$platform/$proto_version 
export PATH=$PROTOBUF_HOME/bin:$PATH
export LD_LIBRARY_PATH=$PROTOBUF_HOME/lib:$LD_LIBRARY_PATH

# grpc

echo $SCRIPT_DIR
