#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR=$SCRIPT_DIR/../..

source $PROJECT_DIR/conf/base.conf 
source $PROJECT_DIR/conf/deps_version.conf 

# grpc cpp plugin
grpc_dir=$PROJECT_DIR/third_party/makefile/gen/grpc 
export PATH=$grpc_dir/bin:$PATH 
export LD_LIBRARY_PATH=$grpc_dir/lib:$LD_LIBRARY_PATH 
export DYLD_LIBRARY_PATH=$grpc_dir/lib:$DYLD_LIBRARY_PATH 

grpc_cpp_plugin=`which grpc_cpp_plugin`

# protobuf
protobuf_dir=$PROJECT_DIR/third_party/makefile/gen/protobuf/$platform/$proto_version
export PATH=$protobuf_dir/bin:$PATH 
export LD_LIBRARY_PATH=$protobuf_dir/lib:$LD_LIBRARY_PATH 
export DYLD_LIBRARY_PATH=$protobuf_dir/lib:$DYLD_LIBRARY_PATH 

protoc=`which protoc`
