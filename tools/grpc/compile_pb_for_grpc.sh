#!/bin/bash -e

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

# compile proto recursive
function compile_proto_recursive() {
  if [ $# -lt 1 ]; then
    echo "You need to specify 'dir' args."
    exit 1
  fi
  for elem in `ls $1`
  do 
    dir_or_file=$1/$elem 
    postfix=".proto"
    if [ -d $dir_or_file ]; then
      compile_proto_recursive $dir_or_file
    fi
    
    if [[ "${dir_or_file: -6}" == "$postfix" ]]; then 
      $protoc -I=$1 --cpp_out=$1 --grpc_out=$1 --plugin=protoc-gen-grpc=$grpc_cpp_plugin $dir_or_file
    fi
  done
}

# exec compile proto operation
compile_proto_recursive $PROJECT_DIR/tools/grpc 


echo $LD_LIBRARY_PATH
echo grpc: `locate libgrpc++.so.1`
