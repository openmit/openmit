#!/bin/bash -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR=$SCRIPT_DIR/..
OPENMIT_DIR=$PROJECT_DIR/openmit 

source $PROJECT_DIR/conf/base.conf 
source $PROJECT_DIR/conf/deps.conf 

protobuf_dir=$PROJECT_DIR/third_party/makefile/gen/protobuf/$platform/$proto_version
protoc=$protobuf_dir/bin/protoc 
protoc_lib=$protobuf_dir/lib

export LD_LIBRARY_PATH=$protoc_lib
export DYLD_LIBRARY_PATH=$protoc_lib

cpp_out=$1
java_out=$wk_dir/message/java_out

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
      $protoc -I=$1 --cpp_out=$1 $dir_or_file
    fi
  done
}

# exec compile proto operation
compile_proto_recursive $PROJECT_DIR/openmit

