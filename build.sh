#!/bin/bash -x
 
cd $(dirname `ls -l $0 | awk '{print $NF;}'`)
wk_dir=`pwd`

is_all_build=0
if [ $# > 1 ]; then
  is_all_build=$1
fi

GCC=`which gcc`
GXX=`which g++`

# step1: compile protobuf
#sh ${wk_dir}/message/compile-pb.sh $wk_dir/test

# step2: build openmit by cmake
if [ "X$is_all_build" != "X1" ] || [ ! -d $wk_dir/build ]; then
  rm -rf $wk_dir/build || true
  mkdir -p $wk_dir/build 
  cd $wk_dir/build
  cmake -D CMAKE_C_COMPILER=${GCC} -D CMAKE_CXX_COMPILER=${GXX} $wk_dir
else
  cd $wk_dir/build
fi

make 
