#!/bin/bash -x
 
cd $(dirname `ls -l $0 | awk '{print $NF;}'`)
wk_dir=`pwd`

# step1: compile protobuf
#sh ${wk_dir}/message/compile-pb.sh $wk_dir/test

# step2: build spacex by cmake
if [ -d ${wk_dir}/build ]; then
	cd ${wk_dir}/build && make clean && rm -rf ./*
else
	mkdir ${wk_dir}/build && cd ${wk_dir}/build
fi

GCC=`which gcc`
GXX=`which g++`

cmake -D CMAKE_C_COMPILER=${GCC} -D CMAKE_CXX_COMPILER=${GXX} ..
make 
