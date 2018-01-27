#!/bin/bash -x
cd $(dirname `ls -ls $0 | awk '{print $NF;}'`)/..
wk_dir=`pwd`

dir="/home/zhouyongsdzh/workspace/openmit/openmit"
num_workers=1
num_servers=1

  #train_path = example/data/libfm/train.txt \
  #valid_path = example/data/libfm/test.txt \
  #test_path = example/data/libfm/test.txt \
  #out_path = ${dir}/example/data/out/libfm
GCC_LIB_PATH="/data1/lantian/open_source/lib"
LD_LIBRARY_PATH=$GCC_LIB_PATH:$HADOOP_HOME/lib/native:$JAVA_HOME/jre/lib/amd64/server:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

#model="lr"
#if model=

${wk_dir}/tracker/dmlc-submit \
  --cluster local \
  --num-servers ${num_servers} \
  --num-workers ${num_workers} \
  ${wk_dir}/bin/openmit example/mit_ps_mf.conf \
  train_path = example/data/libmf/mf.txt.e1 \
  valid_path = example/data/libmf/mf.txt.e1 \
  test_path = example/data/libmf/mf.txt.e1 \
  out_path = example/data/out/libmf
