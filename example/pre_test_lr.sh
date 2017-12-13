#!/bin/bash -x
cd $(dirname `ls -ls $0 | awk '{print $NF;}'`)/..
wk_dir=`pwd`

num_workers=1
num_servers=1

  #train_path = example/data/libsvm/train \
  #valid_path = example/data/libsvm/agaricus.txt.test \
  #test_path = example/data/libsvm/agaricus.txt.test \
  #model_dump = example/data/model_out/model_dump \
  #model_binary = example/data/model_out/model_binary

GCC_LIB_PATH="/data1/lantian/open_source/lib"
LD_LIBRARY_PATH=$GCC_LIB_PATH:$HADOOP_HOME/lib/native:$JAVA_HOME/jre/lib/amd64/server:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
${wk_dir}/tracker/dmlc-submit \
  --cluster local \
  --num-servers ${num_servers} \
  --num-workers ${num_workers} \
  ${wk_dir}/bin/openmit example/pre_test_lr.conf \
  train_path = example/data/libsvm/agaricus.txt.train \
  valid_path = example/data/libsvm/agaricus.txt.test \
  test_path = example/data/libsvm/agaricus.txt.test \
  model_dump = example/data/model_out/model_dump.libfm \
  model_binary = example/data/model_out/model_binary.libfm
