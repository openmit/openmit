#!/bin/bash -x
cd $(dirname `ls -ls $0 | awk '{print $NF;}'`)/..
wk_dir=`pwd`

num_workers=5
num_servers=2

  #train_path = example/data/libsvm/train \
  #valid_path = example/data/libsvm/agaricus.txt.test \
  #test_path = example/data/libsvm/agaricus.txt.test \
  #model_dump = example/data/model_out/model_dump \
  #model_binary = example/data/model_out/model_binary

${wk_dir}/tracker/dmlc-submit \
  --cluster local \
  --num-servers ${num_servers} \
  --num-workers ${num_workers} \
  ${wk_dir}/bin/openmit example/mit_ps.conf \
  train_path = example/data/libfm/train.txt \
  valid_path = example/data/libfm/test.txt \
  test_path = example/data/libfm/test.txt \
  model_dump = example/data/model_out/model_dump.libfm \
  model_binary = example/data/model_out/model_binary.libfm
