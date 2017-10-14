#!/bin/bash -x
cd $(dirname `ls -ls $0 | awk '{print $NF;}'`)/..
wk_dir=`pwd`

num_workers=3
num_servers=2

${wk_dir}/tracker/dmlc-submit \
  --cluster local \
  --num-servers ${num_servers} \
  --num-workers ${num_workers} \
  ${wk_dir}/bin/openmit example/mit_ps.conf \
  train_path = example/data/libsvm/train \
  valid_path = example/data/libsvm/agaricus.txt.test \
  test_path = example/data/libsvm/agaricus.txt.test \
  model_dump = example/data/model_dump
