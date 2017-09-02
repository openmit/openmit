#!/bin/bash -x
cd $(dirname `ls -ls $0 | awk '{print $NF;}'`)/..
wk_dir=`pwd`

num_workers=1
num_servers=5

${wk_dir}/tracker/dmlc-submit \
  --cluster local \
  --num-servers ${num_servers} \
  --num-workers ${num_workers} \
  ${wk_dir}/bin/openmit examples/mit_ps.conf \
  train_path = examples/data/libsvm/train \
  valid_path = examples/data/libsvm/agaricus.txt.test \
  test_path = examples/data/libsvm/agaricus.txt.test \
  model_out = examples/data/model_out

