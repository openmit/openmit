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

${wk_dir}/tracker/dmlc-submit \
  --cluster local \
  --num-servers ${num_servers} \
  --num-workers ${num_workers} \
  ${wk_dir}/bin/openmit example/mit_ps.conf \
  train_path = example/data/libsvm/train \
  valid_path = example/data/libsvm/agaricus.txt.test \
  test_path = example/data/libsvm/agaricus.txt.test \
  out_path = example/data/out/libsvm
