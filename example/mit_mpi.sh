#!/bin/bash -x

cd $(dirname `ls -ls $0 | awk '{print $NF;}'`)/..
wk_dir=`pwd`

if [ $# -lt 1 ]; then
  num_workers=1
else
  num_workers=$1
fi

echo "num_workers: $num_workers"
${wk_dir}/tracker/dmlc-submit \
  --cluster local \
  --num-workers ${num_workers} \
  ${wk_dir}/bin/openmit $wk_dir/example/mit_mpi.conf \
  train_path = $wk_dir/example/data/libsvm/train \
  valid_path = $wk_dir/example/data/libsvm/agaricus.txt.test \
  test_path = $wk_dir/example/data/libsvm/agaricus.txt.test \
  out_path = $wk_dir/example/data/out
