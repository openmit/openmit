#!/bin/bash 

cd $(dirname `ls -l $0 | awk '{print $NF;}'`)/../..
export project_dir=`pwd`
example_dir=$project_dir/example 
distr_dir=$example_dir/distributed-running

source $project_dir/make/openmit.mk
source $distr_dir/env_yarn.sh

num_workers=4
worker_cores=4
worker_memory=4g

$project_dir/tracker/dmlc-submit \
  --cluster=yarn \
  --queue=$QUEUE \
  --num-workers=$num_workers \
  --worker-cores=$worker_cores \
  --worker-memory=$worker_memory \
  --jobname="openmit-yarn" \
  --ship-libcxx $GCC_LIB_PATH \
  --hdfs-tempdir $HDFS_TEMP_DIR \
  $project_dir/bin/openmit $example_dir/openmit.conf \
  train_path=$train_path \
  valid_path=$valid_path \
  model_dump=$model_dump \
  model_binary=$model_binary

code=$#
if [ "x$code" != "x0" ]; then
  echo "[ERROR] openmit yarn run failed!"
else
  echo "[INFO] openmit yarn run success!"
fi

echo "================ $0 finished. code: $code ==================="
