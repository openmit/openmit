#!/bin/bash 

set -o pipefail
set -o errexit

cd $(dirname `ls -l $0 | awk '{print $NF;}'`)/../..
project_dir=`pwd`
example_dir=$project_dir/example 

source $project_dir/make/openmit.mk

num_workers=4
worker_cores=4

$project_dir/tracker/dmlc-submit \
  --cluster=yarn \
  --queue $QUEUE \
  --num-workers=$num_workers \
  --worker-cores=$worker-cores \
  --job-name="openmit-yarn"
  $project_dir/bin/openmit $example_dir/openmit.conf \
  train_path = "$path_to_train" \
  valid_path = "$path_to_valid" \
  model_dump = "$path_to_model_dump" \
  model_binary = "$path_to_model_binary"

if [ $# <> 0 ]; then
  echo "[ERROR] openmit yarn run failed!"
  exit 1
else
  echo "[INFO] openmit yran run success!"
fi

echo "================ $0 done ==================="
