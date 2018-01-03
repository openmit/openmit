#!/bin/bash

set -o pipefail
set -o errexit

for i in {1..1000}
do
  echo "=============== $i ==============="
  ./example/mit_ps.sh
done
