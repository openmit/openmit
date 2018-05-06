#!/bin/bash -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR=$SCRIPT_DIR/../.. 

source $PROJECT_DIR/conf/base.conf
source $PROJECT_DIR/conf/deps.conf 

echo "platform: \"${platform}\""
echo "proto_version \"${proto_version}\""
