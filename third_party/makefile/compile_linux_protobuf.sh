#!/bin/bash -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR=$SCRIPT_DIR/../.. 

source $PROJECT_DIR/conf/base.conf
source $PROJECT_DIR/conf/deps.conf 

echo "platform: \"${platform}\""
echo "proto_version \"${proto_version}\""

#DOWNLOADS_DIR=$SCRIPT_DIR/downloads
#GEN_DIR=$SCRIPT_DIR/gen

if [[ "${platform}" == "" ]]; then
  echo "You can not config 'platform' in $PROJECT_DIR/conf/deps.conf. use default platform: 'linux'"
  platform="linux"
fi

if [[ "${proto_version}" == "" ]]; then
  echo "You can not config 'proto_version' in $PROJECT_DIR/conf/deps.conf. use default proto version: 3.5.0"
  proto_version=3.5.0 
fi 

downloads_protobuf_targz=$DOWNLOADS_DIR/v${proto_version}.tar.gz
downloads_protobuf_dir=$DOWNLOADS_DIR/protobuf-${proto_version}
gen_protobuf_dir=$GEN_DIR/protobuf/${platform}/$proto_version

PROTOBUF_URL="https://github.com/google/protobuf/archive/v${proto_version}.tar.gz"

#if [ ! -f $downloads_protobuf_targz ]; then
#  mkdir -p $DOWNLOADS_DIR || true
#  wget -P $DOWNLOADS_DIR $PROTOBUF_URL 
#fi
#rm -rf $downloads_protobuf_dir || true 
#tar -xvf ${downloads_protobuf_targz} -C ${DOWNLOADS_DIR}
#
#if [[ ! -f "${DOWNLOADS_DIR}/protobuf-${proto_version}/autogen.sh" ]]; then
#  echo "You need to downloads dependencies before running this script: $(basename $0)" 1>&2
#  exit 1
#fi
#
#cd ${DOWNLOADS_DIR}/protobuf-${proto_version} 
#./autogen.sh
#if [ $? -ne 0 ]; then
#  echo "./autogen.sh command failed."
#  exit 1
#fi
#
#mkdir -p $gen_protobuf_dir || true
#./configure --prefix="${gen_protobuf_dir}" --with-pic
#if [ $? -ne 0 ]; then
#  echo "./configure command failed."
#  exit 1
#fi 
#
#make clean && make -j4 && make check
#if [ $? -ne 0 ]; then 
#  echo "make command failed."
#  exit 1
#fi 
#
#make install 
#if [ $? -ne 0 ]; then
#  echo "make install command failed."
#  exit 1
#fi

echo "$(basename $0) finished successfully!!!"
