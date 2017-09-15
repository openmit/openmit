#!/bin/bash -x

git submodule init
git submodule update

PROJECT_PATH=`pwd`/`dirname $0`
THIRD_PARTY_PATH=$PROJECT_PATH/third_party

mkdir -p $THIRD_PARTY_PATH/{include,lib} || true

#cd $PROJECT_PATH/third_party/liblbfgs
#./autogen.sh && ./configure --prefix=$THIRD_PARTY_PATH --disable-shared --enable-static --enable-sse2
#make && make install

cd $PROJECT_PATH/third_party/ps-lite
make -j4 \
  && cp -r include/* $THIRD_PARTY_PATH/include \
  && cp -r build/libps.a $THIRD_PARTY_PATH/lib

cd $PROJECT_PATH/third_party/rabit
make all \
  && cp -r include/* $THIRD_PARTY_PATH/include \
  && cp -r lib/librabit.a $THIRD_PARTY_PATH/lib

cd $PROJECT_PATH/third_party/dmlc-core
make all DMLC_ENABLE_STD_THREAD=1 USE_HDFS=1 DMLC_USE_REGEX=1 \
         #DMLC_USE_GLOG=1 \
         HDFS_INC_PATH=$THIRD_PARTY_PATH/hadoop/include HDFS_LIB_PATH=$THIRD_PARTY_PATH/hadoop/lib \
  && cp -r include/dmlc $THIRD_PARTY_PATH/include \
  && cp libdmlc.a $THIRD_PARTY_PATH/lib

cd $PROJECT_PATH/third_party/googletest
cmake . && make \
  && cp -r googletest/include/gtest $THIRD_PARTY_PATH/include \
  && cp googlemock/gtest/libgtest* $THIRD_PARTY_PATH/lib 

#cd $PROJECT_PATH/third_party/glog
#automake --add-missing && ./configure \
#    && sed -i 's/aclocal-1.14/aclocal/g;s/automake-1.14/automake/g' Makefile \
#    && make \
#    && cp -r src/glog $THIRD_PARTY_PATH/include \
#    && cp .libs/libglog.a $THIRD_PARTY_PATH/lib
