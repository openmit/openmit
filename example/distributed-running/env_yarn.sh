#!/bin/bash -x 

cd $(dirname `ls -l $0 | awk '{print $NF;}'`)
project_dir=`pwd`/../.. 

source $project_dir/make/openmit.mk 

export JAVA_HOME
export HADOOP_HOME 
export HADOOP_HDFS_HOME=$HADOOP_HOME 
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop 

export LD_LIBRARY_PATH=$GCC_LIB_PATH:$HADOOP_HOME/lib/native:$JAVA_HOME/jre/lib/amd64/server:$LD_LIBRARY_PATH
if [ -f "$HADOOP_HOME/bin/hadoop_user_login.sh" ]; then
  source $HADOOP_HOME/bin/hadoop_user_login.sh $HADOOP_USER 
fi
if [ -f "$HADOOP_HOME/etc/hadoop/krb5.conf" ]; then
  export LIBHDFS_OPTS="--Xmx128m -Djava.security.krb5.conf=$HADOOP_HOME/etc/hadoop/krb5.conf"
fi
export DMLC_JOB_CLUSTER=yarn
export PATH=$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH 

echo "`which hdfs`" && echo "`which hadoop`"

echo "============= $0 done ============"
