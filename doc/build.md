## 使用指南

+ [项目构建](#1.项目构建)
    + [Ubuntu/Centos构建](#1.1.Ubuntu/Centos构建)
    + [MacOS构建](#1.2.MacOS构建) 
    + [支持GPU构建](#1.3.支持GPU构建)
+ [使用示例](#2.使用示例)

<h2 id="1.项目构建">项目构建</h2>

OpenMIT项目构建比较简单，构建过程如下:

<h3 id="1.1.Ubuntu/Centos构建">Ubuntu/Centos构建</h3>

+ Step1，下载从github上OpenMIT项目

```bash
git clone --recursive https://github.com/openmit/openmit
```
OpenMIT如果在分布式环境下运行，需要hadoop和jvm环境。如果单机运行，跳过Step2.

+ Step2，检查`HADOOP_HOME`和`JAVA_HOME`环境是否存在

```bash
which hadoop
echo $HADOOP_HOME       // 如果输出都为🈳️，说明没有安装hadoop
echo $JAVA_HOME         // 如果输出为🈳️，说明没有配置，需要安装jvm
```
安装好Hadoop和Java后，在`~/.bashrc`文件中添加：

```bash
export HADOOP_HOME=${path to your hadoop path}
export HADOOP_HDFS_HOME=$HADOOP_HOME
export HDFS_INC_PATH=${HADOOP_HOME}/include
export HDFS_LIB_PATH=${HADOOP_HOME}/lib/native
export JAVA_HOME=${path to your java path}
```
+ Step3，构建依赖项目

```bash
sh build_deps.sh
```
+ Step4，构建OpenMIT项目

```bash
sh build.sh
```
如果构建成功，在lib和bin目录下分别有`libmit.a`和可执行文件`openmit`. 

<h3 id="1.2.MacOS构建">MacOS构建</h3>

> todo

<h3 id="1.3.支持GPU构建">支持GPU构建</h3>

> todo


<h2 id="1.使用示例">使用示例</h2>

[项目构建](#1.项目构建)后，验证程序是否可执行。

+ 单机版MPI任务(单机多核模拟多计算节点)

```bash
 ./examples/mit_mpi.sh 2        // ‘2’表示指定2个worker节点
```
+ 单机版PS任务

```bash
./example/mit_ps.sh 
```




