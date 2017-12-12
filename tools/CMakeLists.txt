#set(exec "test_sarray")
#set(test_src "test_sarray.cc")

#set(exec "test_glog")
#set(test_src "test_glog.cc")

#set(exec "unittest_openmit_data")
#set(test_src "unittest_openmit_data.cc")

#set(exec "data_test")
#set(exec "entry_meta_test")
#set(exec "entry_test")
set(exec "sse_test")

set(test_src "${exec}.cc")

add_executable(${exec} ${test_src})
target_link_libraries(${exec} dmlc hdfs hadoop mit jvm)
