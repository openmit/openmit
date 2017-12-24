cmake_minimum_required(VERSION 2.8)

aux_source_directory(common SRCS_COMMON)
aux_source_directory(executor SRCS_EXECUTOR)
aux_source_directory(entry SRCS_ENTRY)
aux_source_directory(framework/mpi SRCS_FRAMEWORK_MPI)
aux_source_directory(framework/ps SRCS_FRAMEWORK_PS)
aux_source_directory(loss SRCS_LOSS)
aux_source_directory(metric SRCS_METRIC)
aux_source_directory(model SRCS_MODEL)
aux_source_directory(optimizer SRCS_OPTIMIZER)
aux_source_directory(tools/dstruct SRCS_TOOLS_DSTRUCT)
aux_source_directory(tools/dstore SRCS_TOOLS_DSTORE)
aux_source_directory(tools/hash SRCS_TOOLS_HASH)
aux_source_directory(tools/io SRCS_TOOLS_IO)
aux_source_directory(tools/math SRCS_TOOLS_MATH)
aux_source_directory(tools/monitor SRCS_TOOLS_MONITOR)


set(mit_SRCS
  ${SRCS_COMMON}
  ${SRCS_EXECUTOR}
  ${SRCS_ENTRY}
  ${SRCS_FRAMEWORK_MPI}
  ${SRCS_FRAMEWORK_PS}
  ${SRCS_LOSS}
  ${SRCS_METRIC}
  ${SRCS_MODEL}
  ${SRCS_OPTIMIZER}
  ${SRCS_TOOLS_DSTRUCT}
  ${SRCS_TOOLS_DSTORE}
  ${SRCS_TOOLS_HASH}
  ${SRCS_TOOLS_IO}
  ${SRCS_TOOLS_MATH}
  ${SRCS_TOOLS_MONITOR})

# static libraries
set (static_lib mit)
add_library(${static_lib} STATIC ${mit_SRCS})
set_target_properties(${static_lib} PROPERTIES OUTPUT "mit")
set_target_properties(${static_lib} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
target_link_libraries(${static_lib} dmlc ps rabit)

set(exec_name "openmit")
add_executable(${exec_name} "learner.h" "learner.cc" "cli_main.cc")

set(openmit_LINKER_LIBS "")
list(APPEND openmit_LINKER_LIBS ${static_lib})
list(APPEND openmit_LINKER_LIBS pthread)
list(APPEND openmit_LINKER_LIBS dmlc)
list(APPEND openmit_LINKER_LIBS rabit)
list(APPEND openmit_LINKER_LIBS ps)
list(APPEND openmit_LINKER_LIBS protobuf-lite)
list(APPEND openmit_LINKER_LIBS zmq)
list(APPEND openmit_LINKER_LIBS hdfs)
list(APPEND openmit_LINKER_LIBS hadoop)
list(APPEND openmit_LINKER_LIBS jvm)
list(APPEND openmit_LINKER_LIBS rt)
list(APPEND openmit_LINKER_LIBS dl)
list(APPEND openmit_LINKER_LIBS mit)

target_link_libraries(${exec_name} ${openmit_LINKER_LIBS})
