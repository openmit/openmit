aux_source_directory(helloworld SRCS_HELLOWORLD)

set(mit_grpc_SRCS ${SRCS_HELLOWORLD})

set(static_lib mit_grpc)
add_library(${static_lib} STATIC ${mit_grpc_SRCS})
set_target_properties(${static_lib} PROPERTIES OUTPUT "mit")
set_target_properties(${static_lib} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
target_link_libraries(${static_lib} grpc++ protobuf)

set(exec "greeter_client")
add_executable(${exec} "${exec}.cc")
target_link_libraries(${exec} ${static_lib})
set(exec "greeter_server")
add_executable(${exec} "${exec}.cc")
target_link_libraries(${exec} ${static_lib})

