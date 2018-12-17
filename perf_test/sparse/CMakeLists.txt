
ADD_PERF_EXECUTABLE(sparse_pcg    SOURCES KokkosSparse_pcg.cpp)
ADD_PERF_EXECUTABLE(sparse_spgemm SOURCES KokkosSparse_spgemm.cpp)

ADD_PERF_EXECUTABLE(sparse_spmv    
  SOURCES KokkosSparse_spmv.cpp
  INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/spmv
)

