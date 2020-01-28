KOKKOSKERNELS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
KOKKOSKERNELS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

KOKKOSKERNELS_ADD_ENABLED_EXECUTABLE(
  sparse_pcg
  SOURCES KokkosSparse_pcg.cpp
)

#KOKKOSKERNELS_ADD_ENABLED_EXECUTABLE(
#  sparse_block_pcg
#  SOURCES KokkosSparse_block_pcg.cpp
#)

KOKKOSKERNELS_ADD_ENABLED_EXECUTABLE(
  sparse_spgemm
  SPGEMM KokkosSparse_spgemm.cpp
)
  
KOKKOSKERNELS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/spmv)

KOKKOSKERNELS_ADD_ENABLED_EXECUTABLE(
  sparse_spadd
  SPADD KokkosSparse_spadd.cpp
)

KOKKOSKERNELS_ADD_ENABLED_EXECUTABLE(
  sparse_spmv_struct
  SPMV_STRUCT KokkosSparse_spmv_struct.cpp
)

KOKKOSKERNELS_ADD_ENABLED_EXECUTABLE(
  sparse_spmv_struct_tuning
  SPMV_STRUCT KokkosSparse_spmv_struct_tuning.cpp
  )

KOKKOSKERNELS_ADD_ENABLED_EXECUTABLE(
  sparse_spmv
  SPMV KokkosSparse_spmv.cpp
)

KOKKOSKERNELS_ADD_ENABLED_EXECUTABLE(
  sparse_sptrsv
  SPTRSV KokkosSparse_sptrsv.cpp 
  )

KOKKOSKERNELS_ADD_ENABLED_EXECUTABLE(
  sparse_sptrsv_cholmod
  SPTRSV KokkosSparse_sptrsv_cholmod.cpp 
  )

KOKKOSKERNELS_ADD_ENABLED_EXECUTABLE(
  sparse_sptrsv_superlu
  SPTRSV KokkosSparse_sptrsv_superlu.cpp
  )

#KOKKOSKERNELS_ADD_ENABLED_EXECUTABLE(
#  sparse_gs
#  SOURCES KokkosSparse_gs.cpp
#)

KOKKOSKERNELS_ADD_ENABLED_EXECUTABLE(
  sparse_spiluk
  SPILUK KokkosSparse_spiluk.cpp
)
