

SET(TEST_INCLUDE_BINARY_DIRS
 ${CMAKE_CURRENT_BINARY_DIR}
 ${CMAKE_CURRENT_BINARY_DIR}/blas
 ${CMAKE_CURRENT_BINARY_DIR}/graph
 ${CMAKE_CURRENT_BINARY_DIR}/sparse
)
#${CMAKE_CURRENT_BINARY_DIR}/batched

SET(TEST_INCLUDE_SOURCE_DIRS
 ${CMAKE_CURRENT_SOURCE_DIR}
 ${CMAKE_CURRENT_SOURCE_DIR}/blas
 ${CMAKE_CURRENT_SOURCE_DIR}/graph
 ${CMAKE_CURRENT_SOURCE_DIR}/sparse
)
#${CMAKE_CURRENT_SOURCE_DIR}/batched

# Kyungjoo: when kokkoskernels and kokkos are symbolic link and their src directories are overrided
#           the following relative path does not work or users should put kokkoskernels and kokkos
#           at the same place
#SET(GTEST_SOURCE_DIR ${${PARENT_PACKAGE_NAME}_SOURCE_DIR}/../kokkos/tpls/gtest)
# Disables pthreads, this is a problem for serial builds in Trilinos & Sierra if it's enabled.

#TRIBITS_ADD_LIBRARY(
#  kokkoskernels_gtest
#  HEADERS ${GTEST_SOURCE_DIR}/gtest/gtest.h
#  SOURCES ${GTEST_SOURCE_DIR}/gtest/gtest-all.cc
#  TESTONLY
#)
#This comes in from kokkos
#TARGET_COMPILE_OPTIONS(kokkoskernels_gest "-DGTEST_HAS_PTHREAD=0")
#TARGET_INCLUDE_DIRECTORIES(kokkoskernels_gtest PUBLIC ${GTEST_SOURCE_DIR})

function(KokkosKernels_ADD_TEST name folder glob)
  # There are no binary include files - I don't know why this is used
  # removing does not seem to break it
  #INCLUDE_DIRECTORIES(blas_cuda ${CMAKE_CURRENT_BINARY_DIR}/cuda)

  APPEND_GLOB(TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${folder}/${glob})

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    ${name}_${folder}
    NOEXEPREFIX 
    SOURCES
      Test_Main.cpp
      ${TEST_SOURCES}
    NUM_MPI_PROCS 1
    TESTONLYLIBS ${Kokkos_LIBTESTNAME} kokkoskernels
    ARGS ${ARGN}
  )
  TARGET_LINK_LIBRARIES(${name}_${folder} PRIVATE kokkoskernels_test_common)
  TARGET_INCLUDE_DIRECTORIES(${name}_${folder} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/${folder})
  TARGET_INCLUDE_DIRECTORIES(${name}_${folder} PRIVATE ${TEST_INCLUDE_SOURCE_DIRS})
endfunction()


IF (Kokkos_ENABLE_Cuda)
  KokkosKernels_ADD_TEST(blas   cuda Test_Cuda_Blas*.cpp)
  KokkosKernels_ADD_TEST(sparse cuda Test_Cuda_Sparse*.cpp)
  KokkosKernels_ADD_TEST(graph  cuda Test_Cuda_Graph*.cpp)
  KokkosKernels_ADD_TEST(common cuda Test_Cuda_Common_ArithTraits.cpp)
ENDIF ()

IF (Kokkos_ENABLE_OpenMP)
  IF (KOKKOS_ENABLE_DEBUG)
    SET(DISABLE_SLOW_DGEMM_DOUBLE_TEST "--gtest_filter=-openmp.gemm_double")
  ENDIF()

  KokkosKernels_ADD_TEST(blas   openmp Test_OpenMP_Blas*.cpp ${DISABLE_SLOW_DGEMM_DOUBLE_TEST})
  KokkosKernels_ADD_TEST(sparse openmp Test_OpenMP_Sparse*.cpp)
  KokkosKernels_ADD_TEST(graph  openmp Test_OpenMP_Graph*.cpp)
  KokkosKernels_ADD_TEST(common openmp Test_OpenMP_Common_ArithTraits.cpp)
ENDIF ()

IF (Kokkos_ENABLE_Serial)
  IF (KOKKOS_ENABLE_DEBUG)
    SET(DISABLE_SLOW_DGEMM_DOUBLE_TEST "--gtest_filter=-serial.gemm_double")
  ENDIF()
  KokkosKernels_ADD_TEST(blas   serial Test_Serial_Blas*.cpp ${DISABLE_SLOW_DGEMM_DOUBLE_TEST})
  KokkosKernels_ADD_TEST(sparse serial Test_Serial_Sparse*.cpp)
  KokkosKernels_ADD_TEST(graph  serial Test_Serial_Graph*.cpp)
  KokkosKernels_ADD_TEST(common serial Test_Serial_Common_ArithTraits.cpp)
ENDIF ()

IF (Kokkos_ENABLE_Pthread)
  KokkosKernels_ADD_TEST(blas   threads Test_Threads_Blas*.cpp)
  KokkosKernels_ADD_TEST(sparse threads Test_Threads_Sparse*.cpp)
  KokkosKernels_ADD_TEST(graph  threads Test_Threads_Graph*.cpp)
  KokkosKernels_ADD_TEST(common threads Test_Threads_Common_ArithTraits.cpp)
ENDIF ()


 
