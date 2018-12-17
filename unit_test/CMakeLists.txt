
KOKKOS_SET_TRIBITS_INCLUDES(
 UNIT_TEST_INCLUDES
 ${CMAKE_CURRENT_BINARY_DIR}
 ${CMAKE_CURRENT_BINARY_DIR}/blas
 ${CMAKE_CURRENT_BINARY_DIR}/graph
 ${CMAKE_CURRENT_BINARY_DIR}/sparse
)

KOKKOS_APPEND_TRIBITS_INSTALL_INCLUDES(
 UNIT_TEST_INCLUDES
 ${CMAKE_CURRENT_SOURCE_DIR}
 ${CMAKE_CURRENT_SOURCE_DIR}/blas
 ${CMAKE_CURRENT_SOURCE_DIR}/graph
 ${CMAKE_CURRENT_SOURCE_DIR}/sparse
)

#${CMAKE_CURRENT_BINARY_DIR}/batched
#${CMAKE_CURRENT_SOURCE_DIR}/batched

# Kyungjoo: when kokkoskernels and kokkos are symbolic link and their src directories are overrided
#           the following relative path does not work or users should put kokkoskernels and kokkos
#           at the same place
#SET(GTEST_SOURCE_DIR ${${PARENT_PACKAGE_NAME}_SOURCE_DIR}/../kokkos/tpls/gtest)
# Disables pthreads, this is a problem for serial builds in Trilinos & Sierra if it's enabled.
# Just point to a common kokkos_gtest to avoid this

FUNCTION(KokkosKernels_ADD_TEST name folder glob)
  # There are no binary include files - I don't know why this is used
  # removing does not seem to break it
  #INCLUDE_DIRECTORIES(blas_cuda ${CMAKE_CURRENT_BINARY_DIR}/cuda)

  APPEND_GLOB(TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${folder}/${glob})

  KOKKOS_ADD_TEST(
    ${name}_${folder}
    SOURCES ${TEST_SOURCES}
    INCLUDES 
      ${CMAKE_CURRENT_SOURCE_DIR}/${name}
      ${CMAKE_CURRENT_SOURCE_DIR}/${folder}
    ARGS ${ARGN}
  )
ENDFUNCTION()


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


 
