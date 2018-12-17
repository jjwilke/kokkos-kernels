#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../test_common)

# At some point, we may wish to make this into real "performance
# tests," in the sense that they can pass or fail.  At that point, use
# "CATEGORIES PERFORMANCE" to mark them as such.  For now, we just
# build the executables for manual use, but don't run the tests.  They
# build correctly with or without MPI, but only run them with a single
# MPI process.

#Forward through arguments to the add test function
FUNCTION(ADD_PERF_EXECUTABLE NAME)
KOKKOS_ADD_TEST(
  ${NAME}
  TESTONLYLIBS kokkoskernels_test_common
  BUILDONLY
  ${ARGN} #just forward all args
)
ENDFUNCTION()

ADD_SUBDIRECTORY(graph)
ADD_SUBDIRECTORY(sparse)
ADD_SUBDIRECTORY(performance)
#ADD_SUBDIRECTORY(common)

