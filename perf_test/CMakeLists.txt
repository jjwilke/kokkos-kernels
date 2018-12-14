#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../test_common)

# At some point, we may wish to make this into real "performance
# tests," in the sense that they can pass or fail.  At that point, use
# "CATEGORIES PERFORMANCE" to mark them as such.  For now, we just
# build the executables for manual use, but don't run the tests.  They
# build correctly with or without MPI, but only run them with a single
# MPI process.

FUNCTION(ADD_PERF_EXECUTABLE NAME SOURCELIST)
TRIBITS_ADD_EXECUTABLE(
  ${NAME}
  SOURCES ${SOURCELIST}
  NOEXEPREFIX 
)
TARGET_INCLUDE_DIRECTORIES(
 ${NAME} PRIVATE
 ${CMAKE_CURRENT_BINARY_DIR}
 ${CMAKE_CURRENT_SOURCE_DIR}
)
TARGET_LINK_LIBRARIES(
 ${NAME}
 PRIVATE
 kokkoskernels_test_common
)
TARGET_LINK_LIBRARIES(
 ${NAME}
 PRIVATE
 kokkoskernels
)
ENDFUNCTION()

ADD_SUBDIRECTORY(graph)
ADD_SUBDIRECTORY(sparse)
ADD_SUBDIRECTORY(performance)
#ADD_SUBDIRECTORY(common)

