#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

ADD_PERF_EXECUTABLE(graph_color kokkosGraph_color.cpp)
ADD_PERF_EXECUTABLE(graph_color_d2 kokkosGraph_color_d2.cpp)

#Below will probably fail on GPUs.
#TRIBITS_ADD_EXECUTABLE(
#  graph_triangle
#  SOURCES KokkosGraph_triangle.cpp      
#  )

