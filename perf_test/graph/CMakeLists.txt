
ADD_PERF_EXECUTABLE(graph_color SOURCES kokkosGraph_color.cpp)
ADD_PERF_EXECUTABLE(graph_color_d2 SOURCES kokkosGraph_color_d2.cpp)

#Below will probably fail on GPUs.
#TRIBITS_ADD_EXECUTABLE(
#  graph_triangle
#  SOURCES KokkosGraph_triangle.cpp      
#  )

