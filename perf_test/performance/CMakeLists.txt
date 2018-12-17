# Add the two performance tests for the Performance class
# located in Kokkos_Performance.hpp.
# performance_validate validates the behavior of the Performance class.
# performance_example is a simple example of using it.
ASSERT_DEFINED(TPL_ENABLE_yaml-cpp)
IF(TPL_ENABLE_yaml-cpp)
  KOKKOS_ADD_TEST(
   performance_validate
   SOURCES performance_validate.cpp)

  KOKKOS_ADD_TEST(
   performance_example
   SOURCES performance_example.cpp)
ENDIF()

