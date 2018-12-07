
Message("${PACKAGE_NAME}")
TRIBITS_CONFIGURE_FILE(${PACKAGE_NAME}_config.h)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

IF(NOT KOKKOSKERNELS_HAS_TRILINOS)
SET(TRILINOS_INCDIR "include")
ELSE()
SET(TRILINOS_INCDIR ${CMAKE_INSTALL_PREFIX}/${${PROJECT_NAME}_INSTALL_INCLUDE_DIR})
ENDIF()

#-----------------------------------------------------------------------------


SET(INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/impl
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/tpls
  ${CMAKE_CURRENT_SOURCE_DIR}/blas
  ${CMAKE_CURRENT_SOURCE_DIR}/blas/impl
  ${CMAKE_CURRENT_SOURCE_DIR}/sparse
  ${CMAKE_CURRENT_SOURCE_DIR}/sparse/impl
  ${CMAKE_CURRENT_SOURCE_DIR}/graph
  ${CMAKE_CURRENT_SOURCE_DIR}/graph/impl
  ${CMAKE_CURRENT_SOURCE_DIR}/batched
  ${CMAKE_CURRENT_SOURCE_DIR}/batched/impl
  ${CMAKE_CURRENT_SOURCE_DIR}/common
  ${CMAKE_CURRENT_SOURCE_DIR}/impl
)

IF (KokkosKernels_INST_EXECSPACE_SERIAL)
  IF (KokkosKernels_INST_DOUBLE)
    APPEND_GLOB(SOURCES ${DIR}/generated_specializations_cpp/*/*inst_double_*_Serial_*.cpp)
  ENDIF()
  IF (KokkosKernels_INST_FLOAT)
    APPEND_GLOB(SOURCES ${DIR}/generated_specializations_cpp/*/*inst_float_*_Serial_*.cpp)
  ENDIF()
  IF (KokkosKernels_INST_COMPLEX_DOUBLE)
    APPEND_GLOB(SOURCES ${DIR}/generated_specializations_cpp/*/*_complex_double_*_Serial_*.cpp)
  ENDIF()
  IF (KokkosKernels_INST_COMPLEX_FLOAT)
    APPEND_GLOB(SOURCES ${DIR}/generated_specializations_cpp/*/*_complex_float_*_Serial_*.cpp)
  ENDIF()
ENDIF()

IF (KokkosKernels_INST_EXECSPACE_THREADS)
  IF (KokkosKernels_INST_DOUBLE)
    APPEND_GLOB(SOURCES ${DIR}/generated_specializations_cpp/*/*inst_double_*_Threads_*.cpp)
  ENDIF()
  IF (KokkosKernels_INST_FLOAT)
    APPEND_GLOB(SOURCES ${DIR}/generated_specializations_cpp/*/*inst_float_*_Threads_*.cpp)
  ENDIF()
  IF (KokkosKernels_INST_COMPLEX_DOUBLE)
    APPEND_GLOB(SOURCES ${DIR}/generated_specializations_cpp/*/*_complex_double_*_Threads_*.cpp)
  ENDIF()
  IF (KokkosKernels_INST_COMPLEX_FLOAT)
    APPEND_GLOB(SOURCES ${DIR}/generated_specializations_cpp/*/*_complex_float_*_Threads_*.cpp)
  ENDIF()
ENDIF()

IF (KokkosKernels_INST_EXECSPACE_OPENMP)
  IF (KokkosKernels_INST_DOUBLE)
    APPEND_GLOB(SOURCES ${DIR}/generated_specializations_cpp/*/*inst_double_*_OpenMP_*.cpp)
  ENDIF()
  IF (KokkosKernels_INST_FLOAT)
    APPEND_GLOB(SOURCES ${DIR}/generated_specializations_cpp/*/*inst_float_*_OpenMP_*.cpp)
  ENDIF()
  IF (KokkosKernels_INST_COMPLEX_DOUBLE)
    APPEND_GLOB(SOURCES ${DIR}/generated_specializations_cpp/*/*_complex_double_*_OpenMP_*.cpp)
  ENDIF()
  IF (KokkosKernels_INST_COMPLEX_FLOAT)
    APPEND_GLOB(SOURCES ${DIR}/generated_specializations_cpp/*/*_complex_float_*_OpenMP_*.cpp)
  ENDIF()
ENDIF()

IF (KokkosKernels_INST_EXECSPACE_CUDA)
  APPEND_GLOB(SOURCES ${DIR}/tpls/*_Cuda_*.cpp)
  IF (KokkosKernels_INST_DOUBLE)
    APPEND_GLOB(SOURCES ${DIR}/generated_specializations_cpp/*/*inst_double_*_Cuda_*.cpp)
  ENDIF()
  IF (KokkosKernels_INST_FLOAT)
    APPEND_GLOB(SOURCES ${DIR}/generated_specializations_cpp/*/*inst_float_*_Cuda_*.cpp)
  ENDIF()
  IF (KokkosKernels_INST_COMPLEX_DOUBLE)
    APPEND_GLOB(SOURCES ${DIR}/generated_specializations_cpp/*/*_complex_double_*_Cuda_*.cpp)
  ENDIF()
  IF (KokkosKernels_INST_COMPLEX_FLOAT)
    APPEND_GLOB(SOURCES ${DIR}/generated_specializations_cpp/*/*_complex_float_*_Cuda_*.cpp)
  ENDIF()
ENDIF()

LIST(APPEND HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.h)
#LIST(APPEND HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_ETIHelperMacros.h)

#-----------------------------------------------------------------------------

foreach(dir ${INCLUDE_DIRS})
APPEND_GLOB(HEADERS ${dir}/*.hpp)
endforeach()

SET(INCLUDE_TYPE PUBLIC)

if(NOT KOKKOSKERNELS_HAS_TRILINOS)
# Follow a different path here if not Trilinos
# primarily related to creating an interface library
if (SOURCES) #If we have sources file, this is a regular library
add_library(kokkoskernels ${SOURCES})
SET(INCLUDE_TYPE PUBLIC) 
target_link_libraries(kokkoskernels kokkos::kokkos)
else() #this is a header-only interface library
add_library(kokkoskernels INTERFACE)
SET(INCLUDE_TYPE INTERFACE)
target_link_libraries(kokkoskernels INTERFACE kokkos::kokkos)
endif()
else()
TRIBITS_ADD_LIBRARY(
  kokkoskernels
  HEADERS ${HEADERS}
  SOURCES ${SOURCES}
)
endif()

foreach(dir ${INCLUDE_DIRS})
target_include_directories(kokkoskernels ${INCLUDE_TYPE}
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${dir}>
)
endforeach()

target_include_directories(kokkoskernels ${INCLUDE_TYPE}
  $<INSTALL_INTERFACE:${TRILINOS_INCDIR}/generated_specializations_hpp>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/impl/generated_specializations_hpp>
)
APPEND_GLOB(HEADERS_GENERATED  ${CMAKE_CURRENT_SOURCE_DIR}/impl/generated_specializations_hpp/*.hpp)

INSTALL(FILES ${HEADERS_GENERATED}
        DESTINATION ${TRILINOS_INCDIR}/generated_specializations_hpp/)


if (NOT KOKKOSKERNELS_HAS_TRILINOS)
#export and install a package if not part of trilinos
INSTALL (FILES ${HEADERS}
         DESTINATION ${TRILINOS_INCDIR}
)

install(TARGETS kokkoskernels EXPORT kokkoskernels
        INCLUDES DESTINATION include
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

install(EXPORT kokkoskernels
        FILE KokkosKernelsTargets.cmake
        NAMESPACE kokkos::
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake)
endif()


