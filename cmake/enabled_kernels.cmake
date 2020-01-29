SET(KOKKOSKERNELS_AVAILABLE_LIST
  ABS
  AXPBY
  DOT
  GEMV
  GEMM
  GESV
  IAMAX
  MULT
  NRM1
  NRM2
  NRM2W
  NRMINF
  RECIPROCAL
  SCAL
  SUM
  TRSV
  TRSM
  UPDATE
#  MULT_MV
#  DOT_MV
#  ABS_MV
#  AXPBY_MV
#  IAMAX_MV
#  NRM1_MV
#  NRM2_MV
#  NRM2W_MV
#  NRMINF_MV
#  RECIPROCAL_MV
#  SCAL_MV
#  SUM_MV
#  UPDATE_MV
  GAUSS_SEIDEL
  SPADD
  SPGEMM
  SPILUK
  SPMV
  SPTRSV
  BATCHED_ADDRADIAL
  BATCHED_APPLYHOUSEHOLDER
  BATCHED_COPY
  BATCHED_EIGENDECOMPOSITION
  BATCHED_GEMM
  BATCHED_GEMV
  BATCHED_HOUSEHOLDER
  BATCHED_INNERGEMMFIXA
  BATCHED_INNERGEMMFIXB
  BATCHED_INNERGEMMFIXC
  BATCHED_INNERLU
  BATCHED_INNERMULTIPLEDOTPRODUCT
  BATCHED_INNERTRSM
  BATCHED_INVERSELU
  BATCHED_LU
  BATCHED_SCALE
  BATCHED_SET
  BATCHED_SETIDENTITY
  BATCHED_SOLVELU
  BATCHED_TRSM
  BATCHED_TRSV
  GRAPH_DISTANCE1COLOR
  GRAPH_DISTANCE2COLOR
  GRAPH_COLOR
  GRAPH_TRIANGLE
  COMMON #this is a special one that enables certain utility tests
)

MACRO(KERNEL_DEPENDS kernel)
 STRING(TOUPPER ${kernel} KERNEL_UC)
 IF (NOT ${KERNEL_UC} IN_LIST KOKKOSKERNELS_AVAILABLE_LIST)
  MESSAGE(FATAL_ERROR "Cannot add dependencies for unknown kernel ${kernel}")
 ENDIF()
 SET(${KERNEL_UC}_DEPENDENCIES)
 FOREACH(entry ${ARGN})
  STRING(TOUPPER ${entry} ENTRY_UC)
   IF (NOT ${KERNEL_UC} IN_LIST KOKKOSKERNELS_AVAILABLE_LIST)
    MESSAGE(FATAL_ERROR "Cannot add unknown kernel ${entry} to dependency list for ${kernel}")
   ENDIF()
   LIST(APPEND ${KERNEL_UC}_DEPENDENCIES ${ENTRY_UC})
 ENDFOREACH()
ENDMACRO()

KERNEL_DEPENDS(spmv scal)
KERNEL_DEPENDS(scal dot)

FUNCTION(SET_KERNEL_ENABLE_VALUES VALUE)
  CMAKE_PARSE_ARGUMENTS(ENABLE
    ""
    ""
    "NAMES"
    ${ARGN}
  )

  STRING(TOUPPER "${ENABLE_NAMES}" ENABLE_NAMES_UC)
  IF    (${ENABLE_NAMES_UC} STREQUAL "ALL")
    FOREACH(NAME ${KOKKOSKERNELS_AVAILABLE_LIST})
      SET(KOKKOSKERNELS_ENABLE_${NAME} ${VALUE} PARENT_SCOPE)
      SET(KOKKOSKERNELS_ENABLE_${NAME} ${VALUE})
      MESSAGE(STATUS "Setting ALL, including ${NAME}, to ${VALUE}")
    ENDFOREACH()
  ELSEIF(${ENABLE_NAMES_UC} STREQUAL "NONE")
    #do nothing
  ELSE()
    FOREACH(name ${ENABLE_NAMES})
      STRING(TOUPPER ${name} NAME_UC)
      IF (${NAME_UC} IN_LIST KOKKOSKERNELS_AVAILABLE_LIST)
        SET(KOKKOSKERNELS_ENABLE_${NAME_UC} ${VALUE} PARENT_SCOPE)
        SET(KOKKOSKERNELS_ENABLE_${NAME_UC} ${VALUE})
        MESSAGE(STATUS "Setting ${NAME_UC} to ${VALUE}")
      ELSE()
        MESSAGE(FATAL_ERROR "Got bad kernel name ${name} in enabled list")
      ENDIF()
    ENDFOREACH()
  ENDIF()
ENDFUNCTION()

FUNCTION(ENABLE_KERNEL_DEPENDENCIES)
  SET(NEW_ENABLE ON)
  WHILE(NEW_ENABLE)
    SET(NEW_ENABLE OFF)
    #now loop through all the enabled variables and 
    #make sure their dependencies are enabled
    FOREACH(NAME ${KOKKOSKERNELS_AVAILABLE_LIST})
      IF (KOKKOSKERNELS_ENABLE_${NAME})
        FOREACH(DEP ${${NAME}_DEPENDENCIES})
          IF (NOT KOKKOSKERNELS_ENABLE_${DEP})
            MESSAGE(STATUS "Forcing enable kernel ${DEP} for ${NAME}")
            SET(KOKKOSKERNELS_ENABLE_${DEP} ON PARENT_SCOPE)
            SET(KOKKOSKERNELS_ENABLE_${DEP} ON)
            SET(NEW_ENABLE ON)
          ENDIF()
        ENDFOREACH()
      ENDIF()
    ENDFOREACH()
  ENDWHILE()
ENDFUNCTION()

FUNCTION(set_category_enable_values category)
 CMAKE_PARSE_ARGUMENTS(ENABLE
  ""
  ""
  "NAMES"
 ${ARGN})
 STRING(TOUPPER ${category} CATEGORY_UC)
 SET(KOKKOSKERNELS_ENABLE_${CATEGORY_UC} OFF PARENT_SCOPE)
 FOREACH(name ${NAMES})
  STRING(TOUPPER ${name} NAME_UC)
  #If any of the kernels corresponding to this category are enabled
  #set the category as enabled
  IF(KOKKOSKERNELS_ENABLE_${NAME_UC})
    SET(KOKKOSKERNELS_ENABLE_${CATEGORY_UC} ON PARENT_SCOPE)
  ENDIF()
 ENDFOREACH()
ENDFUNCTION()

MACRO(KOKKOSKERNELS_APPEND_ENABLED_SOURCES LIST_NAME)
  CMAKE_PARSE_ARGUMENTS(SOURCES
    ""
    ""
    "${KOKKOSKERNELS_AVAILABLE_LIST}"
    ${ARGN}
  )
  SET(${LIST_NAME})
  FOREACH(KERNEL ${KOKKOSKERNELS_AVAILABLE_LIST})
    IF (KOKKOSKERNELS_ENABLE_${KERNEL})
      #this is a macro - so source dir is the calling directory
      FOREACH(FILE ${SOURCES_${KERNEL}})
        SET(SRCFILE ${CMAKE_CURRENT_SOURCE_DIR}/${FILE})
        IF (NOT EXISTS "${SRCFILE}")
          MESSAGE(FATAL_ERROR "Attempting to add non-existent source file ${SRCFILE}")
        ENDIF()
        LIST(APPEND ${LIST_NAME} ${SRCFILE})
      ENDFOREACH()
    ENDIF()
  ENDFOREACH()
ENDMACRO()

MACRO(KOKKOSKERNELS_ADD_ENABLED_EXECUTABLE NAME)
  KOKKOSKERNELS_APPEND_ENABLED_SOURCES(SOURCES ${ARGN})
  IF(SOURCES)
    KOKKOSKERNELS_ADD_EXECUTABLE(${NAME}
      SOURCES ${SOURCES})
  ENDIF()
ENDMACRO()

MACRO(KOKKOSKERNELS_ADD_ENABLED_TESTS NAME)
  KOKKOSKERNELS_APPEND_ENABLED_SOURCES(SOURCES ${ARGN})
  IF(SOURCES)
    KOKKOSKERNELS_ADD_EXECUTABLE_AND_TEST(${NAME}
      SOURCES 
        ${SOURCES}
        ${KOKKOSKERNELS_UNIT_TEST_DIR}/Test_Main.cpp
      )
  ENDIF()
ENDMACRO()
