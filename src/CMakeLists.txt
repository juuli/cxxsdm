

set(LAPACKE "")
find_library(LAPACKE lapack)
message(STATUS ${LAPACKE})

# Add c++11 flag and some for unix based systems
if(WIN32)
else()
  SET(CMAKE_CXX_FLAGS "-frounding-math -fPIC -O3 -Wall -Wextra")
  set(CMAKE_CXX_FLAGS "-std=gnu++11 ${CMAKE_CXX_FLAGS}")
endif()

#ADD INCLUDE & LINK DIRECTORIES
INCLUDE_DIRECTORIES(/opt/local/include/)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include/)

LINK_DIRECTORIES(/opt/local/lib)
LINK_DIRECTORIES(/usr/lib)

# EXECUTABLES
ADD_EXECUTABLE(DecompositionMain decomposition_main.cpp spatialDecomposition.cpp )
ADD_LIBRARY(SDM SHARED spatialDecomposition.cpp )

if(WIN32)
 set(FFTW_LIB libfftw3f)
else()
  set(FFTW_LIB fftw3f)
endif()

# LINK TARGETS
TARGET_LINK_LIBRARIES(DecompositionMain ${FFTW_LIB} )

TARGET_LINK_LIBRARIES(SDM ${FFTW_LIB} ${LAPACKE})

# Set target output directories
set_target_properties( DecompositionMain PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/ )
set_target_properties( SDM PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/ )
set_target_properties( SDM PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/ )
