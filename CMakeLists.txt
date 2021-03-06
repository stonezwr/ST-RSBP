cmake_minimum_required (VERSION 3.0)
project (CUDA-RSNN)

# The version number.
set (CUDA-RSNN_VERSION_MAJOR 2)
set (CUDA-RSNN_VERSION_MINOR 0)

#    CMake Scripts dir
#set(CMAKE_SCRIPT_DIR ${CMAKE_SOURCE_DIR}/CMakeScripts)

#CMake module path for custom module finding
#set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SCRIPT_DIR})

find_package (CUDA REQUIRED)
include_directories (${CUDA_INCLUDE_DIRS})
#include_directories (${CUDA_SDK_ROOT_DIR})
MESSAGE("${CUDA_INCLUDE_DIRS}/../samples/common/inc")
include_directories ("${CUDA_INCLUDE_DIRS}/../samples/common/inc")
#/usr/local/cuda/NVIDIA_CUDA-7.0_Samples/common/inc")
set(CUDA_SEPARABLE_COMPILATION ON)

# opencv
find_package( OpenCV REQUIRED )
INCLUDE_DIRECTORIES( ${OPENCV_INCLUDE_DIR} )
set(OpenCV_LIBRARIES ${OpenCV_LIBS})

FIND_PACKAGE( OpenMP)
if(OPENMP_FOUND)
	message("OPENMP FOUND")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
include_directories(SYSTEM ${OpenMP_INCLUDE_PATH})



include_directories(${PROJECT_SOURCE_DIR})

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -rdynamic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -rdynamic")


set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
	-gencode arch=compute_60,code=sm_60 # GPU computiblility , this is a code needs to be changed according to the GPU type
  -O3
  --use_fast_math
#  -G
#  -g
  -rdc=true
  -w
  )

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
  )

MESSAGE(${CUDA_NVCC_FLAGS})

#add_library(${EXAMPLE_NAME}.o OBJECT ${source})
#set(EXAMPLE_OBJ_LIB $<TARGET_OBJECTS:${EXAMPLE_NAME}.o>)


file(GLOB CUDA_CNN_CU_SOURCES  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
    "readData/*" 
    "layers/*"
    "dataAugmentation/*" 
    "common/*" 
    "main.cpp"
    "net.*"
    "net_spiking.*")

    #  EXCLUDE_FROM_ALL "build/*")

cuda_add_executable(CUDA-RSNN ${CUDA_CNN_CU_SOURCES})
#cuda_add_library(CUDA-RSNN STATIC ${CUDA_CNN_CPP_SOURCES})
target_link_libraries(CUDA-RSNN
  ${OpenCV_LIBRARIES}
  ${OpenMP_CXX_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_cusparse_LIBRARY}
  ${CUDA_curand_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_cusolver_LIBRARY}
  )
CUDA_ADD_CUBLAS_TO_TARGET(CUDA-RSNN)

#add_subdirectory(net)
