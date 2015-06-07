CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(CUDA 7.0 REQUIRED) # required for std::shared_ptr

add_definitions(-std=c++11 -fPIC)
LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_20 -Xcompiler -fPIC")

INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/THC")
LINK_DIRECTORIES("${Torch_INSTALL_LIB}")

SET(src-cuda SpatialMaxPooling.cu SpatialAveragePooling.cu SpatialConvolution.cu Linear.cu ReLU.cu cunn.cpp)

CUDA_ADD_LIBRARY(cunnproduction SHARED ${src-cuda})
CUDA_ADD_LIBRARY(cunnproduction_static STATIC ${src-cuda})

TARGET_LINK_LIBRARIES(cunnproduction TH THC)

INSTALL(TARGETS cunnproduction LIBRARY DESTINATION lib)
