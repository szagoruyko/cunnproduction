CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

FIND_PACKAGE(Torch)
FIND_PACKAGE(CUDA 7.0 REQUIRED) # required for std::shared_ptr

add_definitions(-std=c++11 -fPIC)
LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_20 -Xcompiler -fPIC")

IF(${Torch_INSTALL_INCLUDE})
  SET(${THC_INSTALL_INCLUDE} ${Torch_INSTALL_INCLUDE}/THC)
  SET(${TH_INSTALL_INCLUDE} ${Torch_INSTALL_INCLUDE}/TH)
  SET(${TH_INSTALL_LIB} ${Torch_INSTALL_LIB})
  SET(${THC_INSTALL_LIB} ${Torch_INSTALL_LIB})
ENDIF()

INCLUDE_DIRECTORIES(
  ${THC_INSTALL_INCLUDE}
  ${TH_INSTALL_INCLUDE}
  ${CUDA_INCLUDE_DIRS}
  cunnproduction
)

LINK_DIRECTORIES(${TH_INSTALL_LIB} ${THC_INSTALL_LIB})

SET(src-cuda SpatialMaxPooling.cu SpatialAveragePooling.cu SpatialConvolution.cu Linear.cu ReLU.cu cunn.cpp)

CUDA_ADD_LIBRARY(cunnproduction SHARED ${src-cuda})
CUDA_ADD_LIBRARY(cunnproduction_static STATIC ${src-cuda})

TARGET_LINK_LIBRARIES(cunnproduction TH THC)

INSTALL(TARGETS cunnproduction LIBRARY DESTINATION lib)
