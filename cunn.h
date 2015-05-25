#include <THC/THC.h>
#include <memory>
#include <vector>

namespace cunn {

/*
 * Abstract class as nn.Module
 */
class Module {
public:
  typedef std::shared_ptr<Module> Ptr;

  Module(THCState *state);
  ~Module();

  virtual THCudaTensor* forward(THCudaTensor *input) = 0;
  THCState *state;
  THCudaTensor *output;
};

/*
 * nn.Sequential
 */
class Sequential {
public:
  void add(Module::Ptr module);

  THCudaTensor* forward(THCudaTensor* input);

  std::vector<Module::Ptr> modules;
};


/*
 * nn.SpatialConvolutionMM
 */
class SpatialConvolution : public Module {
public:
  SpatialConvolution(THCState *state, int nInputPlane, int nOutputPlane, int kW, int kH, int dW = 1, int dH = 1, int padding = 0);
  ~SpatialConvolution();

  THCudaTensor* forward(THCudaTensor *input);

  THCudaTensor *weight, *bias;
  THCudaTensor *finput, *fgradinput;
  int nInputPlane, nOutputPlane, kW, kH, dW, dH, padding;
};

/*
 * nn.SpatialMaxPooling
 */
class SpatialMaxPooling : public Module {
public:
  SpatialMaxPooling(THCState *state, int kW, int kH, int dW, int dH);
  ~SpatialMaxPooling();

  THCudaTensor* forward(THCudaTensor *input);

  int kW, kH, dW, dH;
};

/*
 * nn.ReLU
 */
class ReLU : public Module {
public:
  ReLU(THCState *state);
  ~ReLU();

  THCudaTensor* forward(THCudaTensor *input);
};

/*
 * nn.Linear
 */
class Linear : public Module {
public:
  Linear(THCState *state, int nInputPlane, int nOutputPlane);
  ~Linear();

  THCudaTensor* forward(THCudaTensor *input);

  THCudaTensor *weight, *bias, *buffer;
  int nOutputPlane, nInputPlane;
};

}
