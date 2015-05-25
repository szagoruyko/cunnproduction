#include "cunn.h"

extern "C" {
void cunnrelease_SpatialConvolution(THCState *state,
    THCudaTensor *input,
    THCudaTensor *weight,
    THCudaTensor *bias,
    THCudaTensor *columns,
    THCudaTensor *ones,
    THCudaTensor *output,
    int nInputPlane, int nOutputPlane, int kW, int kH, int dW, int dH, int padding);

void cunnrelease_SpatialMaxPooling(THCState* state,
    THCudaTensor* input, 
    THCudaTensor* output,
    int kW, int kH, int dW, int dH, bool is_ceil);

void cunnrelease_SpatialAveragePooling(THCState* state,
    THCudaTensor* input,
    THCudaTensor* output,
    int kW, int kH, int dW, int dH, bool is_ceil);

void cunnrelease_Linear(THCState *state,
    THCudaTensor *input,
    THCudaTensor *output,
    THCudaTensor *weight,
    THCudaTensor *bias,
    THCudaTensor *buffer);

void cunnrelease_ReLUIP(THCState *state,
    THCudaTensor *input);
}

namespace cunn {

Module::Module(THCState *state) : state(state) {
  output = THCudaTensor_new(state);
}

Module::~Module() {
  THCudaTensor_free(state, output);
}


SpatialConvolutionMM::SpatialConvolutionMM(THCState *state,
    int nInputPlane, int nOutputPlane, int kW, int kH, int dW, int dH, int padding) :
    	Module(state), nInputPlane(nInputPlane), nOutputPlane(nOutputPlane), kW(kW), kH(kH), dW(dW), dH(dH), padding(padding)  {
  weight = THCudaTensor_newWithSize2d(state, nOutputPlane, nInputPlane*kW*kH);
  bias = THCudaTensor_newWithSize1d(state, nOutputPlane);
  finput = THCudaTensor_new(state);
  fgradinput = THCudaTensor_new(state);
}

THCudaTensor*
SpatialConvolutionMM::forward(THCudaTensor *input)
{
  cunnrelease_SpatialConvolution(state, input, weight, bias, finput, fgradinput, output,
      nInputPlane, nOutputPlane, kW, kH, dW, dH, padding);
  return output;
}

SpatialConvolutionMM::~SpatialConvolutionMM()
{
  THCudaTensor_free(state, weight);
  THCudaTensor_free(state, bias);
  THCudaTensor_free(state, finput);
  THCudaTensor_free(state, fgradinput);
}

SpatialMaxPooling::SpatialMaxPooling(THCState *state, int kW, int kH, int dW, int dH) :
  Module(state), kW(kW), kH(kH), dW(dW), dH(dH) {}

SpatialMaxPooling::~SpatialMaxPooling() {}

THCudaTensor*
SpatialMaxPooling::forward(THCudaTensor *input)
{
  cunnrelease_SpatialMaxPooling(state, input, output, kW, kH, dW, dH, false);
  return output;
}


ReLU::ReLU(THCState *state) : Module(state) {}

ReLU::~ReLU() {}

THCudaTensor*
ReLU::forward(THCudaTensor *input)
{
  cunnrelease_ReLUIP(state, input);
  return input;
}


Linear::Linear(THCState *state, int nInputPlane, int nOutputPlane) :
  Module(state), nInputPlane(nInputPlane), nOutputPlane(nOutputPlane)
{
  weight = THCudaTensor_newWithSize2d(state, nOutputPlane, nInputPlane);
  bias = THCudaTensor_newWithSize1d(state, nOutputPlane);
  buffer = THCudaTensor_new(state);
}

Linear::~Linear()
{
  THCudaTensor_free(state, weight);
  THCudaTensor_free(state, bias);
  THCudaTensor_free(state, buffer);
}

THCudaTensor*
Linear::forward(THCudaTensor *input)
{
  cunnrelease_Linear(state, input, output, weight, bias, buffer);
  return output;
}

Module::Ptr
Sequential::get(int j) const
{
  return modules[j];
}

void
Sequential::add(Module::Ptr module)
{
  modules.push_back(module);
}

THCudaTensor*
Sequential::forward(THCudaTensor* input)
{
  THCudaTensor* output = input;
  for(auto& it: modules)
    output = it->forward(output);
  return output;
}

Reshape::Reshape(THCState *state, const std::vector<size_t>& sizes) : Module(state), sizes(sizes) {}

Reshape::~Reshape() {}

THCudaTensor*
Reshape::forward(THCudaTensor* input)
{
  size_t ndim = THCudaTensor_nDimension(state, input);
  // support only one case for now
  THCudaTensor_resize2d(state, output, input->size[0], sizes[0]);
  THCudaTensor_copy(state, output, input);
  return output; 
}

} // namespace cunn
