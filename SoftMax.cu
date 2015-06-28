#include <THC/THC.h>

#define MINUS_LOG_THRESHOLD -18.42
#define SOFTMAX_THREADS 128

__global__ void cunn_SoftMax_updateOutput_kernel(float *output, float *input, int nframe, int dim)
{
  __shared__ float buffer[SOFTMAX_THREADS+1];
  int k = blockIdx.x;
  float *input_k = input + k*dim;
  float *output_k = output + k*dim;

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  // max?
  buffer[threadIdx.x] = -FLT_MAX;
  for (int i=i_start; i<i_end; i+=i_step)
  {
    float z = input_k[i];
    if(buffer[threadIdx.x] < z)
      buffer[threadIdx.x] = z;
  }

  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    float max_k = -FLT_MAX;
    for (int i=0; i<blockDim.x; i++)
    {
      if(max_k < buffer[i])
        max_k = buffer[i];
    }
    buffer[SOFTMAX_THREADS] = max_k;
  }

  __syncthreads();

  // sum?
  float max_k = buffer[SOFTMAX_THREADS];
  buffer[threadIdx.x] = 0;
  for (int i=i_start; i<i_end; i+=i_step) {
    float z = __expf(input_k[i]-max_k);
    buffer[threadIdx.x] += z;
    output_k[i] = z;
  }

  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    float sum_k = 0;
    for (int i=0; i<blockDim.x; i++)
      sum_k += buffer[i];
    buffer[SOFTMAX_THREADS] = sum_k;
  }

  __syncthreads();

  // softmax
  float sum_k = buffer[SOFTMAX_THREADS];
  for (int i=i_start; i<i_end; i+=i_step)
    output_k[i] = output_k[i] / sum_k;
}

extern "C"
void cunnrelease_SoftMax_updateOutput(THCState *state,
    THCudaTensor *input, THCudaTensor *output)
{
  THAssert(THCudaTensor_checkGPU(state, 2, input, output));

  input = THCudaTensor_newContiguous(state, input);
  THCudaTensor_resizeAs(state, output, input);

  if(input->nDimension == 1)
  {
    dim3 blocks(1);
    dim3 threads(SOFTMAX_THREADS);
    cunn_SoftMax_updateOutput_kernel<<<blocks,threads,
      0, THCState_getCurrentStream(state)>>>(THCudaTensor_data(state, output),
                                             THCudaTensor_data(state, input),
                                             1, input->size[0]);
  }
  else if(input->nDimension == 2)
  {
    dim3 blocks(input->size[0]);
    dim3 threads(SOFTMAX_THREADS);
    cunn_SoftMax_updateOutput_kernel<<<blocks,threads,
      0, THCState_getCurrentStream(state)>>>(THCudaTensor_data(state, output),
                                             THCudaTensor_data(state, input),
                                             input->size[0], input->size[1]);
  }
  else
    THError("vector or matrix expected");

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THCudaTensor_free(state, input);
}
