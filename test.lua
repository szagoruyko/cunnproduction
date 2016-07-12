require 'cunn'
require 'inn'

local ffi = require 'ffi'

ffi.cdef[[
void cunnrelease_Linear(THCState *state,
    THCudaTensor *input,
    THCudaTensor *output,
    THCudaTensor *weight,
    THCudaTensor *bias,
    THCudaTensor *buffer);
]]

local C = ffi.load'./build/libcunnproduction.so'

local mytester = torch.Tester()

local cunnreleasetest = torch.TestSuite()

precision_forward = 0

function cunnreleasetest.Linear()
  local from = math.random(1,32)
  local to = math.random(1,32)
  local bs = math.random(2,32)

  local module = nn.Linear(from, to):cuda()
  local input = torch.rand(bs, from):cuda()
  local groundtruth = module:forward(input)

  local output = torch.CudaTensor()
  local buffer = torch.CudaTensor(bs):fill(1)
  C.cunnrelease_Linear(cutorch.getState(),
  	input:cdata(),
	output:cdata(),
	module.weight:cdata(),
	module.bias:cdata(),
	buffer:cdata())

  local error = output - groundtruth
  mytester:asserteq(error:abs():max(), precision_forward, 'error on state (forward) ')
end

mytester:add(cunnreleasetest)
mytester:run()
