local ffi  = require 'ffi'
local fftw = require 'fftw3'
local torchffi = require 'torchffi'
require 'signal'

local function fftr1d(input)
   assert(input:dim() == 1)
   local input_data = torch.data(input) -- double*

   local noutput = input:size(1)/2 + 1;
   local output = torch.DoubleTensor(noutput, 2);
   local output_data = torch.data(output);
   local output_data_cast = ffi.cast('fftw_complex*', output_data)

   local flags = fftw.ESTIMATE
   local plan  = fftw.plan_dft_r2c_1d(input:size(1), input_data, output_data_cast, flags)
   fftw.execute(plan)
   fftw.destroy_plan(plan)
   return output
end

local function ifftr1d(input)
   assert(input:dim() == 2)
   local input_data = torch.data(input) -- double*
   local input_data_cast = ffi.cast('fftw_complex*', input_data)

   local noutput = (input:size(1) - 1) * 2;
   local output = torch.DoubleTensor(noutput):zero();
   local output_data = torch.data(output);

   local flags = fftw.ESTIMATE
   local plan  = fftw.plan_dft_c2r_1d(input:size(1), input_data_cast, output_data, flags)
   fftw.execute(plan)
   fftw.destroy_plan(plan)
   return output
end



input = torch.Tensor(4,2):fill(0)
input[1][1] = 1
input[2][1] = 2
input[3][1] = 3
input[4][1] = 4

print(input)
output = torch.fft(input)
inputi = torch.ifft(output)
print(output)
print(inputi)

input = torch.Tensor(4);
input[1] = 1
input[2] = 2
input[3] = 3
input[4] = 4

output = torch.fft(input)
inputi = torch.ifft(output)
print(input)
print(output)
print(inputi)

input = torch.Tensor(2,2,2):fill(0)
input[1][1][1] = 1
input[1][2][1] = 2
input[2][1][1] = 3
input[2][2][1] = 4

print(input)
output = torch.fft2(input)
inputi = torch.ifft2(output)
print(output)
print(inputi)

input = torch.Tensor(2,2);
input[1][1] = 1
input[1][2] = 2
input[2][1] = 3
input[2][2] = 4

output = torch.fft2(input)
inputi = torch.ifft2(output)
print(input)
print(output)
print(inputi)


-- output = fftr1d(input)
-- inputi = ifftr1d(output)
-- print(output)

-- require 'audio'
-- input2 = input:clone()
-- input2:resize(1,4)
-- output2 = audio.stft(input2, 4, 'rect', 1)