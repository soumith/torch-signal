local ffi  = require 'ffi'
local torchffi = require 'torchffi'
local fftw = require 'fftw3'

local function fftGeneric(inp, n, direction)
   local input
   if inp:dim() == 1 then -- assume that phase is 0
      input = torch.DoubleTensor(inp:size(1), 2):fill(0)
      input[{{}, 1}] = inp
   elseif inp:dim() == 2 and inp:size(2) == 2 then
      input = inp:double()
   else
      error('Input has to be 1D Tensor of size N (Real FFT with N points) or ' .. 
	       '2D Tensor of size Nx2 (Complex FFT with N points)')
   end
   n = n or input:size(1)
   if input:size(1) < n then
      local temp = torch.DoubleTensor(n, 2):fill(0)
      temp[{{1, input:size(1)},{}}] = input
      input = temp
   elseif input:size(1) > n then
      input = input[{{1,n},{}}]
   end
   local input_data = torch.data(input) -- double*
   local input_data_cast = ffi.cast('fftw_complex*', input_data)

   local output = torch.DoubleTensor(input:size(1), 2);
   local output_data = torch.data(output);
   local output_data_cast = ffi.cast('fftw_complex*', output_data)

   local flags = fftw.ESTIMATE
   local plan  = fftw.plan_dft_1d(input:size(1), input_data_cast, output_data_cast, direction, flags)
   fftw.execute(plan)
   fftw.destroy_plan(plan)
   if direction == fftw.BACKWARD then
      output = output:div(input:size(1)) -- normalize
   end
   return output:typeAs(inp)
end


local function fft(input, n)
   return fftGeneric(input, n, fftw.FORWARD)
end

local function ifft(input, n)
   return fftGeneric(input, n, fftw.BACKWARD)
end

local function fft2Generic(inp, m, n, direction)
   local input
   if inp:dim() == 2 then -- assume that phase is 0
      input = torch.DoubleTensor(inp:size(1), inp:size(2), 2):fill(0)
      input[{{}, {}, 1}] = inp
   elseif inp:dim() == 3 and inp:size(3) == 2 then
      input = inp:double()
   else
      error('Input has to be 2D Tensor of size MxN (Real 2D FFT with MxN points) or ' .. 
	       '3D Tensor of size MxNx2 (Complex FFT with MxN points')
   end      
   m = m or input:size(1)
   n = n or input:size(2)
   if input:size(1) < m or input:size(2) < n then
      local temp = torch.DoubleTensor(m, n, 2):fill(0)
      temp[{{1, input:size(1)},{1, input:size(2)},{}}] = input
      input = temp
   elseif input:size(1) > m or input:size(2) > n then
      input = input[{{1,m},{1,n},{}}]
   end
   local input_data = torch.data(input) -- double*
   local input_data_cast = ffi.cast('fftw_complex*', input_data)

   local output = torch.DoubleTensor(input:size(1), input:size(2), 2);
   local output_data = torch.data(output);
   local output_data_cast = ffi.cast('fftw_complex*', output_data)

   local flags = fftw.ESTIMATE
   local plan  = fftw.plan_dft_2d(input:size(1), input:size(2), 
				  input_data_cast, output_data_cast, direction, flags)
   fftw.execute(plan)
   fftw.destroy_plan(plan)
   if direction == fftw.BACKWARD then
      output = output:div(input:size(1) * input:size(2)) -- normalize
   end
   return output:typeAs(inp)
end

local function fft2(input, m, n)
   return fft2Generic(input, m, n, fftw.FORWARD)
end

local function ifft2(input, m, n)
   return fft2Generic(input, m, n, fftw.BACKWARD)
end

torch.fft = fft
torch.ifft = ifft

torch.fft2 = fft2
torch.ifft2 = ifft2