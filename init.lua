local ffi  = require 'ffi'
local torchffi = require 'torchffi'
local fftw = require 'fftw3'

local signal = signal or {}

local function fftGeneric(inp, direction)
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
   input = input:contiguous() -- make sure input is contiguous
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

--[[
   1D FFT
   Takes Real inputs (1D tensor of N points)
   or complex inputs 2D tensor of (Nx2) size for N points
   
   Output matches with matlab output
]]--
function signal.fft(input)
   return fftGeneric(input, fftw.FORWARD)
end

--[[
   inverse 1D FFT
   Takes Real inputs (1D tensor of N points)
   or complex inputs 2D tensor of (Nx2) size for N points

   Output matches with matlab output
]]--
function signal.ifft(input)
   return fftGeneric(input, fftw.BACKWARD)
end

--[[
 real to complex dft.
   This function retains only the positive frequencies.
   Input is a 1D real tensor
   Output is 2D complex tensor of size (input:size(1)/2 + 1, 2)
]]--
function signal.rfft(inp)
   local input
   if inp:dim() == 1 then
      input = inp:double()
   else
      error('Input has to be 1D Tensor of size N (Real FFT with N points)')
   end
   input = input:contiguous() -- make sure input is contiguous
   local input_data = torch.data(input) -- double*

   local output = torch.DoubleTensor(math.floor((input:size(1)/2) + 1), 2);
   local output_data = torch.data(output);
   local output_data_cast = ffi.cast('fftw_complex*', output_data)

   local flags = fftw.ESTIMATE
   local plan  = fftw.plan_dft_r2c_1d(input:size(1), input_data, output_data_cast, flags)
   fftw.execute(plan)
   fftw.destroy_plan(plan)
   return output:typeAs(inp)
end

--[[
   complex to real dft. This function is the exact inverse of signal.rfft
]]--
function signal.irfft(inp)
   local input
   if inp:dim() == 2 and inp:size(2) == 2 then
      input = inp:double()
   else
      error('Input has to be 2D Tensor of size Nx2 (Complex input with N points)')
   end
   input = input:contiguous() -- make sure input is contiguous
   local input_data = torch.data(input) -- double*
   local input_data_cast = ffi.cast('fftw_complex*', input_data)

   local output = torch.DoubleTensor((input:size(1) - 1) * 2);
   local output_data = torch.data(output);

   local flags = fftw.ESTIMATE
   local plan  = fftw.plan_dft_c2r_1d(input:size(1), input_data_cast, output_data, flags)
   fftw.execute(plan)
   fftw.destroy_plan(plan)
   return output:typeAs(inp)
end

local function fft2Generic(inp, direction)
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
   input = input:contiguous() -- make sure input is contiguous
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

--[[
   2D FFT
   Takes Real inputs (2D tensor of NxM points)
   or complex inputs 3D tensor of (NxMx2) size for NxM points

   Output matches with matlab output
]]--
function signal.fft2(input)
   return fft2Generic(input, fftw.FORWARD)
end

--[[
   2D Inverse FFT
   Takes Real inputs (2D tensor of NxM points)
   or complex inputs 3D tensor of (NxMx2) size for NxM points

   Output matches with matlab output
]]--
function signal.ifft2(input)
   return fft2Generic(input, fftw.BACKWARD)
end

local function fft3Generic(inp, direction)
   local input
   if inp:dim() == 3 then -- assume that phase is 0
      input = torch.DoubleTensor(inp:size(1), inp:size(2), input:size(3), 2):fill(0)
      input[{{}, {}, {}, 1}] = inp
   elseif inp:dim() == 4 and inp:size(4) == 2 then
      input = inp:double()
   else
      error('Input has to be 3D Tensor of size MxNxP (Real 3D FFT with MxNxP points) or ' .. 
	       '4D Tensor of size MxNxPx2 (Complex FFT with MxNxP points')
   end
   input = input:contiguous() -- make sure input is contiguous
   local input_data = torch.data(input) -- double*
   local input_data_cast = ffi.cast('fftw_complex*', input_data)

   local output = torch.DoubleTensor(input:size(1), input:size(2), input:size(3), 2);
   local output_data = torch.data(output);
   local output_data_cast = ffi.cast('fftw_complex*', output_data)

   local flags = fftw.ESTIMATE
   local plan  = fftw.plan_dft_3d(input:size(1), input:size(2), input:size(3),
				  input_data_cast, output_data_cast, direction, flags)
   fftw.execute(plan)
   fftw.destroy_plan(plan)
   if direction == fftw.BACKWARD then
      output = output:div(input:size(1) * input:size(2) * input:size(3)) -- normalize
   end
   return output:typeAs(inp)
end

--[[
   3D FFT
   Takes Real inputs (3D tensor of NxMxP points)
   or complex inputs 4D tensor of (NxMxPx2) size for NxMxP points

   Output matches with matlab output
]]--
function signal.fft3(input)
   return fft3Generic(input, fftw.FORWARD)
end

--[[
   3D Inverse FFT
   Takes Real inputs (3D tensor of NxMxP points)
   or complex inputs 4D tensor of (NxMxPx2) size for NxMxP points

   Output matches with matlab output
]]--
function signal.ifft3(input)
   return fft3Generic(input, fftw.BACKWARD)
end

local function apply_window(window, window_type)
   window_type = window_type or 'rect'
   local window_size = window:size(1)
   local m = window_size - 1
   window = window:contiguous()
   local wdata = torch.data(window)
   if window_type == 'hamming' then
      for i=0,window_size-1 do
	 wdata[i] = wdata[i] * (.53836 - .46164 * math.cos(2 * math.pi * i / m));
      end
   elseif window_type == 'hann' then
      for i=0,window_size-1 do
	 wdata[i] = wdata[i] * (.5 - .5 * math.cos(2 * math.pi * i / m));
      end
   elseif window_type == 'bartlett' then
      for i=0,window_size-1 do
	 wdata[i] = wdata[i] * (2 / m * ((m/2) - math.abs(i - (m/2))));
      end
   end
   return window
end

--[[
   1D short-time fourier transforms
   Run a window across your signal and calculate fourier transforms on that window.
   To make sure that the windows are not discontinuous at the edges, you can optionally apply a window preprocessor.
   The available window preprocessors are: hamming, hann, bartlett
]]--
function signal.stft(inp, window_size, window_stride, window_type)
   if inp:dim() ~= 1 then error('Need 1D Tensor input') end
   local input = inp:double()
   local length = input:size(1)
   local nwindows = math.floor(((length - window_size)/window_stride) + 1);
   local noutput  = math.floor(window_size/2 + 1);
   local output   = torch.DoubleTensor(nwindows, noutput, 2):zero()
   local window_index = 1
   for i=1,length,window_stride do
      if (i+window_size-1) > length then break; end
      local window = input[{{i,i+window_size}}]
      -- apply preprocessing
      apply_window(window, window_type)
      -- fft
      local winout = signal.rfft(window)
      output[window_index] = winout    
      window_index = window_index + 1
   end
   return output:typeAs(inp)
end

--[[
   Takes the stft and generates a pretty spectrogram by
   taking the magnitude of the stft, and applying a (natural log * 10)
   Also transposes the output, to have time on the X axis.
]]--
function signal.spectrogram(inp, window_size, window_stride)
   -- calculate stft
   local stftout = signal.stft(inp, window_size, window_stride)
      -- calculate magnitude of signal and convert to dB to make it look prettier
   local stftout_r = stftout:select(3,1)
   local stftout_c = stftout:select(3,2)
   stftout_r:pow(2)
   stftout_c:pow(2)
   local stftout_magnitude = stftout_r + stftout_c
   stftout_magnitude = stftout_magnitude + 0.01 -- adding constant to avoid log(0)
   output = stftout_magnitude:log() * 10
   return output:transpose(1,2)
end


--[[
   Correct phase angles to produce smoother phase plots   
   Unwrap radian phases by adding multiples of 2*pi as appropriate to
   remove jumps greater than **tol**. **tol** defaults to pi.   

   Output matches with matlab output
]]--
function signal.unwrap(a, tol)
   if a:dim() ~= 1 then error('Input has to be 1D tensor') end
   tol = tol or math.pi
   tol = math.abs(tol)
   local twopi = 2*math.pi;
   local m = a:size(1)

   a = a:clone()
  -- Handle case where we only have one sample
   if (m == 1) then return a end

   a = a:contiguous()
   local adata = torch.data(a)
   for i=0,m-2 do
      local val = adata[i+1] - adata[i]
      if math.abs(val) > tol then
	 adata[i+1] = adata[i+1] - twopi * math.ceil((val - tol) / twopi)
      end
   end
   return a
end

--[[
   unwraps the phase and removes phase corresponding to integer lag.

   Output matches with matlab output
]]--
function signal.rcunwrap(x)
   if x:dim() ~= 1 then error('Input has to be 1D tensor') end
   local n = x:size(1)
   local nh = math.floor((n+1)/2); -- since n is positive, nh always rounds towards zero
   local y = signal.unwrap(x):contiguous()
   local ydata = torch.data(y)
   local nd = math.floor(0.5 + (y[nh+1]/math.pi)) -- round
   if nd == 0 then return y,nd; end
   for i=0,y:size(1)-1 do
      ydata[i] = ydata[i] - (math.pi * nd * i / nh)
   end
   return y,nd
end

--[[
   Adds phase corresponding to integer lag

   Output matches with matlab output
]]--
function signal.rcwrap(y, nd)
   if y:dim() ~= 1 then error('Input has to be 1D tensor') end
   y = y:clone():contiguous()
   nd = nd or 0
   if nd == 0 then return y; end
   local n = y:size(1)
   local nh = math.floor((n+1)/2);
   local ydata = torch.data(y)
   for i=0,y:size(1)-1 do
      ydata[i] = ydata[i] + (math.pi*nd*i/nh);
   end
   return y
end

local function angle(h)
   return torch.atan2(h[{{},2}],h[{{},1}])
end

--[[
   1D Complex cepstral analysis
   Returns the cepstrum and a phase shift factor "nd" that is useful to invert the signal back.

   Output matches with matlab output
]]--
function signal.cceps(x)
   --[[
      logh = log(abs(h)) + sqrt(-1)*rcunwrap(angle(h));
      y = real(ifft(logh));
   ]]--   
   if not(x:dim() == 1 or (x:dim() == 2 and x:size(2) == 2)) then 
      error('Input has to be 1D tensor or Nx2 2D tensor') 
   end
   local h = signal.fft(x);
   local hsquare = torch.pow(h,2)
   local habs = torch.sqrt(hsquare[{{},1}] + hsquare[{{},2}])
   local logh = h:clone();
   local nd
   logh[{{},1}] = torch.log(habs)
   logh[{{},2}],nd = signal.rcunwrap(angle(h))
   local y = signal.ifft(logh)
   return y[{{},1}], nd -- real part
end

--[[
   1D Inverse Complex cepstral analysis.
   Takes in the outputs of cceps to produce the input signal back

   Output matches with matlab output
]]--
function signal.icceps(xhat,nd)
   if xhat:dim() ~= 1 then error('Input has to be 1D tensor') end
   nd = nd or 0
   local logh = signal.fft(xhat);
   local h = logh:clone()
   local real = logh[{{},1}]
   local imag = signal.rcwrap(logh[{{},2}],nd)
   h[{{},1}] = torch.exp(real):cmul( torch.cos(imag))
   h[{{},2}] = torch.exp(real):cmul(torch.sin(imag));  
   local x = signal.ifft(h);
   return x[{{},1}]
end

--[[
   Real cepstrum and minimum phase reconstruction
   The real cepstrum is the inverse Fourier transform of the real logarithm of the magnitude of the Fourier transform of a sequence.

   Output matches with matlab output
]]--
function signal.rceps(x)
   if x:dim() ~= 1 then error('Input has to be 1D tensor') end
   -- y=real(ifft(log(abs(fft(x)))));
   local h = signal.fft(x);
   local hsquare = torch.pow(h,2)
   local hlog = torch.log(torch.sqrt(hsquare[{{},1}] + hsquare[{{},2}]))
   local ih = signal.ifft(hlog)
   return ih[{{},1}]
end

local function dctGeneric(inp, direction)
   local input
   if inp:dim() == 1 then
      input = inp:double()
   else
      error('Input has to be 1D Tensor of size N (Real FFT with N points)')
   end
   input = input:contiguous() -- make sure input is contiguous
   local input_data = torch.data(input) -- double*
   local output = input:clone()
   local output_data = torch.data(output) -- double*
   local flags = fftw.ESTIMATE
   local dcttype
   if direction == fftw.FORWARD then
      dcttype = fftw.r2r_kind(fftw.REDFT10)
   else
      dcttype = fftw.r2r_kind(fftw.REDFT01)
   end   
   local plan = fftw.plan_r2r_1d(input:size(1), input_data, output_data, dcttype, flags)
   fftw.execute(plan)
   fftw.destroy_plan(plan)
   if direction == fftw.BACKWARD then
      output = output:div(2 * input:size(1)) -- normalize by 2n
   end
   return output:typeAs(inp)   
end

--[[
   1D Discrete Cosine Transform (DCT)
   Takes Real inputs (1D tensor of N points)

   To see what is exactly computed, see section REDFT10 over here: 
   http://www.fftw.org/doc/1d-Real_002deven-DFTs-_0028DCTs_0029.html
]]--
function signal.dct(input)
   return dctGeneric(input, fftw.FORWARD)
end

--[[
   inverse 1D Discrete Cosine Transform (DCT)
   Takes Real inputs (1D tensor of N points)

   To see what is exactly computed, see section REDFT01 over here: 
   http://www.fftw.org/doc/1d-Real_002deven-DFTs-_0028DCTs_0029.html
]]--
function signal.idct(input)
   return dctGeneric(input, fftw.BACKWARD)
end

local function dct2Generic(inp, direction)
   local input
   if inp:dim() == 2 then
      input = inp:double()
   else
      error('Input has to be 2D Tensor of size NxM (Real FFT with NxM points)')
   end
   input = input:contiguous() -- make sure input is contiguous
   local input_data = torch.data(input) -- double*
   local output = input:clone()
   local output_data = torch.data(output) -- double*
   local flags = fftw.ESTIMATE
   local dcttype
   if direction == fftw.FORWARD then
      dcttype = fftw.r2r_kind(fftw.REDFT10)
   else
      dcttype = fftw.r2r_kind(fftw.REDFT01)
   end   
   local plan = fftw.plan_r2r_2d(input:size(1), input:size(2), 
				 input_data, output_data, dcttype, dcttype, flags)
   fftw.execute(plan)
   fftw.destroy_plan(plan)
   if direction == fftw.BACKWARD then
      output = output:div(2 * input:size(1) * 2 * input:size(2)) -- normalize by 2n * 2m
   end
   return output:typeAs(inp)   
end

--[[
   2D Discrete Cosine Transform (DCT)
   Takes Real inputs (2D tensor of NxM points)

   To see what is exactly computed, see section REDFT10 over here: 
   http://www.fftw.org/doc/1d-Real_002deven-DFTs-_0028DCTs_0029.html
]]--
function signal.dct2(input)
   return dct2Generic(input, fftw.FORWARD)
end

--[[
   inverse 2D Discrete Cosine Transform (DCT)
   Takes Real inputs (2D tensor of NxM points)

   To see what is exactly computed, see section REDFT01 over here: 
   http://www.fftw.org/doc/1d-Real_002deven-DFTs-_0028DCTs_0029.html
]]--
function signal.idct2(input)
   return dct2Generic(input, fftw.BACKWARD)
end

local function dct3Generic(inp, direction)
   local input
   if inp:dim() == 3 then
      input = inp:double()
   else
      error('Input has to be 3D Tensor of size NxM (Real FFT with NxMxP points)')
   end
   input = input:contiguous() -- make sure input is contiguous
   local input_data = torch.data(input) -- double*
   local output = input:clone()
   local output_data = torch.data(output) -- double*
   local flags = fftw.ESTIMATE
   local dcttype
   if direction == fftw.FORWARD then
      dcttype = fftw.r2r_kind(fftw.REDFT10)
   else
      dcttype = fftw.r2r_kind(fftw.REDFT01)
   end   
   local plan = fftw.plan_r2r_3d(input:size(1), input:size(2), input:size(3),
				 input_data, output_data, 
				 dcttype, dcttype, dcttype, flags)
   fftw.execute(plan)
   fftw.destroy_plan(plan)
   if direction == fftw.BACKWARD then
      -- normalize by 2n * 2m * 2p
      output = output:div(2 * input:size(1) * 2 * input:size(2) * 2 * input:size(3))
   end
   return output:typeAs(inp)   
end

--[[
   3D Discrete Cosine Transform (DCT)
   Takes Real inputs (3D tensor of NxMXP points)

   To see what is exactly computed, see section REDFT10 over here: 
   http://www.fftw.org/doc/1d-Real_002deven-DFTs-_0028DCTs_0029.html
]]--
function signal.dct3(input)
   return dct3Generic(input, fftw.FORWARD)
end

--[[
   inverse 3D Discrete Cosine Transform (DCT)
   Takes Real inputs (3D tensor of NxMxP points)

   To see what is exactly computed, see section REDFT01 over here: 
   http://www.fftw.org/doc/1d-Real_002deven-DFTs-_0028DCTs_0029.html
]]--
function signal.idct3(input)
   return dct3Generic(input, fftw.BACKWARD)
end

--[[
   Discrete-time analytic signal using Hilbert transform
   Takes 1D inputs

   Output matches with matlab output
]]--
function signal.hilbert(xr)
   if xr:dim() ~= 1 then error('Input has to be 1D tensor') end
   local x = signal.fft(xr)
   local h = xr:clone():fill(0):contiguous()
   local n = h:size(1)
   local nby2 = math.floor(n/2)
   local hd = torch.data(h)
   if 2*nby2 == n then --even
      hd[0] = 1 -- i=1
      hd[nby2] = 1 -- i=(n/2)+1
      for i=1,(nby2-1) do -- 2,3,...,(n/2)
	 hd[i] = 2
      end
   else -- odd
      hd[0] = 1 -- i=1
      for i=1,(nby2) do -- 2,3,...,((n+1)/2)
	 hd[i] = 2
      end
   end
   x[{{},1}] = x[{{},1}]:cmul(h)
   x[{{},2}] = x[{{},2}]:cmul(h)
   return signal.ifft(x)
end

--[[
 [B A] = butter(LPorder,cutoff,'low'); %design f_nyquist/2-lowpass filter
LPorder = 6
cutoff = 0.5

output is LPorder+1 points

]]--
local function butter(LPorder, cutoff, types)
   
end


return signal