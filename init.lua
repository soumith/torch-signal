local ffi  = require 'ffi'
local torchffi = require 'torchffi'
local fftw = require 'fftw3'
local complex = require 'signal.complex'
require 'signal.extramath'

local signal = signal or {}
signal.experimental = {}
signal.incomplete = {}

local function fftGeneric(inp, direction)
   local input
   if inp:dim() == 1 then -- assume that phase is 0
      input = torch.DoubleTensor(inp:size(1), 2):zero()
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

   local output = torch.DoubleTensor(input:size(1), 2):zero();
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

   local output = torch.DoubleTensor(math.floor((input:size(1)/2) + 1), 2):zero();
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

   local output = torch.DoubleTensor((input:size(1) - 1) * 2):zero();
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
      input = torch.DoubleTensor(inp:size(1), inp:size(2), 2):zero()
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

   local output = torch.DoubleTensor(input:size(1), input:size(2), 2):zero();
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
      input = torch.DoubleTensor(inp:size(1), inp:size(2), input:size(3), 2):zero()
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

   local output = torch.DoubleTensor(input:size(1), input:size(2), input:size(3), 2):zero();
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
   1D complex short-time fourier transforms
   Run a window across your signal and calculate fourier transforms on that window.
   To make sure that the windows are not discontinuous at the edges, you can optionally apply a window preprocessor.
   The available window preprocessors are: hamming, hann, bartlett
]]--
function signal.stft(inp, window_size, window_stride, window_type)
   if inp:dim() ~= 1 then error('Need 1D Tensor input') end
   local input = inp:double()
   local length = input:size(1)
   local nwindows = math.floor(((length - window_size)/window_stride) + 1);
   local noutput  = window_size
   local output   = torch.DoubleTensor(nwindows, noutput, 2):zero()
   local window_index = 1
   for i=1,length,window_stride do
      if (i+window_size-1) > length then break; end
      local window = input[{{i,i+window_size-1}}]
      -- apply preprocessing
      apply_window(window, window_type)
      -- fft
      local winout = signal.fft(window)
      output[window_index] = winout
      window_index = window_index + 1
   end
   return output:typeAs(inp)
end

--[[
   1D real short-time fourier transforms
   Run a window across your signal and calculate fourier transforms on that window.
   To make sure that the windows are not discontinuous at the edges, you can optionally apply a window preprocessor.
   rfft is used for fourier transform, so only the positive frequencies are retained
   The available window preprocessors are: hamming, hann, bartlett
]]--
function signal.rstft(inp, window_size, window_stride, window_type)
   if inp:dim() ~= 1 then error('Need 1D Tensor input') end
   local input = inp:double()
   local length = input:size(1)
   local nwindows = math.floor(((length - window_size)/window_stride) + 1);
   local noutput  = math.floor(window_size/2 + 1);
   local output   = torch.DoubleTensor(nwindows, noutput, 2):zero()
   local window_index = 1
   for i=1,length,window_stride do
      if (i+window_size-1) > length then break; end
      local window = input[{{i,i+window_size-1}}]
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
   Takes the rstft(x) and generates a pretty spectrogram by
   taking the magnitude of the stft, and applying a (natural log * 10)
   Also transposes the output, to have time on the X axis.
]]--
function signal.spectrogram(inp, window_size, window_stride)
   -- calculate stft
   local stftout = signal.rstft(inp, window_size, window_stride)
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
   local nd = math.round((y[nh+1]/math.pi))
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

--[[
   1D Complex cepstral analysis
   Returns the cepstrum and a phase shift factor "nd" that is useful to invert the signal back.

   Output matches with matlab output
]]--
function signal.cceps(x)
   --[[
      logh = log(abs(h)) + sqrt(-1)*rcunwrap(complex.angle(h));
      y = real(ifft(logh));
   ]]--   
   if not(x:dim() == 1 or (x:dim() == 2 and x:size(2) == 2)) then 
      error('Input has to be 1D tensor or Nx2 2D tensor') 
   end
   local h = signal.fft(x);
   local logh = h:clone();
   logh[{{},1}] = torch.log(complex.abs(h))
   local nd
   logh[{{},2}],nd = signal.rcunwrap(complex.angle(h))
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
   h[{{},1}] = complex.real(logh)
   h[{{},2}] = signal.rcwrap(complex.imag(logh),nd)
   local x = signal.ifft(complex.exp(h));
   return complex.real(x)
end

--[[
   Real cepstrum and minimum phase reconstruction
   The real cepstrum is the inverse Fourier transform of the real logarithm of the magnitude of the Fourier transform of a sequence.

   Output matches with matlab output
]]--
function signal.rceps(x)
   if x:dim() ~= 1 then error('Input has to be 1D tensor') end
   -- y=real(ifft(log(abs(fft(x)))));
   return complex.real(signal.ifft(torch.log(complex.abs(signal.fft(x)))))
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
   local output = input:clone():zero()
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
   local output = input:clone():zero()
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
   local output = input:clone():zero()
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
   local h = xr:clone():zero():contiguous()
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

local function buttap(n)   
   local z = nil
   local pimag = torch.range(1,n-1,2):mul(math.pi/(2*n)):add(math.pi/2)
   local p = torch.Tensor(pimag:size(1),2):zero()
   p[{{},2}] = pimag
   p = complex.exp(p)
   local pconj = complex.conj(p)
   local pfull
   if n % 2 == 1 then
      pfull = torch.Tensor(p:size(1)*2 + 1, 2):zero()
      for i=1,p:size(1) do
	 pfull[i*2-1] = p[i]
	 pfull[i*2] = pconj[i]
      end
      pfull[pfull:size(1)][1] = -1
      pfull[pfull:size(1)][2] = 0
   else
      pfull = torch.Tensor(p:size(1)*2, 2):zero()
      for i=1,p:size(1) do
	 pfull[i*2-1] = p[i]
	 pfull[i*2] = pconj[i]
      end
   end
   local k = complex.prod(pfull)[1][1] -- only the real part
   return z, pfull, k
end

--[[
   Zero-pole to state-space conversion.
   Not fully implemented, dont use. Only implemented it to be enough for signal.butter()
]]--
local function zp2ss(z,p,k)
   error('TODO')
   if z ~= nil then
      error('Not implemented when z is not empty')
   end
   if p:size(1) % 2 == 1 then
      error('Not implemented for odd number of poles')
   end
   --  TODO: Strip infinities in p and z and throw away.
   
   -- p should have real elements and exact complex conjugate pair.
   p = cplxpair(p,0);
   -- Initialize state-space matrices for running series
   -- Now we have an even number of poles and zeros, although not
   -- necessarily the same number - there may be more poles.
   --   H(s) = (s^2+num(2)s+num(3))/(s^2+den(2)s+den(3))
   -- Loop through rest of pairs, connecting in series to build the model.
   i = 1;
   -- Take care of any left over unmatched pole pairs.
   --   H(s) = 1/(s^2+den(2)s+den(3))
   local a = torch.Tensor()
   local b = torch.zeros(1) -- 0,1
   local c = torch.zeros(1) -- 1,0
   local d = 1
   while i < p:size(1) do
      local den = complex.real(complex.poly(p[{{i,i+1},{}}]));
      local wn = math.sqrt(torch.prod(complex.abs(p[{{i,i+1},{}}])));
      if wn == 0 then wn = 1; end
      local t = torch.diag(torch.Tensor({1, 1/wn})); -- Balancing transformation
      local a1 = torch.gesv(torch.Tensor({{-den[2] -den[3]},{1,0}})*t, t);
      local b1 = torch.gesv(torch.Tensor({{1}, {0}}),t);
      local c1 = torch.Tensor({0,1})*t;
      local d1 = 0;
      -- [a,b,c,d] = series(a,b,c,d,a1,b1,c1,d1);
      -- Next lines perform series connection
      local ma1 = a:size(1);
      local na2 = a1:size(2);
      --[[
      a = [a zeros(ma1,na2); b1*c a1];
      b = [b; b1*d];
      c = [d1*c c1];
      d = d1*d;
      ]]--
      i = i + 2;
   end
   -- Apply gain k:
   c = c*k;
   d = d*k;   
end

--[[
   Digital butterworth filter
   output is n+1 points
   
   n - number of poles
   Wn - cutoff frequency (between 0 and 1.0). where 1.0 is half the sampling rate
   ftype - 'high', 'low', 'stop', 'pass'. If 'stop' or 'pass', the pass Wn = {W1, W2}
]]--
function signal.incomplete.butter(n, Wn, ftype)
   error('TODO')
   if n > 500 then
      error('Filter order too large. Keep it < 500')
   end
   ftype = ftype or 'low'
   if ftype == 'low' or ftype == 'high' then
      if type(Wn) ~= 'number' then error('Wn should be a number') end
   elseif ftype == 'pass' or ftype == 'stop' then
      if type(Wn) ~= 'table' or #(Wn) ~= 2 then
	 error('Wn should be a table with 2 numbers')
      end
   end
   local fs=2
   local u, Bw
   if type(Wn) == 'number' then
      u = 2*fs*math.tan(math.pi*Wn/fs);
   else
      u = {2*fs*math.tan(math.pi*Wn[1]/fs), 2*fs*math.tan(math.pi*Wn[2]/fs)}
   end
   if ftype == 'low' or ftype == 'high' then
      Wn = u;
   elseif ftype == 'pass' or ftype == 'stop' then
      Bw = u[2] - u[1];
      Wn = math.sqrt(u[1]*u[2]);   -- center frequency
   end
   local z,p,k = buttap(n)
   local a,b,c,d = zp2ss(z,p,k)
   if ftype == 'low' then
      a,b,c,d = signal.lp2lp(a,b,c,d,Wn);
   elseif ftype == 'pass' then
      a,b,c,d = signal.lp2bp(a,b,c,d,Wn,Bw);
   elseif ftype == 'high' then
      a,b,c,d = signal.lp2hp(a,b,c,d,Wn);
   elseif ftype == 'stop' then
      a,b,c,d = signal.lp2bs(a,b,c,d,Wn,Bw);
   end
   a,b,c,d = signal.bilinear(a,b,c,d,fs);
   local den = complex.poly(a);
   ----------------------------------------
   -- buttnum
   if type(Wn) == 'number' then
      Wn = 2*math.atan2(Wn,4)
   else
      Wn = {2*math.atan2(Wn[1],4), 2*math.atan2(Wn[2],4)}
   end
   local r,w
   if ftype == 'low' then
      r = torch.Tensor(n):fill(-1)
      w = 0
   elseif ftype == 'pass' then
      r = torch.Tensor(n,2):fill(1)
      r[{{},2}]:fill(-1)
      w = Wn
   elseif ftype == 'high' then
      r = torch.ones(n)
      w = math.pi
   elseif ftype == 'stop' then
      -- todo
      error('stop not implemented')
      -- too lazy to implement this    r = exp(j * Wn * ( (-1).^(0:2*n-1)' ));
      w = 0
   end
   local num = complex.poly(r);
   -- now normalize so |H(w)| == 1:
   local kern = torch.Tensor(num:size(1),2):fill(-1)
   kern[{{},1}] = torch.range(0,num:size(1) - 1)
   if ftype == 'low' or ftype == 'high' then
      kern[{{},1}] = kern[{{},1}]:mul(w)
   else
      error('not implemented')
   end
   num = complex.real(num * (kern * den))/ (kern * num)
   
   -- local kern = exp(-j*w*(0:length(num)-1));
   -- num = real(num*(kern*den(:))/(kern*num(:)));
--------------------------------------------------------
   return num, den
end

--[[
   returns an L-point Hann window in a 1D tensor. L must be a positive integer.
   When 'periodic' is specified, hann computes a length L+1 window and returns the first L points.
   flag: 'periodic' or 'symmetric'. 'symmetric' is default

   Output matches with matlab output
]]--
function signal.hann(L, flag)
   if flag == 'periodic' then
      L = L + 1
   end
   local N = L - 1   
   local out = torch.zeros(L)
   local odata = torch.data(out)
   for i=0,N do
      odata[i] = 0.5 * (1-math.cos(2 * math.pi * i / N))
   end

   if flag == 'periodic' then
      return out[{{1,L-1}}]
   else
      return out
   end      
end

--[[
   returns an N-point Blackman window in a 1D tensor. 
   N must be a positive integer.
   When 'periodic' is specified, computes a length N+1 window and returns the first N points.
   flag: 'periodic' or 'symmetric'. 'symmetric' is default

   Output matches with matlab output
]]--
function signal.blackman(N, flag)
   if N == 1 then return torch.Tensor({1}); end
   if flag == 'periodic' then
      N = N + 1
   end
   local M, idx
   if N % 2 == 1 then
      M = (N+1)/2
      idx = M-2
   else
      M = N/2
      idx = M-1
   end
   local out = torch.zeros(N)
   local odata = torch.data(out)
   for i=0,M-1 do
      odata[i] = 
	 0.42 
	 - 0.5 * math.cos(2*math.pi*i/(N-1)) 
	 + 0.08 * math.cos(4*math.pi*i/(N-1))
   end
   for i=M,N-1 do
      odata[i] = odata[idx]
      idx = idx - 1
   end

   if flag == 'periodic' then
      return out[{{1,N-1}}]
   else
      return out
   end      
end

--[[
   returns an N-point minimum 4-term Blackman-Harris window in a 1D tensor. 
   The window is minimum in the sense that its maximum sidelobes are minimized.
   N must be a positive integer.   
   flag: 'periodic' or 'symmetric'. 'symmetric' is default

   Output matches with matlab output
]]--
function signal.blackmanharris(N, flag)   
   local a0 = 0.35875
   local a1 = 0.48829
   local a2 = 0.14128
   local a3 = 0.01168
   local out = torch.zeros(N)
   local odata = torch.data(out)
   local cos = math.cos
   if flag == 'periodic' then
      local c1 = 2 * math.pi / N
      local c2 = 4 * math.pi / N
      local c3 = 6 * math.pi / N
      for i=0,N-1 do
	 odata[i] = a0 - a1*cos(c1*i) + a2*cos(c2*i) - a3*cos(c3*i)
      end      
   else
      local c1 = 2 * math.pi / (N-1)
      local c2 = 4 * math.pi / (N-1)
      local c3 = 6 * math.pi / (N-1)
      for i=0,N-1 do
	 odata[i] = a0 - a1*cos(c1*i) + a2*cos(c2*i) - a3*cos(c3*i)
      end
   end
   return out
end


local function genCQTkernel(fmax, bins,fs, q, atomHopFactor, thresh, winFlag)
   atomHopFactor = atomHopFactor or 0.25
   q = q or 1
   thresh = thresh or 0.0005
   winFlag = winFlag or 'sqrt_blackmanharris'
   local fmin = (fmax/2)*math.pow(2,(1/bins))
   local Q = 1/(math.pow(2,(1/bins))-1);
   Q = Q*q;
   -- length of the largest atom [samples]
   local Nk_max = math.round(Q * fs / fmin) 
   -- length of the shortest atom [samples]
   local Nk_min = math.round(Q * fs / (fmin * math.pow(2,((bins-1)/bins))))
    local atomHOP = math.round(Nk_min*atomHopFactor) -- atom hop size

   -- first possible center position within the frame
   local first_center = math.ceil(Nk_max/2) 

   -- lock the first center to an integer multiple of the atom hop size
   first_center = atomHOP * math.ceil(first_center/atomHOP) 

   -- use smallest possible FFT size (increase sparsity)
   local FFTLen = math.pow(2, math.nextpow2(first_center + math.ceil(Nk_max/2))) 

   -- number of temporal atoms per FFT Frame
   local winNr = math.floor((FFTLen-math.ceil(Nk_max/2)-first_center)/atomHOP)+1 

   local last_center = first_center + (winNr-1)*atomHOP;
   local fftHOP = (last_center + atomHOP) - first_center; -- hop size of FFT frames
   local fftOLP = (FFTLen-fftHOP/FFTLen)*100; -- overlap of FFT frames in percent
   
   local tempKernel= torch.zeros(FFTLen,2)
   local sparKernel = nil
   -- Compute kernel
   local atomInd = 0
   for k=1,bins do
      -- N[k] = (fs/fk)*Q. Rounding will be omitted in future versions
      local Nk = math.round( Q * fs / (fmin*math.pow(2,((k-1)/bins))) )
      -- get a window filter
      local winFct
      if winFlag == 'hann' then
	 winFct = signal.hann(Nk, 'periodic')	 
      elseif winFlag == 'sqrt_hann' then
	 winFct = torch.sqrt(signal.hann(Nk, 'periodic'))
      elseif winFlag == 'blackman' then
	 winFct = signal.blackman(Nk, 'periodic')
      elseif winFlag == 'sqrt_blackman' then
	 winFct = torch.sqrt(signal.blackman(Nk, 'periodic'))
      elseif winFlag == 'blackmanharris' then
	 winFct = signal.blackmanharris(Nk, 'periodic')
      else --  winFlag == 'sqrt_blackmanharris'
	 winFct = torch.sqrt(signal.blackmanharris(Nk, 'periodic'))
      end
      local fk = fmin*math.pow(2,((k-1)/bins))
      local tmp = torch.zeros(Nk, 2)
      tmp[{{},2}] = torch.range(0,Nk-1):mul((fk * 2 * math.pi / fs))
      local tempKernelBin = complex.exp(tmp)
      tempKernelBin = complex.cmul(tempKernelBin, (winFct/Nk));
      local atomOffset = first_center - math.ceil(Nk/2);
      for i = 1,winNr do
      	 local shift = atomOffset + ((i-1) * atomHOP);
      	 tempKernel[{{1+shift,Nk+shift},{}}] = tempKernelBin;
      	 atomInd = atomInd+1;
      	 local specKernel= signal.fft(tempKernel);
	 local thresSelector = complex.abs(specKernel):le(thresh)
	 thresSelector = torch.repeatTensor(thresSelector, 2, 1):t()
      	 specKernel[thresSelector] = 0
	 -- append this to the global kernel
	 if not sparKernel then 
	    sparKernel = torch.zeros(winNr * bins, 
				     specKernel:size(1), specKernel:size(2)) 
	 end
	 sparKernel[{i+(k-1)*winNr,{},{}}] = specKernel
      	 tempKernel = tempKernel:zero() --reset window     
      end 
   end
   sparKernel = sparKernel:transpose(1,2) / FFTLen

   -- Normalize the magnitudes of the atoms
   local sparMag = complex.abs(sparKernel)
   local _,wx1 = torch.max(sparMag[{{},1}],1)
   local _,wx2 = torch.max(sparMag[{{},sparMag:size(2)}],1)
   wx1 = wx1[1]
   wx2 = wx2[1]
   local wK=sparMag[{{wx1,wx2},{}}]
   wK = torch.diag(torch.mm(wK,wK:t()))
   wK = wK[{{math.round(1/q)+1,(wK:size(1)-math.round(1/q)-2)}}]
   local weight = 1/torch.mean(torch.abs(wK));
   weight = weight*(fftHOP/FFTLen); 
   weight = math.sqrt(weight); -- sqrt because the same weight is applied in icqt again
   sparKernel = sparKernel * weight

   --[[
      -- For debugging
      print(sparKernel)
      sparKernel = sparKernel:transpose(1,2)
      for i=1,sparKernel[3]:size(1) do
      print(sparKernel[3][i][1] .. ' ' .. sparKernel[3][i][2] .. 'i')
      end
   ]]--

   return {fKernel = sparKernel, fftLEN = FFTLen, fftHOP = fftHOP, fftOverlap = fftOLP, 
   bins=bins, firstcenter = first_center, atomHOP = atomHOP, atomNr = winNr, Nk_max = Nk_max, Q = Q, fmin = fmin}
end
signal.genCQTkernel = genCQTkernel

local function cqtcell2sparse(Xcq,octaves,bins,firstcenter,atomHOP,atomNr)
   local emptyHops = firstcenter/atomHOP;
   local drop = emptyHops*math.pow(2,(octaves-1))-emptyHops; -- distance between first value in highest octave and first value in lowest octave
   local spCQT = torch.zeros(bins*octaves,Xcq[1]:size(2)*atomNr-drop, 2);

   for i=1,octaves do
      drop = emptyHops*math.pow(2,(octaves-i))-emptyHops; -- first coefficients of all octaves have to be in synchrony
      local X = Xcq[i]; 
      if  atomNr > 1 then -- more than one atom per bin --> reshape
	 local Xoct = torch.zeros(bins,atomNr*X:size(2)-drop,2);
	 for u=1,bins do -- reshape to continous windows for each bin (for the case of several wins per frame)
	    local octX_bin = X[{{(u-1)*atomNr+1,u*atomNr},{},{}}];
	    local Xcont = torch.reshape(octX_bin,octX_bin:size(1)*octX_bin:size(2),2);
	    Xoct[{u,{},{}}] = Xcont[{{1+drop,Xcont:size(1)},{}}];
	 end
	 X = Xoct;
      else
	 X = X[{{},{1+drop,X:size(2)}}];
      end
      local binVec = torch.range(bins*octaves-bins*i+1,bins*octaves-bins*(i-1));
      -- spCQT[{{binVec},{1,math.pow(2,(i-1):size(X,2)*2^(i-1)}}] = X;

   end   
   return spCQT
end

--[[
   Constant-Q transform
   x - 1D input Tensor
   fmin - lowest frequency of interest
   fmax - highest frequency of interest
   bins frequency bins per octave
   fs - sampling rate
]]--
function signal.incomplete.cqt(x, fmin, fmax, bins, fs, atomHopFactor, q, 
		   thres, winFlag)
   atomHopFactor = atomHopFactor or 0.25
   q = q or 1
   thresh = thresh or 0.0005
   winFlag = winFlag or 'sqrt_blackmanharris'

   local octaveNr = math.ceil(math.log(fmax/fmin)/math.log(2));
   fmin = (fmax/math.pow(2,octaveNr)) * math.pow(2,(1/bins)); -- set fmin to actual value
   local xlen_init = x:size(1);

   -- anti-aliasing butterworth filter
   -- if (not B) or (not A) then
   --    local LPorder = 6
   --    local cutoff = 0.5
   --    B,A = butter(LPorder, cutoff, 'low') -- design f_nyquist/2-lowpass filter
   -- end
  
   -- calculate CQT
   --[[
      Generate a cqt kernel
      For each octave,
        Calculate STFT on input
        Apply Kernel on input
        Apply anti-aliasing filter on input
        Halve the sample rate of input
   ]]--
   local cqtKernel = genCQTkernel(fmax, bins,fs, q, atomHopFactor, thresh, winFlag);
   local maxBlock = cqtKernel.fftLEN * math.pow(2,(octaveNr-1)) -- largest FFT Block (virtual)
   local suffixZeros = maxBlock; 
   local prefixZeros = maxBlock;
   local xpadded = torch.zeros(prefixZeros + suffixZeros + x:size(1))
   xpadded[{{prefixZeros+1,prefixZeros+x:size(1)}}] = x
   x = xpadded
   local OVRLP = cqtKernel.fftLEN - cqtKernel.fftHOP
   local K = cqtKernel.fKernel:transpose(1,2) 
   local cellCQT = {}
   for i = 1,octaveNr do
      -- local xx = x:unfold(1, cqtKernel.fftLEN, cqtKernel.fftHOP)
      -- local XX = signal.fft(xx) -- applying fft to each column (each FFT frame)

      local XX = signal.stft(x, cqtKernel.fftLEN, cqtKernel.fftHOP, 'rect')
      -- calculating cqt coefficients for all FFT frames for this octave
      table.insert(cellCQT, complex.mm(K,XX:transpose(1,2))) 
      
      if i ~= octaveNr then
	 -- Anti-aliasing not implemented
	 -- x = filtfilt(B,A,x); %anti aliasing filter

	 -- hacky way of dropping sample rate by 2
	 local xs = x:size(1)
	 if xs % 2 == 1 then x = x[{{1,x:size(1)-1}}] end
	 local tmp = torch.reshape(x, x:size(1)/2, 2)
	 x = tmp[{{},1}]:contiguous()
      end
   end

   -- local spCQT = cqtcell2sparse(cellCQT, octaveNr, bins, cqtKernel.firstcenter, cqtKernel.atomHOP, cqtKernel.atomNr);
   
   local intParam = {}
   intParam.sufZeros = suffixZeros
   intParam.preZeros = prefixZeros
   intParam.xlen_init = xlen_init
   intParam.fftLEN = cqtKernel.fftLEN
   intParam.fftHOP = cqtKernel.fftHOP
   intParam.q = q
   intParam.filtCoeffA = A
   intParam.filtConeffB = B
   intParam.firstcenter = cqtKernel.firstcenter
   intParam.atomHOP = cqtKernel.atomHOP
   intParam.atomNr = cqtKernel.atomNr
   intParam.Nk_max = cqtKernel.Nk_max
   intParam.Q = cqtKernel.Q

   local Xcqt = {}
   Xcqt['spCQT'] = spCQT
   Xcqt['fKernel'] = cqtKernel.fKernel
   Xcqt['fmax'] = fmax
   Xcqt['fmin'] = fmin
   Xcqt['octaveNr'] = octaveNr
   Xcqt['bins'] = cqtKernel.bins
   Xcqt['intParams'] = intParam
   
   return Xcqt
end


return signal