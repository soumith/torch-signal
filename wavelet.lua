local signal = require 'signal.convolution'
local xmath = require 'signal.extramath'

local wavelet = {}

--[[ 
   Haar wavelet (1D)
   return the phi and psi functions of a haar wavelet
]]--
function wavelet.haar1d()
   -- phi would be convolution with a 1d kernel [1/2,1/2] and stride 2
   -- psi would be convolution with a 1d kernel [1/2,-1/2] and stride 2
   local phi_kernel = torch.Tensor({0.5,0.5})
   local psi_kernel = torch.Tensor({0.5,-0.5})
   
   local function phi(input)
      return signal.conv1d(input, phi_kernel:typeAs(input), 2)
   end
   local function psi(input)
      return signal.conv1d(input, psi_kernel:typeAs(input), 2)
   end
   return phi, psi
end

--[[ 
   Daubechies wavelet (1D)
   return the phi and psi functions of a daubechies wavelet
]]
function wavelet.daubechies1d()
   error('implementation not finished')
   -- calculate kernel
   -- phi(r) =  ho * phi(2r)    + h1 * phi(2r- 1) + h2 * phi(2r - 2) + h3 * phi(2r - 3)
   -- psi(r) = -ho * phi(2r- 1) + h1 * phi(2r)    - h2 * phi(2r + 1) + h3 * phi(2r + 2)
   local function phi(input)
   end

   local function psi(input)
   end
   return phi, psi
end


--[[
   pads a given function to the nearest upper bound power of 2.
   fills the padding with zeros
]]--
local function pad2(input)
   local input_size = input:size(1)
   local log2 = xmath.log2(input_size)
   if log2 % 2 ~= 0 then
      local padded_size = math.pow(2, math.ceil(log2))
      local temp = torch.zeros(padded_size):typeAs(input)
      temp[{{1, input_size}}] = input
      input = temp
   end
   return input
end

--[[
   Calculates the discrete wavelet transform, given the phi and psi functions
   phi and psi are functions that take the input signal and give out the 
   scaled signal, and the wavelet coefficients respectively.

   input - input signal
   \phi φ(x) - scaling function
   \psi ψ(x) - wavelet function
   [maxlevels] - maximum number of levels to recurse
]]--
function wavelet.dwt1d(input, phi, psi, maxlevels)
   -- pad the input to 2^n, fill with zeros
   input = pad2(input)

   -- number of levels is log2(n)
   local level = level or xmath.log2(input:size(1))
   
   local output = output or {}

   for lev = level, 1,-1 do
      if maxlevels and (level-lev) == maxlevels then break; end
      -- calculate wavelet coefficients with psi
      table.insert(output , psi(input))
      -- scale the input
      input = phi(input)  
   end
   
   return output, input
end


return wavelet
