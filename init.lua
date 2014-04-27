require 'torch'

local signal = signal or {}

local fft = require 'signal.fft'
-- fft functions go into the root signal namespace
for k,v in pairs(fft) do
   signal[k] = v
end



return signal
