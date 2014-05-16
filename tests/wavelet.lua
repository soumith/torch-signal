local signal = require 'signal'

torch.manualSeed(1)

phi,psi = signal.wavelet.haar1d()

-- input = torch.Tensor({3,1,0,4,8,6,9,9})

-- print(input)
-- local output = signal.wavelet.dwt1d(input, phi, psi)
-- print(output)


input = torch.Tensor({32.0, 10.0, 20.0, 38.0, 37.0, 28.0, 38.0, 34.0, 18.0, 24.0, 
		      18.0, 9.0, 23.0, 24.0, 28.0, 34.0}):float()
local wavelet, scale = signal.wavelet.dwt1d(input, phi, psi)
for k,v in ipairs(wavelet) do
   print(v)
end
print(wavelet)
print(scale)
