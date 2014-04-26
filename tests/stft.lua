require 'audio'
local signal = require 'signal'
require 'gfx.js'

torch.setdefaulttensortype('torch.FloatTensor')

inp = audio.samplevoice():float()
print(#(inp[1]))
stft = signal.stft(inp[1], 1024, 512, 'hamming')
stft = signal.stft(inp[1], 1024, 512)
stft = signal.stft(inp[1], 1024, 512, 'bartlett')
a=os.clock()
stft = signal.stft(inp[1], 1024, 512, 'hann')
print('Time taken for stft from signal package: ' .. os.clock()-a)
a=os.clock()
stft2 = audio.stft(inp, 1024, 'hann', 512)
print('Time taken for stft from audio package: ' .. os.clock()-a)
print(#stft)
-- display magnitude
gfx.image(stft[{{1,100},{1,100},1}])
gfx.image(stft2[{{1,100},{stft2:size(2)-100,stft2:size(2)},1}])

spect =  signal.spectrogram(inp[1], 1024, 512)
gfx.image(spect)
