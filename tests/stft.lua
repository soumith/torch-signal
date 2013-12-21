require 'audio'
local signal = require 'signal'
require 'image'

inp = audio.samplevoice()
print(#(inp[1]))
stft = signal.stft(inp[1], 1024, 512, 'hamming')
stft = signal.stft(inp[1], 1024, 512, 'hann')
stft = signal.stft(inp[1], 1024, 512, 'bartlett')
stft = signal.stft(inp[1], 1024, 512)

stft2 = audio.stft(inp, 1024, 'rect', 512)
print(#stft)
-- display magnitude
image.display(stft[{{1,100},{1,100},1}])
image.display(stft2[{{1,100},{stft2:size(2)-100,stft2:size(2)},1}])

spect =  signal.spectrogram(inp[1], 1024, 512)
image.display(spect)