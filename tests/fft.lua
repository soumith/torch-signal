signal = require 'signal'

function sfft()
   print('hello')
   -- local input = torch.Tensor({{1,0},{2,0},{3,0},{4,0}})
   -- local output = signal.fft(input)
   local input = torch.Tensor({1,2,3,4})
   local output = signal.dct(input)
   print('world')
end


while true do
   sfft()
   collectgarbage()
   sfft()
   collectgarbage()
end