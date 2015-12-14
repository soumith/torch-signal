local signal = {}
local C = require 'signal.ffi'

--[[
   1D valid convolution with stride
]]--
function signal.conv1d(input, kernel, stride, mode)
  kernel = kernel:typeAs(input)
  local input_data = input:data()
  local kernel_data = kernel:data()

  local input_size = input:size(1)
  local kernel_size = kernel:size(1)

  mode = mode or 'valid'
  local output_size = 0
  if mode == 'valid' then
    output_size = math.floor((input_size - kernel_size + stride)/stride)
  elseif mode == 'same' then
    output_size = math.floor(math.max(input_size/stride, kernel_size/stride))
  else
    error("Supported modes are 'valid' or 'same'")
  end
  local output = torch.zeros(output_size):typeAs(input)
  local output_data = output:data()

  if input:type() == 'torch.FloatTensor' then
    C['signal_conv1d_float'](output_data, input_data, kernel_data, output_size, kernel_size, stride)
  else
    C['signal_conv1d_double'](output_data, input_data, kernel_data, output_size, kernel_size, stride)
  end
  return output
end

return signal
