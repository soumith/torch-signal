require 'signal'

local mytester = torch.Tester()
local precision = 1e-5

local ffttest = {}

function ffttest.fft()
   do
      local input = torch.Tensor({{1,0},{2,0},{3,0},{4,0}})
      local output = torch.fft(input)
      local inputi = torch.ifft(output)
      for i=1,input:size(1) do
	 for j=1,input:size(2) do
	    mytester:asserteq(input[i][j], inputi[i][j], 'error in fft+ifft')
	 end
      end   
   end
   do
      local input = torch.Tensor({1,2,3,4})
      local output = torch.fft(input)
      local inputi = torch.ifft(output)
      for i=1,input:size(1) do
	 mytester:asserteq(input[i], inputi[i][1], 'error in fft+ifft')
      end   
   end
end

function ffttest.fft2()
   do
      local input = torch.Tensor(2,2,2):fill(0)
      input[1][1][1] = 1
      input[1][2][1] = 2
      input[2][1][1] = 3
      input[2][2][1] = 4
      local output = torch.fft2(input)
      local inputi = torch.ifft2(output)
      for i=1,input:size(1) do
	 for j=1,input:size(2) do
	    for k=1,input:size(3) do
	       mytester:asserteq(input[i][j][k], inputi[i][j][k], 'error in fft2+ifft2')
	    end
	 end
      end
   end
   do
      local input = torch.Tensor({{1,2},{3,4}})
      local output = torch.fft2(input)
      local inputi = torch.ifft2(output)
      for i=1,input:size(1) do
	 for j=1,input:size(2) do
	    mytester:asserteq(input[i][j], inputi[i][j][1], 'error in fft2+ifft2')
	 end
      end
   end
end

mytester:add(ffttest)

mytester:run()