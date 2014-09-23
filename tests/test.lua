local signal = require 'signal'

local mytester = torch.Tester()
local precision = 1e-5

local signaltest = {}
-- local assertcount = 0

local function asserteq(a,b, err, precision)
   precision = precision or 1e-5
   local c = a-b
   mytester:assertlt(c, precision, err)
end

function signaltest.fft()
   do
      local input = torch.Tensor({{1,0},{2,0},{3,0},{4,0}})
      local output = signal.fft(input)
      local inputi = signal.ifft(output)
      for i=1,input:size(1) do
	 for j=1,input:size(2) do
	    asserteq(input[i][j], inputi[i][j], 'error in fft+ifft')
	 end
      end   
   end
   do
      local input = torch.randn(1000,2)
      local output = signal.fft(input)
      local inputi = signal.ifft(output)
      for i=1,input:size(1) do
	 asserteq(input[i][1], inputi[i][1], 'error in fft+ifft')
	 asserteq(input[i][2], inputi[i][2], 'error in fft+ifft')
      end
   end
end

function signaltest.fft2()
   do
      local input = torch.Tensor(2,2,2):fill(0)
      input[1][1][1] = 1
      input[1][2][1] = 2
      input[2][1][1] = 3
      input[2][2][1] = 4
      local output = signal.fft2(input)
      local inputi = signal.ifft2(output)
      for i=1,input:size(1) do
	 for j=1,input:size(2) do
	    for k=1,input:size(3) do
	       asserteq(input[i][j][k], inputi[i][j][k], 'error in fft2+ifft2')
	    end
	 end
      end
   end
   do
      local input = torch.randn(50,100,2)
      local output = signal.fft2(input)
      local inputi = signal.ifft2(output)
      for i=1,input:size(1) do
	 for j=1,input:size(2) do
	    asserteq(input[i][j][1], inputi[i][j][1], 'error in fft2+ifft2')
	    asserteq(input[i][j][2], inputi[i][j][2], 'error in fft2+ifft2')
	 end
      end
   end
end

function signaltest.rfft()
   local input = torch.randn(100)
   local output1 = signal.fft(input)
   local output2 = signal.rfft(input)
   for i=1,output1:size(1)/2+1 do
      asserteq(output1[i][1], output2[i][1], 'error in rfft')
      asserteq(output1[i][2], output2[i][2], 'error in rfft')
   end
end

function signaltest.rfft2()
   local input = torch.randn(100, 50)
   local output1 = signal.fft2(input)
   local output2 = signal.rfft2(input)
   for i=1,output1:size(1) do
      for j=1,output1:size(2)/2+1 do
	 asserteq(output1[i][j][1], output2[i][j][1], 'error in rfft2', 1e-4)
	 asserteq(output1[i][j][2], output2[i][j][2], 'error in rfft2', 1e-4)
      end
   end
end

function signaltest.irfft()
   local input = torch.randn(100)
   local output1 = signal.fft(input)
   local outputi1 = signal.ifft(output1)
   local output2 = signal.rfft(input)
   local output2_copy = output2:clone()
   local outputi2 = signal.irfft(output2)
   for i=1,outputi1:size(1) do
      asserteq(outputi1[i][1], outputi2[i], 'error in irfft')
   end
   for i=1,output2:size(1) do
      for j=1,output2:size(2) do
         asserteq(output2[i][j], output2_copy[i][j], 'error in preserving of irfft input')
      end
   end
end

function signaltest.irfft2()
   local input = torch.randn(100, 50)
   local output1 = signal.fft2(input)
   local outputi1 = signal.ifft2(output1)
   local output2 = signal.rfft2(input)
   local output2_copy = output2:clone()
   local outputi2 = signal.irfft2(output2)
   for i=1,outputi1:size(1) do
      for j=1,outputi1:size(2) do
	 asserteq(outputi1[i][j][1], outputi2[i][j], 'error in irfft2')
      end
   end
   for i=1,output2:size(1) do
      for j=1,output2:size(2) do
         for k=1,output2:size(3) do
             asserteq(output2[i][j][k], output2_copy[i][j][k], 'error in preserving of irfft2 input')
         end
      end
   end
end

function signaltest.dct()
   local inp=torch.randn(10000)

   local out=signal.dct(inp)
   local inpi = signal.idct(out)
   for i=1,inp:size(1) do
      asserteq(inp[i], inpi[i], 'error in dct')
   end
end

function signaltest.dct2()
   local inp=torch.randn(50,100)

   local out=signal.dct2(inp)
   local inpi = signal.idct2(out)
   for i=1,inp:size(1) do
      for j=1,inp:size(2) do
	 asserteq(inp[i][j], inpi[i][j], 'error in dct2')
      end
   end
end

function signaltest.dct3()
   local inp=torch.randn(30,20,10)

   local out=signal.dct3(inp)
   local inpi = signal.idct3(out)
   for i=1,inp:size(1) do
      for j=1,inp:size(2) do
	 for k=1,inp:size(3) do
	    asserteq(inp[i][j][k], inpi[i][j][k], 'error in dct3')
	 end
      end
   end
end

function signaltest.rceps()
   local input =torch.Tensor({12,2,9,16})
   local out = signal.rceps(input)
   asserteq(out[1], 2.52129598, 'error in rceps', 1e-5)
   asserteq(out[2], 0.64123734, 'error in rceps', 1e-5)
   asserteq(out[3], -0.14020901, 'error in rceps', 1e-5)
   asserteq(out[4], 0.64123734, 'error in rceps', 1e-5)
end

function signaltest.hilbert()
   -- even
   local a=torch.Tensor({12,2,9,16})
   local b = signal.hilbert(a)
   -- 12.0000 + 7.0000i   2.0000 + 1.5000i   9.0000 - 7.0000i  16.0000 - 1.5000i
   asserteq(b[1][1] , 12.00, 'error in hilbert', 1e-5)
   asserteq(b[1][2] ,  7.00, 'error in hilbert', 1e-5)
   asserteq(b[2][1] ,  2.00, 'error in hilbert', 1e-5)
   asserteq(b[2][2] ,  1.50, 'error in hilbert', 1e-5)
   asserteq(b[3][1] ,  9.00, 'error in hilbert', 1e-5)
   asserteq(b[3][2] , -7.00, 'error in hilbert', 1e-5)
   asserteq(b[4][1] , 16.00, 'error in hilbert', 1e-5)
   asserteq(b[4][2] , -1.50, 'error in hilbert', 1e-5)

   -- odd
   local a=torch.Tensor({12,2,9,16,25})
   local b=signal.hilbert(a)
   -- 12.0000 +13.1402i   2.0000 + 0.5388i   9.0000 - 6.7285i  16.0000 - 8.3955i  25.0000 + 1.4450i
   asserteq(b[1][1] , 12.00,   'error in hilbert', 1e-4)
   asserteq(b[1][2] , 13.1402, 'error in hilbert', 1e-4)
   asserteq(b[2][1] , 2.00,    'error in hilbert', 1e-4)
   asserteq(b[2][2] , 0.5388,  'error in hilbert', 1e-4)
   asserteq(b[3][1] , 9.00,    'error in hilbert', 1e-4)
   asserteq(b[3][2] , -6.7285, 'error in hilbert', 1e-4)
   asserteq(b[4][1] , 16.00,   'error in hilbert', 1e-4)
   asserteq(b[4][2] , -8.3955, 'error in hilbert', 1e-4)
   asserteq(b[5][1] , 25.00,   'error in hilbert', 1e-4)
   asserteq(b[5][2] , 1.445,   'error in hilbert', 1e-4)
end

function signaltest.cceps()
   local a=torch.Tensor({12,2,9,16})
   local b,nd=signal.cceps(a)
   local c=signal.icceps(b,nd)
   -- Expected: 2.5213   -0.0386   -0.1402    1.3211
   asserteq(b[1] , 2.5213,   'error in cceps+icceps', 1e-4)
   asserteq(b[2] , -0.0386,   'error in cceps+icceps', 1e-4)
   asserteq(b[3] , -0.1402,   'error in cceps+icceps', 1e-4)
   asserteq(b[4] , 1.3211,   'error in cceps+icceps', 1e-4)

   for i=1,a:size(1) do
      asserteq(a[i] , c[i],   'error in cceps+icceps')
   end

   local a=torch.randn(1000)
   local b,nd=signal.cceps(a)
   local c=signal.icceps(b,nd)
   for i=1,a:size(1) do
      asserteq(a[i] , c[i],   'error in cceps+icceps')
   end
end


mytester:add(signaltest)

print('Running tests at float precision')
torch.setdefaulttensortype('torch.FloatTensor')
mytester:run()

print('Running tests at double precision')
torch.setdefaulttensortype('torch.DoubleTensor')
mytester:run()

