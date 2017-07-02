require 'xlua'
local signal = require 'signal'

function test_rfftM(A)
   local Af = signal.rfftM(A)
   for i = 1, A:size(1) do
      assert((Af[i] - signal.rfft(A[i])):norm() == 0)
   end
end

function test_irfftM(A)
   A[{{}, {1}, {2}}] = 0 -- 0th order, always real
   if A:size(2) % 2 == 1 then
      A[{{}, {-1}, {2}}] = 0 -- nyquist freq is always real
   end
   local Aif = signal.irfftM(A)
   for i = 1, A:size(1) do
      assert((Aif[i] - signal.irfft(A[i])):norm() == 0)
   end
end

-- test forward transform, both odd and even nfft
test_rfftM(torch.randn(100, 100))
test_rfftM(torch.randn(100, 101))
print('rfftM OK')

-- test inverse transform, both odd and even nfft
test_irfftM(torch.randn(100, 100, 2))
test_irfftM(torch.randn(100, 101, 2))
print('irfftM OK')

-- evaluate runtime improvements
local A = torch.randn(100, 100)
local p = xlua.Profiler()
local iters = 1000

p:start('single rfftM')
for i = 1, iters do
   signal.rfftM(A)
end
p:lap('single rfftM', iters)

p:start('multiple rfft')
for i = 1, iters do
   for i = 1, A:size(1) do
      signal.rfft(A[i])
   end
end
p:lap('multiple rfft', iters)

-- This also offers a significant speedup over the
-- signal.stft implementation, with some care.

-- Define a quick and dirty stft using rfftM
-- We're going to use a rectangular window.
-- It'd be easy enough to allocate a window and
-- and cmul the framed input windows.
local function stft(input, window_size, window_stride)
   A = input:unfold(1, window_stride, window_size)
   return signal.rfftM(A)
end

local samples = torch.randn(16000) -- 1 s long clip @ 16 kHz
local window_size = 400 -- emulate 25 ms @ 16 kHz
local window_stride = 160 -- emulate 10 ms @ 16 kHz
p:start('rstft using rfftM')
for i = 1, iters do
  stft(samples, window_size, window_stride)
end
p:lap('rstft using rfftM')

p:start('signal rstft')
for i = 1, iters do
  signal.rstft(samples, window_size, window_stride)
end
p:lap('signal rstft')

p:printAll()
