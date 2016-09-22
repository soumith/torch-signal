require 'xlua'
local signal = require 'signal'

function test_fftM(A)
    local Af = signal.fftM(A)
    local Aif = signal.ifftM(A)
    for i=1,A:size(1) do
        assert((Af[i] - signal.fft(A[i])):norm() == 0)
        assert((Aif[i] - signal.ifft(A[i])):norm() == 0)
    end
end

-- test real tensors
test_fftM(torch.randn(100,100))
print("Real tensors OK")

-- test complex tensors
test_fftM(torch.randn(100,100,2))
print("Complex tensors OK")

-- evaluate runtime improvements
local A = torch.randn(100,100)
local p = xlua.Profiler()
local iters = 1000

p:start('single fftM')
for i=1,iters do
    signal.fftM(A)
end
p:lap('single fftM', iters)

p:start('multiple fft')
for i=1,iters do
    for i=1,A:size(1) do
        signal.fft(A[i])
    end
end
p:lap('multiple fft', iters)

p:printAll()
