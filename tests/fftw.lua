local fftw = require 'fftw3'
local ffi = require 'ffi'
local NOT_USED_BUT_CRASHES_WITHOUT_THIS_LINE = ffi.load('fftw3')

function test()
   print('start')   
   local inp = ffi.new('double[?]', 100); local out = ffi.new('double[?]', 100)
   local plan = fftw.plan_r2r_1d(100, inp, out, 5, 64)
   fftw.destroy_plan(plan)
   print('end')
end

while true do
   test()
   collectgarbage()
end
