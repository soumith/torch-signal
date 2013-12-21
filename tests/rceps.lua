signal = require 'signal'

a=torch.Tensor({12,2,9,16})

b=signal.rceps(a)
print(b)

-- Expected:  2.5213    0.6412   -0.1402    0.6412