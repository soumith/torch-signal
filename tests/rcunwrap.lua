signal = require 'signal'

a=torch.Tensor({12,2,9,16})

b=signal.rcunwrap(a)
print(b)

-- Expected: 12.0000    6.7124   -0.4248   -7.5619