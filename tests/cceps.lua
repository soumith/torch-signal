signal = require 'signal'

a=torch.Tensor({12,2,9,16})


b,nd=signal.cceps(a)
c=signal.icceps(b,nd)
print(a)
print(b)
print(c)

-- Expected: 2.5213   -0.0386   -0.1402    1.3211