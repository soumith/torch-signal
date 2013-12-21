signal = require 'signal'

a=torch.Tensor({12,2,9,16})

b=signal.rcwrap(a, 0.5)
print(b)
