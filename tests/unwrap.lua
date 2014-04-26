signal = require 'signal'

a=torch.FloatTensor({12,2,9,16})
b=signal.unwrap(a)
print(b)

a=torch.Tensor({12,4,9,16})
b=signal.unwrap(a)
print(b)

a=torch.Tensor({12,8,9,16})
b=signal.unwrap(a)
print(b)

a=torch.Tensor({12,22,9,16})
b=signal.unwrap(a)
print(b)

a=torch.Tensor({12,20,9,16})
b=signal.unwrap(a)
print(b)

a=torch.Tensor({12,18,9,16})
b=signal.unwrap(a)
print(b)
