signal = require 'signal'

a=torch.Tensor({12,2,9,16})


b=torch.poly(a)
print(a)
print(b)
print('-- Expected: 1         -39         518       -2616        3456')

a=torch.Tensor({{12,2,9,16},{12,2,9,16},{12,2,9,16},{12,2,9,16}})
b=torch.poly(a)
print(a)
print(b)
print('-- Expected: 1.0000  -39.0000    0.0000         0         0')
