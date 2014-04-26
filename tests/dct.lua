signal = require 'signal'

a=torch.FloatTensor({12,2,9,16})

b=signal.dct(a)
c=signal.idct(b)
print(a)
print(b)
print(c)

a=torch.Tensor({{12,2,9,16},{12,2,9,16},{12,2,9,16}})
b=signal.dct2(a)
c=signal.idct2(b)
print(a)
print(b)
print(c)

a=torch.FloatTensor({{{12,2,9,16},{12,2,9,16},{12,2,9,16}},{{12,2,9,16},{12,2,9,16},{12,2,9,16}}})
print(a)
b=signal.dct3(a)
c=signal.idct3(b)
print(b)
print(c)



