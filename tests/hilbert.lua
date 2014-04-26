signal = require 'signal'

a=torch.FloatTensor({12,2,9,16})

b=signal.hilbert(a)
print(a)
print(b)

print('expected -- 12.0000 + 7.0000i   2.0000 + 1.5000i   9.0000 - 7.0000i  16.0000 - 1.5000i')

a=torch.Tensor({12,2,9,16,25})

b=signal.hilbert(a)
print(a)
print(b)

print('expected -- 12.0000 +13.1402i   2.0000 + 0.5388i   9.0000 - 6.7285i  16.0000 - 8.3955i  25.0000 + 1.4450i')
