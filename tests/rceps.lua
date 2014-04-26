signal = require 'signal'

a=torch.FloatTensor({12,2,9,16})

b=signal.rceps(a)
print(b)

print('-- Expected:  2.5213    0.6412   -0.1402    0.6412')
