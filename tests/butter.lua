signal = require 'signal'

b=signal.butter(6, 0.5, 'low')
print(b)
-- Expected: 