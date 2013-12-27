signal = require 'signal'


fs = 44100;
bins_per_octave = 24;
fmax = fs/3;     -- center frequency of the highest frequency bin 
fmin = fmax/512; -- lower boundary for CQT (lowest frequency bin will be immediately above this): fmax/<power of two> 

-- genCQTkernel(fmax, bins,fs, q, atomHopFactor, thresh, winFlag)
print(fmax, bins_per_octave, fs)
a=signal.genCQTkernel(fmax, bins_per_octave, fs)
print(a)