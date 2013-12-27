signal = require 'signal'

-- init values for CQT
fs = 44100;
bins_per_octave = 24;
fmax = fs/3;     -- center frequency of the highest frequency bin 
fmin = fmax/512; -- lower boundary for CQT (lowest frequency bin will be immediately above this): fmax/<power of two> 

--  generate/read input signal
x = torch.randn(1*fs,1);
-- Drop frequencies outside [fmin fmax] to allow calculating 
--  the SNR after inverse transform
xpadded = torch.zeros(1000 + x:size(1))
xpadded[{{500+1,500+x:size(1)}}] = x
x = xpadded
w1 = 2*(fmin/(fs/2)); 
w2 = 0.8*(fmax/(fs/2));
-- [B,A] = butter(6,[w1 w2]); x = filtfilt(B,A,x); 

-- CQT
Xcqt = signal.cqt(x,fmin,fmax,bins_per_octave,fs);

-- ***computing cqt with optional input parameters***********
-- Xcqt = cqt(x,fmin,fmax,bins_per_octave,fs,'q',1,'atomHopFactor',0.25,'thresh',0.0005,'win','sqrt_blackmanharris');