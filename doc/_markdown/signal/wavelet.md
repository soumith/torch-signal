
<a class="entityLink" href="https://github.com/soumith/torch-signal/blob/272138111678b6da3f3a873204b4cb14aaef2e37/wavelet.lua#L9">[src]</a>
<a name="signal.wavelet.haar1d"></a>


### signal.wavelet.haar1d() ###

Haar wavelet (1D)
   return the phi and psi functions of a haar wavelet

<a class="entityLink" href="https://github.com/soumith/torch-signal/blob/272138111678b6da3f3a873204b4cb14aaef2e37/wavelet.lua#L29">[src]</a>
<a name="signal.wavelet.daubechies1d"></a>


### signal.wavelet.daubechies1d() ###

 
   Daubechies wavelet (1D)
   return the phi and psi functions of a daubechies wavelet


<a class="entityLink" href="https://github.com/soumith/torch-signal/blob/272138111678b6da3f3a873204b4cb14aaef2e37/wavelet.lua#L67">[src]</a>
<a name="signal.wavelet.dwt1d"></a>


### signal.wavelet.dwt1d(input, phi, psi, maxlevels) ###

Calculates the discrete wavelet transform, given the phi and psi functions
   phi and psi are functions that take the input signal and give out the 
   scaled signal, and the wavelet coefficients respectively.

   input - input signal
   \phi φ(x) - scaling function
   \psi ψ(x) - wavelet function
   [maxlevels] - maximum number of levels to recurse
