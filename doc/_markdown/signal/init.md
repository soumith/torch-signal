
<a class="entityLink" href="https://github.com/https://github.com/soumith/torch-signal//blob/c3ea4516bf3c6c17077c342b9010873d21f732d6/init.lua#L42">[src]</a>
<a name="signal.fft"></a>
### signal.fft(input) ###

1D FFT
   Takes Real inputs (1D tensor of N points)
   or complex inputs 2D tensor of (Nx2) size for N points
   
   Output matches with matlab output

<a class="entityLink" href="https://github.com/https://github.com/soumith/torch-signal//blob/c3ea4516bf3c6c17077c342b9010873d21f732d6/init.lua#L53">[src]</a>
<a name="signal.ifft"></a>
### signal.ifft(input) ###

inverse 1D FFT
   Takes Real inputs (1D tensor of N points)
   or complex inputs 2D tensor of (Nx2) size for N points

   Output matches with matlab output

<a class="entityLink" href="https://github.com/https://github.com/soumith/torch-signal//blob/c3ea4516bf3c6c17077c342b9010873d21f732d6/init.lua#L63">[src]</a>
<a name="signal.rfft"></a>
### signal.rfft(inp) ###

real to complex dft.
   This function retains only the positive frequencies.
   Input is a 1D real tensor
   Output is 2D complex tensor of size (input:size(1)/2 + 1, 2)

<a class="entityLink" href="https://github.com/https://github.com/soumith/torch-signal//blob/c3ea4516bf3c6c17077c342b9010873d21f732d6/init.lua#L87">[src]</a>
<a name="signal.irfft"></a>
### signal.irfft(inp) ###

complex to real dft. This function is the exact inverse of signal.rfft

<a class="entityLink" href="https://github.com/https://github.com/soumith/torch-signal//blob/c3ea4516bf3c6c17077c342b9010873d21f732d6/init.lua#L145">[src]</a>
<a name="signal.fft2"></a>
### signal.fft2(input) ###

2D FFT
   Takes Real inputs (2D tensor of NxM points)
   or complex inputs 3D tensor of (NxMx2) size for NxM points

   Output matches with matlab output

<a class="entityLink" href="https://github.com/https://github.com/soumith/torch-signal//blob/c3ea4516bf3c6c17077c342b9010873d21f732d6/init.lua#L156">[src]</a>
<a name="signal.ifft2"></a>
### signal.ifft2(input) ###

2D Inverse FFT
   Takes Real inputs (2D tensor of NxM points)
   or complex inputs 3D tensor of (NxMx2) size for NxM points

   Output matches with matlab output

<a class="entityLink" href="https://github.com/https://github.com/soumith/torch-signal//blob/c3ea4516bf3c6c17077c342b9010873d21f732d6/init.lua#L197">[src]</a>
<a name="signal.fft3"></a>
### signal.fft3(input) ###

3D FFT
   Takes Real inputs (3D tensor of NxMxP points)
   or complex inputs 4D tensor of (NxMxPx2) size for NxMxP points

   Output matches with matlab output

<a class="entityLink" href="https://github.com/https://github.com/soumith/torch-signal//blob/c3ea4516bf3c6c17077c342b9010873d21f732d6/init.lua#L208">[src]</a>
<a name="signal.ifft3"></a>
### signal.ifft3(input) ###

3D Inverse FFT
   Takes Real inputs (3D tensor of NxMxP points)
   or complex inputs 4D tensor of (NxMxPx2) size for NxMxP points

   Output matches with matlab output

<a class="entityLink" href="https://github.com/https://github.com/soumith/torch-signal//blob/c3ea4516bf3c6c17077c342b9010873d21f732d6/init.lua#L240">[src]</a>
<a name="signal.stft"></a>
### signal.stft(inp, window_size, window_stride, window_type) ###

1D short-time fourier transforms
   Run a window across your signal and calculate fourier transforms on that window.
   To make sure that the windows are not discontinuous at the edges, you can optionally apply a window preprocessor.
   The available window preprocessors are: hamming, hann, bartlett

<a class="entityLink" href="https://github.com/https://github.com/soumith/torch-signal//blob/c3ea4516bf3c6c17077c342b9010873d21f732d6/init.lua#L266">[src]</a>
<a name="signal.spectrogram"></a>
### signal.spectrogram(inp, window_size, window_stride) ###

Takes the stft and generates a pretty spectrogram by
   taking the magnitude of the stft, and applying a (natural log * 10)
   Also transposes the output, to have time on the X axis.

<a class="entityLink" href="https://github.com/https://github.com/soumith/torch-signal//blob/c3ea4516bf3c6c17077c342b9010873d21f732d6/init.lua#L288">[src]</a>
<a name="signal.unwrap"></a>
### signal.unwrap(a, tol) ###

Correct phase angles to produce smoother phase plots   
   Unwrap radian phases by adding multiples of 2*pi as appropriate to
   remove jumps greater than **tol**. **tol** defaults to pi.   

   Output matches with matlab output

<a class="entityLink" href="https://github.com/https://github.com/soumith/torch-signal//blob/c3ea4516bf3c6c17077c342b9010873d21f732d6/init.lua#L315">[src]</a>
<a name="signal.rcunwrap"></a>
### signal.rcunwrap(x) ###

unwraps the phase and removes phase corresponding to integer lag.

   Output matches with matlab output

<a class="entityLink" href="https://github.com/https://github.com/soumith/torch-signal//blob/c3ea4516bf3c6c17077c342b9010873d21f732d6/init.lua#L334">[src]</a>
<a name="signal.rcwrap"></a>
### signal.rcwrap(y, nd) ###

Adds phase corresponding to integer lag

   Output matches with matlab output

<a class="entityLink" href="https://github.com/https://github.com/soumith/torch-signal//blob/c3ea4516bf3c6c17077c342b9010873d21f732d6/init.lua#L358">[src]</a>
<a name="signal.cceps"></a>
### signal.cceps(x) ###

1D Complex cepstral analysis
   Returns the cepstrum and a phase shift factor "nd" that is useful to invert the signal back.

   Output matches with matlab output

<a class="entityLink" href="https://github.com/https://github.com/soumith/torch-signal//blob/c3ea4516bf3c6c17077c342b9010873d21f732d6/init.lua#L383">[src]</a>
<a name="signal.icceps"></a>
### signal.icceps(xhat,nd) ###

1D Inverse Complex cepstral analysis.
   Takes in the outputs of cceps to produce the input signal back

   Output matches with matlab output

<a class="entityLink" href="https://github.com/https://github.com/soumith/torch-signal//blob/c3ea4516bf3c6c17077c342b9010873d21f732d6/init.lua#L402">[src]</a>
<a name="signal.rceps"></a>
### signal.rceps(x) ###

Real cepstrum and minimum phase reconstruction
   The real cepstrum is the inverse Fourier transform of the real logarithm of the magnitude of the Fourier transform of a sequence.

   Output matches with matlab output

<a class="entityLink" href="https://github.com/https://github.com/soumith/torch-signal//blob/c3ea4516bf3c6c17077c342b9010873d21f732d6/init.lua#L446">[src]</a>
<a name="signal.dct"></a>
### signal.dct(input) ###

1D Discrete Cosine Transform (DCT)
   Takes Real inputs (1D tensor of N points)

   To see what is exactly computed, see section REDFT10 over here: 
   http://www.fftw.org/doc/1d-Real_002deven-DFTs-_0028DCTs_0029.html

<a class="entityLink" href="https://github.com/https://github.com/soumith/torch-signal//blob/c3ea4516bf3c6c17077c342b9010873d21f732d6/init.lua#L457">[src]</a>
<a name="signal.idct"></a>
### signal.idct(input) ###

inverse 1D Discrete Cosine Transform (DCT)
   Takes Real inputs (1D tensor of N points)

   To see what is exactly computed, see section REDFT01 over here: 
   http://www.fftw.org/doc/1d-Real_002deven-DFTs-_0028DCTs_0029.html

<a class="entityLink" href="https://github.com/https://github.com/soumith/torch-signal//blob/c3ea4516bf3c6c17077c342b9010873d21f732d6/init.lua#L496">[src]</a>
<a name="signal.dct2"></a>
### signal.dct2(input) ###

2D Discrete Cosine Transform (DCT)
   Takes Real inputs (2D tensor of NxM points)

   To see what is exactly computed, see section REDFT10 over here: 
   http://www.fftw.org/doc/1d-Real_002deven-DFTs-_0028DCTs_0029.html

<a class="entityLink" href="https://github.com/https://github.com/soumith/torch-signal//blob/c3ea4516bf3c6c17077c342b9010873d21f732d6/init.lua#L507">[src]</a>
<a name="signal.idct2"></a>
### signal.idct2(input) ###

inverse 2D Discrete Cosine Transform (DCT)
   Takes Real inputs (2D tensor of NxM points)

   To see what is exactly computed, see section REDFT01 over here: 
   http://www.fftw.org/doc/1d-Real_002deven-DFTs-_0028DCTs_0029.html

<a class="entityLink" href="https://github.com/https://github.com/soumith/torch-signal//blob/c3ea4516bf3c6c17077c342b9010873d21f732d6/init.lua#L548">[src]</a>
<a name="signal.dct3"></a>
### signal.dct3(input) ###

3D Discrete Cosine Transform (DCT)
   Takes Real inputs (3D tensor of NxMXP points)

   To see what is exactly computed, see section REDFT10 over here: 
   http://www.fftw.org/doc/1d-Real_002deven-DFTs-_0028DCTs_0029.html

<a class="entityLink" href="https://github.com/https://github.com/soumith/torch-signal//blob/c3ea4516bf3c6c17077c342b9010873d21f732d6/init.lua#L559">[src]</a>
<a name="signal.idct3"></a>
### signal.idct3(input) ###

inverse 3D Discrete Cosine Transform (DCT)
   Takes Real inputs (3D tensor of NxMxP points)

   To see what is exactly computed, see section REDFT01 over here: 
   http://www.fftw.org/doc/1d-Real_002deven-DFTs-_0028DCTs_0029.html

<a class="entityLink" href="https://github.com/https://github.com/soumith/torch-signal//blob/c3ea4516bf3c6c17077c342b9010873d21f732d6/init.lua#L569">[src]</a>
<a name="signal.hilbert"></a>
### signal.hilbert(xr) ###

Discrete-time analytic signal using Hilbert transform
   Takes 1D inputs

   Output matches with matlab output
