# Signals package for Torch-7

## Fourier Transforms
### fft(X, [n])
* Computes 1D DFT of X.
* If given an optional argument "n", truncates X, or pads X with zeros to create an n-dimensional input before doing the transform.
* The result is n x 2.
* X can be 1D Tensor of size n (real input), or 2D Tensor of size nx2 (complex input)

### ifft(X, [n])
* Computes 1D inverse DFT of X.
* If given an optional argument "n", truncates X, or pads X with zeros to create an n-dimensional input before doing the transform.
* The result is n x 2.
* X can be 1D Tensor of size n (real input), or 2D Tensor of size nx2 (complex input)

### fft2(X, [m], [n])
* Computes 2D DFT of X.
* If given optional arguments "m" and "n", truncates X, or pads X with zeros to create an mxn dimensional input before doing the transform.
* The result is m x n x 2.
* X can be 2D Tensor of size m x n (real input), or 3D Tensor of size m x n x 2 (complex input)

### ifft2(X, [m], [n])
* Computes 2D inverse DFT of X.
* If given optional arguments "m" and "n", truncates X, or pads X with zeros to create an mxn dimensional input before doing the transform.
* The result is m x n x 2.
* X can be 2D Tensor of size m x n (real input), or 3D Tensor of size m x n x 2 (complex input)