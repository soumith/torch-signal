
<a class="entityLink" href="https://github.com/soumith/torch-signal/blob/272138111678b6da3f3a873204b4cb14aaef2e37/convolution.lua#L6">[src]</a>
<a name="signal.conv1d"></a>


### signal.conv1d(input, kernel, stride [, mode]) ###

1D convolution with specified stride. `mode` is `"valid"` by default, supported modes are `"valid"` and `"same"`.

Note: unlike the `conv` function in MATLAB/Octave, this function does not reverse the kernel.
