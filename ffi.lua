local ffi = require 'ffi'
ffi.cdef[[
void signal_conv1d_float(float *y, float *x, float *k, const long yn, const long kn, long stride);
void signal_conv1d_double(double *y, double *x, double *k, const long yn, const long kn, long stride);
]]

return ffi.load(package.searchpath('libsignal', package.cpath))

