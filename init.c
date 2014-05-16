
#define TSIG_CONCAT_EXPAND(x,y,z) x ## y ## _ ## z
#define TSIG_CONCAT(x,y,z) TSIG_CONCAT_EXPAND(x,y,z)
#define signal_(NAME) TSIG_CONCAT(signal_,NAME, real)

#define real float
#include "conv.c"
#undef real

#define real double
#include "conv.c"
#undef real
