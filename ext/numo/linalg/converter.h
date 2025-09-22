#ifndef NUMO_LINALG_ALT_CONVERTER_H
#define NUMO_LINALG_ALT_CONVERTER_H 1

#include <ruby.h>

#include <cblas.h>

#include <numo/narray.h>

double conv_double(VALUE val);
double one_double(void);
double zero_double(void);
float conv_float(VALUE val);
float one_float(void);
float zero_float(void);
dcomplex conv_dcomplex(VALUE val);
dcomplex one_dcomplex(void);
dcomplex zero_dcomplex(void);
scomplex conv_scomplex(VALUE val);
scomplex one_scomplex(void);
scomplex zero_scomplex(void);

#endif /* NUMO_LINALG_ALT_CONVERTER_H */
