#include "converter.h"

double conv_double(VALUE val) {
  return NUM2DBL(val);
}

double one_double(void) {
  return 1.0;
}

double zero_double(void) {
  return 0.0;
}

float conv_float(VALUE val) {
  return (float)NUM2DBL(val);
}

float one_float(void) {
  return 1.0f;
}

float zero_float(void) {
  return 0.0f;
}

dcomplex conv_dcomplex(VALUE val) {
  dcomplex z;
  REAL(z) = NUM2DBL(rb_funcall(val, rb_intern("real"), 0));
  IMAG(z) = NUM2DBL(rb_funcall(val, rb_intern("imag"), 0));
  return z;
}

dcomplex one_dcomplex(void) {
  dcomplex z;
  REAL(z) = 1.0;
  IMAG(z) = 0.0;
  return z;
}

dcomplex zero_dcomplex(void) {
  dcomplex z;
  REAL(z) = 0.0;
  IMAG(z) = 0.0;
  return z;
}

scomplex conv_scomplex(VALUE val) {
  scomplex z;
  REAL(z) = (float)NUM2DBL(rb_funcall(val, rb_intern("real"), 0));
  IMAG(z) = (float)NUM2DBL(rb_funcall(val, rb_intern("imag"), 0));
  return z;
}

scomplex one_scomplex(void) {
  scomplex z;
  REAL(z) = 1.0f;
  IMAG(z) = 0.0f;
  return z;
}

scomplex zero_scomplex(void) {
  scomplex z;
  REAL(z) = 0.0f;
  IMAG(z) = 0.0f;
  return z;
}
