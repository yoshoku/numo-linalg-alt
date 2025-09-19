/**
 * Copyright (c) 2025 Atsushi Tatsuma
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "linalg.h"

VALUE rb_mLinalg;
VALUE rb_mLinalgBlas;
VALUE rb_mLinalgLapack;

char blas_char(VALUE nary_arr) {
  char type = 'n';
  const size_t n = RARRAY_LEN(nary_arr);
  for (size_t i = 0; i < n; i++) {
    VALUE arg = rb_ary_entry(nary_arr, i);
    if (RB_TYPE_P(arg, T_ARRAY)) {
      arg = rb_funcall(numo_cNArray, rb_intern("asarray"), 1, arg);
    }
    if (CLASS_OF(arg) == numo_cBit || CLASS_OF(arg) == numo_cInt64 || CLASS_OF(arg) == numo_cInt32 ||
        CLASS_OF(arg) == numo_cInt16 || CLASS_OF(arg) == numo_cInt8 || CLASS_OF(arg) == numo_cUInt64 ||
        CLASS_OF(arg) == numo_cUInt32 || CLASS_OF(arg) == numo_cUInt16 || CLASS_OF(arg) == numo_cUInt8) {
      if (type == 'n') {
        type = 'd';
      }
    } else if (CLASS_OF(arg) == numo_cDFloat) {
      if (type == 'c' || type == 'z') {
        type = 'z';
      } else {
        type = 'd';
      }
    } else if (CLASS_OF(arg) == numo_cSFloat) {
      if (type == 'n') {
        type = 's';
      }
    } else if (CLASS_OF(arg) == numo_cDComplex) {
      type = 'z';
    } else if (CLASS_OF(arg) == numo_cSComplex) {
      if (type == 'n' || type == 's') {
        type = 'c';
      } else if (type == 'd') {
        type = 'z';
      }
    }
  }
  return type;
}

static VALUE linalg_blas_char(int argc, VALUE* argv, VALUE self) {
  VALUE nary_arr = Qnil;
  rb_scan_args(argc, argv, "*", &nary_arr);

  const char type = blas_char(nary_arr);
  if (type == 'n') {
    rb_raise(rb_eTypeError, "invalid data type for BLAS/LAPACK");
    return Qnil;
  }

  return rb_str_new(&type, 1);
}

static VALUE linalg_blas_call(int argc, VALUE* argv, VALUE self) {
  VALUE fn_name = Qnil;
  VALUE nary_arr = Qnil;
  VALUE kw_args = Qnil;
  rb_scan_args(argc, argv, "1*:", &fn_name, &nary_arr, &kw_args);

  const char type = blas_char(nary_arr);
  if (type == 'n') {
    rb_raise(rb_eTypeError, "invalid data type for BLAS/LAPACK");
    return Qnil;
  }

  char fn_str[256];
  snprintf(fn_str, sizeof(fn_str), "%c%s",
           type, rb_id2name(rb_to_id(rb_to_symbol(fn_name))));
  ID fn_id = rb_intern(fn_str);
  size_t n = RARRAY_LEN(nary_arr);
  VALUE ret = Qnil;

  if (NIL_P(kw_args)) {
    VALUE* args = ALLOCA_N(VALUE, n);
    for (size_t i = 0; i < n; i++) {
      args[i] = rb_ary_entry(nary_arr, i);
    }
    ret = rb_funcallv(self, fn_id, n, args);
  } else {
    VALUE* args = ALLOCA_N(VALUE, n + 1);
    for (size_t i = 0; i < n; i++) {
      args[i] = rb_ary_entry(nary_arr, i);
    }
    args[n] = kw_args;
    ret = rb_funcallv_kw(self, fn_id, n + 1, args, RB_PASS_KEYWORDS);
  }

  return ret;
}

static VALUE linalg_dot(VALUE self, VALUE a_, VALUE b_) {
  VALUE a = IsNArray(a_) ? a_ : rb_funcall(numo_cNArray, rb_intern("asarray"), 1, a_);
  VALUE b = IsNArray(b_) ? b_ : rb_funcall(numo_cNArray, rb_intern("asarray"), 1, b_);

  VALUE arg_arr = rb_ary_new3(2, a, b);
  const char type = blas_char(arg_arr);
  if (type == 'n') {
    rb_raise(rb_eTypeError, "invalid data type for BLAS/LAPACK");
    return Qnil;
  }

  VALUE ret = Qnil;
  narray_t* a_nary = NULL;
  narray_t* b_nary = NULL;
  GetNArray(a, a_nary);
  GetNArray(b, b_nary);
  const int a_ndim = NA_NDIM(a_nary);
  const int b_ndim = NA_NDIM(b_nary);

  if (a_ndim == 1) {
    if (b_ndim == 1) {
      ID fn_id = type == 'c' || type == 'z' ? rb_intern("dotu") : rb_intern("dot");
      ret = rb_funcall(rb_mLinalgBlas, rb_intern("call"), 3, ID2SYM(fn_id), a, b);
    } else {
      VALUE kw_args = rb_hash_new();
      if (!RTEST(nary_check_contiguous(b)) && RTEST(rb_funcall(b, rb_intern("fortran_contiguous?"), 0))) {
        b = rb_funcall(b, rb_intern("transpose"), 0);
        rb_hash_aset(kw_args, ID2SYM(rb_intern("trans")), rb_str_new_cstr("N"));
      } else {
        rb_hash_aset(kw_args, ID2SYM(rb_intern("trans")), rb_str_new_cstr("T"));
      }
      char fn_name[] = "xgemv";
      fn_name[0] = type;
      VALUE argv[3] = { b, a, kw_args };
      ret = rb_funcallv_kw(rb_mLinalgBlas, rb_intern(fn_name), 3, argv, RB_PASS_KEYWORDS);
    }
  } else {
    if (b_ndim == 1) {
      VALUE kw_args = rb_hash_new();
      if (!RTEST(nary_check_contiguous(a)) && RTEST(rb_funcall(b, rb_intern("fortran_contiguous?"), 0))) {
        a = rb_funcall(a, rb_intern("transpose"), 0);
        rb_hash_aset(kw_args, ID2SYM(rb_intern("trans")), rb_str_new_cstr("T"));
      } else {
        rb_hash_aset(kw_args, ID2SYM(rb_intern("trans")), rb_str_new_cstr("N"));
      }
      char fn_name[] = "xgemv";
      fn_name[0] = type;
      VALUE argv[3] = { a, b, kw_args };
      ret = rb_funcallv_kw(rb_mLinalgBlas, rb_intern(fn_name), 3, argv, RB_PASS_KEYWORDS);
    } else {
      VALUE kw_args = rb_hash_new();
      if (!RTEST(nary_check_contiguous(a)) && RTEST(rb_funcall(b, rb_intern("fortran_contiguous?"), 0))) {
        a = rb_funcall(a, rb_intern("transpose"), 0);
        rb_hash_aset(kw_args, ID2SYM(rb_intern("transa")), rb_str_new_cstr("T"));
      } else {
        rb_hash_aset(kw_args, ID2SYM(rb_intern("transa")), rb_str_new_cstr("N"));
      }
      if (!RTEST(nary_check_contiguous(b)) && RTEST(rb_funcall(b, rb_intern("fortran_contiguous?"), 0))) {
        b = rb_funcall(b, rb_intern("transpose"), 0);
        rb_hash_aset(kw_args, ID2SYM(rb_intern("transb")), rb_str_new_cstr("T"));
      } else {
        rb_hash_aset(kw_args, ID2SYM(rb_intern("transb")), rb_str_new_cstr("N"));
      }
      char fn_name[] = "xgemm";
      fn_name[0] = type;
      VALUE argv[3] = { a, b, kw_args };
      ret = rb_funcallv_kw(rb_mLinalgBlas, rb_intern(fn_name), 3, argv, RB_PASS_KEYWORDS);
    }
  }

  RB_GC_GUARD(a);
  RB_GC_GUARD(b);

  return ret;
}

void Init_linalg(void) {
  rb_require("numo/narray");

  /**
   * Document-module: Numo::Linalg
   * Numo::Linalg is a subset library from Numo::Linalg consisting only of methods used in Machine Learning algorithms.
   */
  rb_mLinalg = rb_define_module_under(rb_mNumo, "Linalg");
  /**
   * Document-module: Numo::Linalg::Blas
   * Numo::Linalg::Blas is wrapper module of BLAS functions.
   * @!visibility private
   */
  rb_mLinalgBlas = rb_define_module_under(rb_mLinalg, "Blas");
  /**
   * Document-module: Numo::Linalg::Lapack
   * Numo::Linalg::Lapack is wrapper module of LAPACK functions.
   * @!visibility private
   */
  rb_mLinalgLapack = rb_define_module_under(rb_mLinalg, "Lapack");

  /* The version of OpenBLAS used in background library. */
  rb_define_const(rb_mLinalg, "OPENBLAS_VERSION", rb_str_new_cstr(OPENBLAS_VERSION));

  /**
   * Returns BLAS char ([sdcz]) defined by data-type of arguments.
   *
   * @overload blas_char(a, ...) -> String
   *   @param [Numo::NArray] a
   *   @return [String]
   */
  rb_define_module_function(rb_mLinalg, "blas_char", RUBY_METHOD_FUNC(linalg_blas_char), -1);
  /**
   * Calculates dot product of two vectors / matrices.
   *
   * @overload dot(a, b) -> [Float|Complex|Numo::NArray]
   *   @param [Numo::NArray] a
   *   @param [Numo::NArray] b
   *   @return [Float|Complex|Numo::NArray]
   */
  rb_define_module_function(rb_mLinalg, "dot", RUBY_METHOD_FUNC(linalg_dot), 2);
  /**
   * Calls BLAS function prefixed with BLAS char.
   *
   * @overload call(func, *args)
   *   @param func [Symbol] BLAS function name without BLAS char.
   *   @param args arguments of BLAS function.
   * @example
   *   Numo::Linalg::Blas.call(:gemv, a, b)
   */
  rb_define_singleton_method(rb_mLinalgBlas, "call", RUBY_METHOD_FUNC(linalg_blas_call), -1);

  define_linalg_blas_dot(rb_mLinalgBlas);
  define_linalg_blas_dot_sub(rb_mLinalgBlas);
  define_linalg_blas_gemm(rb_mLinalgBlas);
  define_linalg_blas_gemv(rb_mLinalgBlas);
  define_linalg_blas_nrm2(rb_mLinalgBlas);
  define_linalg_lapack_geqrf(rb_mLinalgLapack);
  define_linalg_lapack_orgqr(rb_mLinalgLapack);

  rb_define_alias(rb_singleton_class(rb_mLinalgBlas), "znrm2", "dznrm2");
  rb_define_alias(rb_singleton_class(rb_mLinalgBlas), "cnrm2", "scnrm2");
}
