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

#include "linalg.hpp"

#include "converter.hpp"
#include "util.hpp"

#include "blas/dot.hpp"
#include "blas/dot_sub.hpp"
#include "blas/gemm.hpp"
#include "blas/gemv.hpp"
#include "blas/nrm2.hpp"
#include "lapack/geqrf.hpp"
#include "lapack/gesdd.hpp"
#include "lapack/gesv.hpp"
#include "lapack/gesvd.hpp"
#include "lapack/getrf.hpp"
#include "lapack/getri.hpp"
#include "lapack/heev.hpp"
#include "lapack/heevd.hpp"
#include "lapack/heevr.hpp"
#include "lapack/hegv.hpp"
#include "lapack/hegvd.hpp"
#include "lapack/hegvx.hpp"
#include "lapack/lange.hpp"
#include "lapack/orgqr.hpp"
#include "lapack/potrf.hpp"
#include "lapack/potrs.hpp"
#include "lapack/syev.hpp"
#include "lapack/syevd.hpp"
#include "lapack/syevr.hpp"
#include "lapack/sygv.hpp"
#include "lapack/sygvd.hpp"
#include "lapack/sygvx.hpp"
#include "lapack/trtrs.hpp"
#include "lapack/ungqr.hpp"

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

  std::string fn_str = type + std::string(rb_id2name(rb_to_id(rb_to_symbol(fn_name))));
  ID fn_id = rb_intern(fn_str.c_str());
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

extern "C" void Init_linalg(void) {
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

  Linalg::Dot<Linalg::numo_cDFloatId, double, Linalg::DDot>::define_module_function(rb_mLinalgBlas, "ddot");
  Linalg::Dot<Linalg::numo_cSFloatId, float, Linalg::SDot>::define_module_function(rb_mLinalgBlas, "sdot");
  Linalg::DotSub<Linalg::numo_cDComplexId, double, Linalg::ZDotuSub>::define_module_function(rb_mLinalgBlas, "zdotu");
  Linalg::DotSub<Linalg::numo_cSComplexId, float, Linalg::CDotuSub>::define_module_function(rb_mLinalgBlas, "cdotu");
  Linalg::Gemm<Linalg::numo_cDFloatId, double, Linalg::DGemm, Linalg::DConverter>::define_module_function(rb_mLinalgBlas, "dgemm");
  Linalg::Gemm<Linalg::numo_cSFloatId, float, Linalg::SGemm, Linalg::SConverter>::define_module_function(rb_mLinalgBlas, "sgemm");
  Linalg::Gemm<Linalg::numo_cDComplexId, dcomplex, Linalg::ZGemm, Linalg::ZConverter>::define_module_function(rb_mLinalgBlas, "zgemm");
  Linalg::Gemm<Linalg::numo_cSComplexId, scomplex, Linalg::CGemm, Linalg::CConverter>::define_module_function(rb_mLinalgBlas, "cgemm");
  Linalg::Gemv<Linalg::numo_cDFloatId, double, Linalg::DGemv, Linalg::DConverter>::define_module_function(rb_mLinalgBlas, "dgemv");
  Linalg::Gemv<Linalg::numo_cSFloatId, float, Linalg::SGemv, Linalg::SConverter>::define_module_function(rb_mLinalgBlas, "sgemv");
  Linalg::Gemv<Linalg::numo_cDComplexId, dcomplex, Linalg::ZGemv, Linalg::ZConverter>::define_module_function(rb_mLinalgBlas, "zgemv");
  Linalg::Gemv<Linalg::numo_cSComplexId, scomplex, Linalg::CGemv, Linalg::CConverter>::define_module_function(rb_mLinalgBlas, "cgemv");
  Linalg::Nrm2<Linalg::numo_cDFloatId, double, Linalg::DNrm2>::define_module_function(rb_mLinalgBlas, "dnrm2");
  Linalg::Nrm2<Linalg::numo_cSFloatId, float, Linalg::SNrm2>::define_module_function(rb_mLinalgBlas, "snrm2");
  Linalg::Nrm2<Linalg::numo_cDComplexId, double, Linalg::DZNrm2>::define_module_function(rb_mLinalgBlas, "dznrm2");
  Linalg::Nrm2<Linalg::numo_cSComplexId, float, Linalg::SCNrm2>::define_module_function(rb_mLinalgBlas, "scnrm2");
  Linalg::GeSv<Linalg::numo_cDFloatId, double, Linalg::DGeSv>::define_module_function(rb_mLinalgLapack, "dgesv");
  Linalg::GeSv<Linalg::numo_cSFloatId, float, Linalg::SGeSv>::define_module_function(rb_mLinalgLapack, "sgesv");
  Linalg::GeSv<Linalg::numo_cDComplexId, lapack_complex_double, Linalg::ZGeSv>::define_module_function(rb_mLinalgLapack, "zgesv");
  Linalg::GeSv<Linalg::numo_cSComplexId, lapack_complex_float, Linalg::CGeSv>::define_module_function(rb_mLinalgLapack, "cgesv");
  Linalg::GeSvd<Linalg::numo_cDFloatId, Linalg::numo_cDFloatId, double, double, Linalg::DGeSvd>::define_module_function(rb_mLinalgLapack, "dgesvd");
  Linalg::GeSvd<Linalg::numo_cSFloatId, Linalg::numo_cSFloatId, float, float, Linalg::SGeSvd>::define_module_function(rb_mLinalgLapack, "sgesvd");
  Linalg::GeSvd<Linalg::numo_cDComplexId, Linalg::numo_cDFloatId, lapack_complex_double, double, Linalg::ZGeSvd>::define_module_function(rb_mLinalgLapack, "zgesvd");
  Linalg::GeSvd<Linalg::numo_cSComplexId, Linalg::numo_cSFloatId, lapack_complex_float, float, Linalg::CGeSvd>::define_module_function(rb_mLinalgLapack, "cgesvd");
  Linalg::GeSdd<Linalg::numo_cDFloatId, Linalg::numo_cDFloatId, double, double, Linalg::DGeSdd>::define_module_function(rb_mLinalgLapack, "dgesdd");
  Linalg::GeSdd<Linalg::numo_cSFloatId, Linalg::numo_cSFloatId, float, float, Linalg::SGeSdd>::define_module_function(rb_mLinalgLapack, "sgesdd");
  Linalg::GeSdd<Linalg::numo_cDComplexId, Linalg::numo_cDFloatId, lapack_complex_double, double, Linalg::ZGeSdd>::define_module_function(rb_mLinalgLapack, "zgesdd");
  Linalg::GeSdd<Linalg::numo_cSComplexId, Linalg::numo_cSFloatId, lapack_complex_float, float, Linalg::CGeSdd>::define_module_function(rb_mLinalgLapack, "cgesdd");
  Linalg::GeTrf<Linalg::numo_cDFloatId, double, Linalg::DGeTrf>::define_module_function(rb_mLinalgLapack, "dgetrf");
  Linalg::GeTrf<Linalg::numo_cSFloatId, float, Linalg::SGeTrf>::define_module_function(rb_mLinalgLapack, "sgetrf");
  Linalg::GeTrf<Linalg::numo_cDComplexId, lapack_complex_double, Linalg::ZGeTrf>::define_module_function(rb_mLinalgLapack, "zgetrf");
  Linalg::GeTrf<Linalg::numo_cSComplexId, lapack_complex_float, Linalg::CGeTrf>::define_module_function(rb_mLinalgLapack, "cgetrf");
  Linalg::GeTri<Linalg::numo_cDFloatId, double, Linalg::DGeTri>::define_module_function(rb_mLinalgLapack, "dgetri");
  Linalg::GeTri<Linalg::numo_cSFloatId, float, Linalg::SGeTri>::define_module_function(rb_mLinalgLapack, "sgetri");
  Linalg::GeTri<Linalg::numo_cDComplexId, lapack_complex_double, Linalg::ZGeTri>::define_module_function(rb_mLinalgLapack, "zgetri");
  Linalg::GeTri<Linalg::numo_cSComplexId, lapack_complex_float, Linalg::CGeTri>::define_module_function(rb_mLinalgLapack, "cgetri");
  Linalg::TrTrs<Linalg::numo_cDFloatId, double, Linalg::DTrTrs>::define_module_function(rb_mLinalgLapack, "dtrtrs");
  Linalg::TrTrs<Linalg::numo_cSFloatId, float, Linalg::STrTrs>::define_module_function(rb_mLinalgLapack, "strtrs");
  Linalg::TrTrs<Linalg::numo_cDComplexId, lapack_complex_double, Linalg::ZTrTrs>::define_module_function(rb_mLinalgLapack, "ztrtrs");
  Linalg::TrTrs<Linalg::numo_cSComplexId, lapack_complex_float, Linalg::CTrTrs>::define_module_function(rb_mLinalgLapack, "ctrtrs");
  Linalg::PoTrf<Linalg::numo_cDFloatId, double, Linalg::DPoTrf>::define_module_function(rb_mLinalgLapack, "dpotrf");
  Linalg::PoTrf<Linalg::numo_cSFloatId, float, Linalg::SPoTrf>::define_module_function(rb_mLinalgLapack, "spotrf");
  Linalg::PoTrf<Linalg::numo_cDComplexId, lapack_complex_double, Linalg::ZPoTrf>::define_module_function(rb_mLinalgLapack, "zpotrf");
  Linalg::PoTrf<Linalg::numo_cSComplexId, lapack_complex_float, Linalg::CPoTrf>::define_module_function(rb_mLinalgLapack, "cpotrf");
  Linalg::PoTrs<Linalg::numo_cDFloatId, double, Linalg::DPoTrs>::define_module_function(rb_mLinalgLapack, "dpotrs");
  Linalg::PoTrs<Linalg::numo_cSFloatId, float, Linalg::SPoTrs>::define_module_function(rb_mLinalgLapack, "spotrs");
  Linalg::PoTrs<Linalg::numo_cDComplexId, lapack_complex_double, Linalg::ZPoTrs>::define_module_function(rb_mLinalgLapack, "zpotrs");
  Linalg::PoTrs<Linalg::numo_cSComplexId, lapack_complex_float, Linalg::CPoTrs>::define_module_function(rb_mLinalgLapack, "cpotrs");
  Linalg::GeQrf<Linalg::numo_cDFloatId, double, Linalg::DGeQrf>::define_module_function(rb_mLinalgLapack, "dgeqrf");
  Linalg::GeQrf<Linalg::numo_cSFloatId, float, Linalg::SGeQrf>::define_module_function(rb_mLinalgLapack, "sgeqrf");
  Linalg::GeQrf<Linalg::numo_cDComplexId, lapack_complex_double, Linalg::ZGeQrf>::define_module_function(rb_mLinalgLapack, "zgeqrf");
  Linalg::GeQrf<Linalg::numo_cSComplexId, lapack_complex_float, Linalg::CGeQrf>::define_module_function(rb_mLinalgLapack, "cgeqrf");
  Linalg::OrgQr<Linalg::numo_cDFloatId, double, Linalg::DOrgQr>::define_module_function(rb_mLinalgLapack, "dorgqr");
  Linalg::OrgQr<Linalg::numo_cSFloatId, float, Linalg::SOrgQr>::define_module_function(rb_mLinalgLapack, "sorgqr");
  Linalg::UngQr<Linalg::numo_cDComplexId, lapack_complex_double, Linalg::ZUngQr>::define_module_function(rb_mLinalgLapack, "zungqr");
  Linalg::UngQr<Linalg::numo_cSComplexId, lapack_complex_float, Linalg::CUngQr>::define_module_function(rb_mLinalgLapack, "cungqr");
  Linalg::SyEv<Linalg::numo_cDFloatId, double, Linalg::DSyEv>::define_module_function(rb_mLinalgLapack, "dsyev");
  Linalg::SyEv<Linalg::numo_cSFloatId, float, Linalg::SSyEv>::define_module_function(rb_mLinalgLapack, "ssyev");
  Linalg::HeEv<Linalg::numo_cDComplexId, Linalg::numo_cDFloatId, lapack_complex_double, double, Linalg::ZHeEv>::define_module_function(rb_mLinalgLapack, "zheev");
  Linalg::HeEv<Linalg::numo_cSComplexId, Linalg::numo_cSFloatId, lapack_complex_float, float, Linalg::CHeEv>::define_module_function(rb_mLinalgLapack, "cheev");
  Linalg::SyEvd<Linalg::numo_cDFloatId, double, Linalg::DSyEvd>::define_module_function(rb_mLinalgLapack, "dsyevd");
  Linalg::SyEvd<Linalg::numo_cSFloatId, float, Linalg::SSyEvd>::define_module_function(rb_mLinalgLapack, "ssyevd");
  Linalg::HeEvd<Linalg::numo_cDComplexId, Linalg::numo_cDFloatId, lapack_complex_double, double, Linalg::ZHeEvd>::define_module_function(rb_mLinalgLapack, "zheevd");
  Linalg::HeEvd<Linalg::numo_cSComplexId, Linalg::numo_cSFloatId, lapack_complex_float, float, Linalg::CHeEvd>::define_module_function(rb_mLinalgLapack, "cheevd");
  Linalg::SyEvr<Linalg::numo_cDFloatId, double, Linalg::DSyEvr>::define_module_function(rb_mLinalgLapack, "dsyevr");
  Linalg::SyEvr<Linalg::numo_cSFloatId, float, Linalg::SSyEvr>::define_module_function(rb_mLinalgLapack, "ssyevr");
  Linalg::HeEvr<Linalg::numo_cDComplexId, Linalg::numo_cDFloatId, lapack_complex_double, double, Linalg::ZHeEvr>::define_module_function(rb_mLinalgLapack, "zheevr");
  Linalg::HeEvr<Linalg::numo_cSComplexId, Linalg::numo_cSFloatId, lapack_complex_float, float, Linalg::CHeEvr>::define_module_function(rb_mLinalgLapack, "cheevr");
  Linalg::SyGv<Linalg::numo_cDFloatId, double, Linalg::DSyGv>::define_module_function(rb_mLinalgLapack, "dsygv");
  Linalg::SyGv<Linalg::numo_cSFloatId, float, Linalg::SSyGv>::define_module_function(rb_mLinalgLapack, "ssygv");
  Linalg::HeGv<Linalg::numo_cDComplexId, Linalg::numo_cDFloatId, lapack_complex_double, double, Linalg::ZHeGv>::define_module_function(rb_mLinalgLapack, "zhegv");
  Linalg::HeGv<Linalg::numo_cSComplexId, Linalg::numo_cSFloatId, lapack_complex_float, float, Linalg::CHeGv>::define_module_function(rb_mLinalgLapack, "chegv");
  Linalg::SyGvd<Linalg::numo_cDFloatId, double, Linalg::DSyGvd>::define_module_function(rb_mLinalgLapack, "dsygvd");
  Linalg::SyGvd<Linalg::numo_cSFloatId, float, Linalg::SSyGvd>::define_module_function(rb_mLinalgLapack, "ssygvd");
  Linalg::HeGvd<Linalg::numo_cDComplexId, Linalg::numo_cDFloatId, lapack_complex_double, double, Linalg::ZHeGvd>::define_module_function(rb_mLinalgLapack, "zhegvd");
  Linalg::HeGvd<Linalg::numo_cSComplexId, Linalg::numo_cSFloatId, lapack_complex_float, float, Linalg::CHeGvd>::define_module_function(rb_mLinalgLapack, "chegvd");
  Linalg::SyGvx<Linalg::numo_cDFloatId, double, Linalg::DSyGvx>::define_module_function(rb_mLinalgLapack, "dsygvx");
  Linalg::SyGvx<Linalg::numo_cSFloatId, float, Linalg::SSyGvx>::define_module_function(rb_mLinalgLapack, "ssygvx");
  Linalg::HeGvx<Linalg::numo_cDComplexId, Linalg::numo_cDFloatId, lapack_complex_double, double, Linalg::ZHeGvx>::define_module_function(rb_mLinalgLapack, "zhegvx");
  Linalg::HeGvx<Linalg::numo_cSComplexId, Linalg::numo_cSFloatId, lapack_complex_float, float, Linalg::CHeGvx>::define_module_function(rb_mLinalgLapack, "chegvx");
  Linalg::LanGe<Linalg::numo_cDFloatId, double, Linalg::DLanGe>::define_module_function(rb_mLinalgLapack, "dlange");
  Linalg::LanGe<Linalg::numo_cSFloatId, float, Linalg::SLanGe>::define_module_function(rb_mLinalgLapack, "slange");
  Linalg::LanGe<Linalg::numo_cDComplexId, lapack_complex_double, Linalg::ZLanGe>::define_module_function(rb_mLinalgLapack, "zlange");
  Linalg::LanGe<Linalg::numo_cSComplexId, lapack_complex_float, Linalg::CLanGe>::define_module_function(rb_mLinalgLapack, "clange");

  rb_define_alias(rb_singleton_class(rb_mLinalgBlas), "znrm2", "dznrm2");
  rb_define_alias(rb_singleton_class(rb_mLinalgBlas), "cnrm2", "scnrm2");
}
