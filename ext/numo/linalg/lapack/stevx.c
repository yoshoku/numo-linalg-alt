#include "stevx.h"

#define DEF_LINALG_FUNC(tDType, tNAryClass, fLapackFunc)                                       \
  struct _stevx_option_##tDType {                                                              \
    int matrix_layout;                                                                         \
    char jobz;                                                                                 \
    char range;                                                                                \
    tDType vl;                                                                                 \
    tDType vu;                                                                                 \
    lapack_int il;                                                                             \
    lapack_int iu;                                                                             \
  };                                                                                           \
                                                                                               \
  static void _iter_##fLapackFunc(na_loop_t* const lp) {                                       \
    tDType* d = (tDType*)NDL_PTR(lp, 0);                                                       \
    tDType* e = (tDType*)NDL_PTR(lp, 1);                                                       \
    int* m = (int*)NDL_PTR(lp, 2);                                                             \
    tDType* w = (tDType*)NDL_PTR(lp, 3);                                                       \
    tDType* z = (tDType*)NDL_PTR(lp, 4);                                                       \
    int* ifail = (int*)NDL_PTR(lp, 5);                                                         \
    int* info = (int*)NDL_PTR(lp, 6);                                                          \
    struct _stevx_option_##tDType* opt = (struct _stevx_option_##tDType*)(lp->opt_ptr);        \
    const lapack_int n = (lapack_int)NDL_SHAPE(lp, 0)[0];                                      \
    const tDType abstol = 0.0;                                                                 \
    const lapack_int ldz = opt->range != 'I' ? n : opt->iu - opt->il + 1;                      \
    const lapack_int i = LAPACKE_##fLapackFunc(                                                \
      opt->matrix_layout, opt->jobz, opt->range, n, d, e, opt->vl, opt->vu, opt->il, opt->iu,  \
      abstol, m, w, z, ldz, ifail                                                              \
    );                                                                                         \
    *info = (int)i;                                                                            \
  }                                                                                            \
                                                                                               \
  static VALUE _linalg_lapack_##fLapackFunc(int argc, VALUE* argv, VALUE self) {               \
    VALUE d_vnary = Qnil;                                                                      \
    VALUE e_vnary = Qnil;                                                                      \
    VALUE kw_args = Qnil;                                                                      \
    rb_scan_args(argc, argv, "2:", &d_vnary, &e_vnary, &kw_args);                              \
    ID kw_table[7] = { rb_intern("jobz"), rb_intern("range"), rb_intern("vl"),                 \
                       rb_intern("vu"),   rb_intern("il"),    rb_intern("iu"),                 \
                       rb_intern("order") };                                                   \
    VALUE kw_values[7] = { Qundef, Qundef, Qundef, Qundef, Qundef, Qundef, Qundef };           \
    rb_get_kwargs(kw_args, kw_table, 0, 7, kw_values);                                         \
    const char jobz = kw_values[0] != Qundef ? get_job(kw_values[0], "jobz") : 'V';            \
    const char range = kw_values[1] != Qundef ? get_range(kw_values[1]) : 'A';                 \
    const tDType vl = kw_values[2] != Qundef ? NUM2DBL(kw_values[2]) : 0.0;                    \
    const tDType vu = kw_values[3] != Qundef ? NUM2DBL(kw_values[3]) : 0.0;                    \
    const lapack_int il = kw_values[4] != Qundef ? NUM2INT(kw_values[4]) : 0;                  \
    const lapack_int iu = kw_values[5] != Qundef ? NUM2INT(kw_values[5]) : 0;                  \
    const int matrix_layout =                                                                  \
      kw_values[6] != Qundef ? get_matrix_layout(kw_values[6]) : LAPACK_ROW_MAJOR;             \
                                                                                               \
    if (CLASS_OF(d_vnary) != tNAryClass) {                                                     \
      d_vnary = rb_funcall(tNAryClass, rb_intern("cast"), 1, d_vnary);                         \
    }                                                                                          \
    if (!RTEST(nary_check_contiguous(d_vnary))) {                                              \
      d_vnary = nary_dup(d_vnary);                                                             \
    }                                                                                          \
    if (CLASS_OF(e_vnary) != tNAryClass) {                                                     \
      e_vnary = rb_funcall(tNAryClass, rb_intern("cast"), 1, e_vnary);                         \
    }                                                                                          \
    if (!RTEST(nary_check_contiguous(e_vnary))) {                                              \
      e_vnary = nary_dup(e_vnary);                                                             \
    }                                                                                          \
                                                                                               \
    narray_t* d_nary = NULL;                                                                   \
    GetNArray(d_vnary, d_nary);                                                                \
    if (NA_NDIM(d_nary) != 1) {                                                                \
      rb_raise(rb_eArgError, "input array d must be 1-dimensional");                           \
      return Qnil;                                                                             \
    }                                                                                          \
    narray_t* e_nary = NULL;                                                                   \
    GetNArray(e_vnary, e_nary);                                                                \
    if (NA_NDIM(e_nary) != 1) {                                                                \
      rb_raise(rb_eArgError, "input array e must be 1-dimensional");                           \
      return Qnil;                                                                             \
    }                                                                                          \
    if (range == 'V' && vu <= vl) {                                                            \
      rb_raise(rb_eArgError, "vu must be greater than vl");                                    \
      return Qnil;                                                                             \
    }                                                                                          \
    const size_t n = NA_SHAPE(d_nary)[0];                                                      \
    if (range == 'I' && (il < 1 || il > (lapack_int)n)) {                                      \
      rb_raise(rb_eArgError, "il must satisfy 1 <= il <= n");                                  \
      return Qnil;                                                                             \
    }                                                                                          \
    if (range == 'I' && (iu < 1 || iu > (lapack_int)n)) {                                      \
      rb_raise(rb_eArgError, "iu must satisfy 1 <= iu <= n");                                  \
      return Qnil;                                                                             \
    }                                                                                          \
    if (range == 'I' && iu < il) {                                                             \
      rb_raise(rb_eArgError, "iu must be greater than or equal to il");                        \
      return Qnil;                                                                             \
    }                                                                                          \
                                                                                               \
    size_t m = range != 'I' ? n : (size_t)(iu - il + 1);                                       \
    size_t w_shape[1] = { m };                                                                 \
    size_t z_shape[2] = { n, m };                                                              \
    size_t ifail_shape[1] = { m };                                                             \
    ndfunc_arg_in_t ain[2] = { { OVERWRITE, 1 }, { OVERWRITE, 1 } };                           \
    ndfunc_arg_out_t aout[5] = { { numo_cInt32, 0 },                                           \
                                 { tNAryClass, 1, w_shape },                                   \
                                 { tNAryClass, 2, z_shape },                                   \
                                 { numo_cInt32, 1, ifail_shape },                              \
                                 { numo_cInt32, 0 } };                                         \
    ndfunc_t ndf = { _iter_##fLapackFunc, NO_LOOP | NDF_EXTRACT, 2, 5, ain, aout };            \
    struct _stevx_option_##tDType opt = { matrix_layout, jobz, range, vl, vu, il, iu };        \
    VALUE ret = na_ndloop3(&ndf, &opt, 2, d_vnary, e_vnary);                                   \
                                                                                               \
    RB_GC_GUARD(d_vnary);                                                                      \
    RB_GC_GUARD(e_vnary);                                                                      \
    return ret;                                                                                \
  }

DEF_LINALG_FUNC(double, numo_cDFloat, dstevx)
DEF_LINALG_FUNC(float, numo_cSFloat, sstevx)

#undef DEF_LINALG_FUNC

void define_linalg_lapack_stevx(VALUE mLapack) {
  rb_define_module_function(mLapack, "dstevx", _linalg_lapack_dstevx, -1);
  rb_define_module_function(mLapack, "sstevx", _linalg_lapack_sstevx, -1);
}
