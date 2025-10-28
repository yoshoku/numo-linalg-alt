#include "gees.h"

#define DEF_GEES_OPTION(fLapackFunc, tSelectFunc)                                              \
  struct _gees_option_##fLapackFunc {                                                          \
    int matrix_layout;                                                                         \
    char jobvs;                                                                                \
    char sort;                                                                                 \
    tSelectFunc select;                                                                        \
  };

#define DEF_GEES_SORT_FUNC(tDType, fLapackFunc)                                                \
  lapack_logical _sort_nil_##fLapackFunc(const tDType* wr, const tDType* wi) {                 \
    return 0;                                                                                  \
  }                                                                                            \
  lapack_logical _sort_lhp_##fLapackFunc(const tDType* wr, const tDType* wi) {                 \
    if (*wr < (tDType)0) {                                                                     \
      return 1;                                                                                \
    }                                                                                          \
    return 0;                                                                                  \
  }                                                                                            \
  lapack_logical _sort_rhp_##fLapackFunc(const tDType* wr, const tDType* wi) {                 \
    if (*wr >= (tDType)0) {                                                                    \
      return 1;                                                                                \
    }                                                                                          \
    return 0;                                                                                  \
  }                                                                                            \
  lapack_logical _sort_iup_##fLapackFunc(const tDType* wr, const tDType* wi) {                 \
    tDType magnitude = *wr * *wr + *wi * *wi;                                                  \
    if (magnitude <= (tDType)1) {                                                              \
      return 1;                                                                                \
    }                                                                                          \
    return 0;                                                                                  \
  }                                                                                            \
  lapack_logical _sort_ouc_##fLapackFunc(const tDType* wr, const tDType* wi) {                 \
    tDType magnitude = *wr * *wr + *wi * *wi;                                                  \
    if (magnitude > (tDType)1) {                                                               \
      return 1;                                                                                \
    }                                                                                          \
    return 0;                                                                                  \
  }

#define DEF_GEES_SORT_FUNC_COMPLEX(                                                            \
  tDType, tElType, fLapackRealFunc, fLapackImagFunc, fLapackFunc                               \
)                                                                                              \
  lapack_logical _sort_nil_##fLapackFunc(const tDType* w) {                                    \
    return 0;                                                                                  \
  }                                                                                            \
  lapack_logical _sort_lhp_##fLapackFunc(const tDType* w) {                                    \
    if (fLapackRealFunc(*w) < 0.0) {                                                           \
      return 1;                                                                                \
    }                                                                                          \
    return 0;                                                                                  \
  }                                                                                            \
  lapack_logical _sort_rhp_##fLapackFunc(const tDType* w) {                                    \
    if (fLapackRealFunc(*w) >= 0.0) {                                                          \
      return 1;                                                                                \
    }                                                                                          \
    return 0;                                                                                  \
  }                                                                                            \
  lapack_logical _sort_iup_##fLapackFunc(const tDType* w) {                                    \
    tElType real = fLapackRealFunc(*w);                                                        \
    tElType imag = fLapackImagFunc(*w);                                                        \
    tElType magnitude = real * real + imag * imag;                                             \
    if (magnitude <= (tElType)1.0) {                                                           \
      return 1;                                                                                \
    }                                                                                          \
    return 0;                                                                                  \
  }                                                                                            \
  lapack_logical _sort_ouc_##fLapackFunc(const tDType* w) {                                    \
    tElType real = fLapackRealFunc(*w);                                                        \
    tElType imag = fLapackImagFunc(*w);                                                        \
    tElType magnitude = real * real + imag * imag;                                             \
    if (magnitude > (tElType)1.0) {                                                            \
      return 1;                                                                                \
    }                                                                                          \
    return 0;                                                                                  \
  }

#define DEF_LINALG_FUNC(tDType, tNAryClass, fLapackFunc)                                       \
  static void _iter_##fLapackFunc(na_loop_t* const lp) {                                       \
    tDType* a = (tDType*)(NDL_PTR(lp, 0));                                                     \
    tDType* wr = (tDType*)(NDL_PTR(lp, 1));                                                    \
    tDType* wi = (tDType*)(NDL_PTR(lp, 2));                                                    \
    tDType* vs = (tDType*)(NDL_PTR(lp, 3));                                                    \
    int* sdim = (int*)(NDL_PTR(lp, 4));                                                        \
    int* info = (int*)(NDL_PTR(lp, 5));                                                        \
    struct _gees_option_##fLapackFunc* opt =                                                   \
      (struct _gees_option_##fLapackFunc*)(lp->opt_ptr);                                       \
    const lapack_int n =                                                                       \
      (lapack_int)(opt->matrix_layout == LAPACK_ROW_MAJOR ? NDL_SHAPE(lp, 0)[0]                \
                                                          : NDL_SHAPE(lp, 0)[1]);              \
    const lapack_int lda = n;                                                                  \
    const lapack_int ldvs = n;                                                                 \
    lapack_int s = 0;                                                                          \
    lapack_int i = LAPACKE_##fLapackFunc(                                                      \
      opt->matrix_layout, opt->jobvs, opt->sort, opt->select, n, a, lda, &s, wr, wi, vs, ldvs  \
    );                                                                                         \
    *sdim = (int)s;                                                                            \
    *info = (int)i;                                                                            \
  }                                                                                            \
                                                                                               \
  static VALUE _linalg_lapack_##fLapackFunc(int argc, VALUE* argv, VALUE self) {               \
    VALUE a_vnary = Qnil;                                                                      \
    VALUE kw_args = Qnil;                                                                      \
    rb_scan_args(argc, argv, "1:", &a_vnary, &kw_args);                                        \
    ID kw_table[3] = { rb_intern("order"), rb_intern("jobvs"), rb_intern("sort") };            \
    VALUE kw_values[3] = { Qundef, Qundef, Qundef };                                           \
    rb_get_kwargs(kw_args, kw_table, 0, 3, kw_values);                                         \
    const int matrix_layout =                                                                  \
      kw_values[0] != Qundef ? get_matrix_layout(kw_values[0]) : LAPACK_ROW_MAJOR;             \
    const char jobvs = kw_values[1] != Qundef ? get_job(kw_values[1], "jobvs") : 'V';          \
    VALUE sort_val = kw_values[2] != Qundef ? kw_values[2] : Qnil;                             \
    const char sort_ch = NIL_P(sort_val) ? 'N' : 'S';                                          \
                                                                                               \
    if (CLASS_OF(a_vnary) != tNAryClass) {                                                     \
      a_vnary = rb_funcall(tNAryClass, rb_intern("cast"), 1, a_vnary);                         \
    }                                                                                          \
    if (!RTEST(nary_check_contiguous(a_vnary))) {                                              \
      a_vnary = nary_dup(a_vnary);                                                             \
    }                                                                                          \
                                                                                               \
    narray_t* a_nary = NULL;                                                                   \
    GetNArray(a_vnary, a_nary);                                                                \
    if (NA_NDIM(a_nary) != 2) {                                                                \
      rb_raise(rb_eArgError, "input array must be 2-dimensional array");                       \
      return Qnil;                                                                             \
    }                                                                                          \
                                                                                               \
    size_t n = matrix_layout == LAPACK_ROW_MAJOR ? NA_SHAPE(a_nary)[0] : NA_SHAPE(a_nary)[1];  \
    size_t shape_wr[1] = { n };                                                                \
    size_t shape_wi[1] = { n };                                                                \
    size_t shape_vs[2] = { n, jobvs == 'N' ? 1 : n };                                          \
    ndfunc_arg_in_t ain[1] = { { OVERWRITE, 2 } };                                             \
    ndfunc_arg_out_t aout[5] = { { tNAryClass, 1, shape_wr },                                  \
                                 { tNAryClass, 1, shape_wi },                                  \
                                 { tNAryClass, 2, shape_vs },                                  \
                                 { numo_cInt32, 0 },                                           \
                                 { numo_cInt32, 0 } };                                         \
    ndfunc_t ndf = { _iter_##fLapackFunc, NO_LOOP | NDF_EXTRACT, 1, 5, ain, aout };            \
    struct _gees_option_##fLapackFunc opt = { matrix_layout, jobvs, sort_ch, NULL };           \
    const char* sort_str = NIL_P(sort_val) ? "" : StringValueCStr(sort_val);                   \
    if (NIL_P(sort_val)) {                                                                     \
      opt.select = _sort_nil_##fLapackFunc;                                                    \
    } else if (strcmp(sort_str, "lhp") == 0) {                                                 \
      opt.select = _sort_lhp_##fLapackFunc;                                                    \
    } else if (strcmp(sort_str, "rhp") == 0) {                                                 \
      opt.select = _sort_rhp_##fLapackFunc;                                                    \
    } else if (strcmp(sort_str, "iup") == 0) {                                                 \
      opt.select = _sort_iup_##fLapackFunc;                                                    \
    } else if (strcmp(sort_str, "ouc") == 0) {                                                 \
      opt.select = _sort_ouc_##fLapackFunc;                                                    \
    } else {                                                                                   \
      rb_raise(rb_eArgError, "invalid value for sort option");                                 \
      return Qnil;                                                                             \
    }                                                                                          \
    VALUE res = na_ndloop3(&ndf, &opt, 1, a_vnary);                                            \
    VALUE ret = rb_ary_concat(rb_ary_new3(1, a_vnary), res);                                   \
                                                                                               \
    RB_GC_GUARD(sort_val);                                                                     \
    RB_GC_GUARD(a_vnary);                                                                      \
    return ret;                                                                                \
  }

#define DEF_LINALG_FUNC_COMPLEX(tDType, tNAryClass, fLapackFunc)                               \
  static void _iter_##fLapackFunc(na_loop_t* const lp) {                                       \
    tDType* a = (tDType*)(NDL_PTR(lp, 0));                                                     \
    tDType* w = (tDType*)(NDL_PTR(lp, 1));                                                     \
    tDType* vs = (tDType*)(NDL_PTR(lp, 2));                                                    \
    int* sdim = (int*)(NDL_PTR(lp, 3));                                                        \
    int* info = (int*)(NDL_PTR(lp, 4));                                                        \
    struct _gees_option_##fLapackFunc* opt =                                                   \
      (struct _gees_option_##fLapackFunc*)(lp->opt_ptr);                                       \
    const lapack_int n =                                                                       \
      (lapack_int)(opt->matrix_layout == LAPACK_ROW_MAJOR ? NDL_SHAPE(lp, 0)[0]                \
                                                          : NDL_SHAPE(lp, 0)[1]);              \
    const lapack_int lda = n;                                                                  \
    const lapack_int ldvs = n;                                                                 \
    lapack_int s = 0;                                                                          \
    lapack_int i = LAPACKE_##fLapackFunc(                                                      \
      opt->matrix_layout, opt->jobvs, opt->sort, opt->select, n, a, lda, &s, w, vs, ldvs       \
    );                                                                                         \
    *sdim = (int)s;                                                                            \
    *info = (int)i;                                                                            \
  }                                                                                            \
                                                                                               \
  static VALUE _linalg_lapack_##fLapackFunc(int argc, VALUE* argv, VALUE self) {               \
    VALUE a_vnary = Qnil;                                                                      \
    VALUE kw_args = Qnil;                                                                      \
    rb_scan_args(argc, argv, "1:", &a_vnary, &kw_args);                                        \
    ID kw_table[3] = { rb_intern("order"), rb_intern("jobvs"), rb_intern("sort") };            \
    VALUE kw_values[3] = { Qundef, Qundef, Qundef };                                           \
    rb_get_kwargs(kw_args, kw_table, 0, 3, kw_values);                                         \
    const int matrix_layout =                                                                  \
      kw_values[0] != Qundef ? get_matrix_layout(kw_values[0]) : LAPACK_ROW_MAJOR;             \
    const char jobvs = kw_values[1] != Qundef ? get_job(kw_values[1], "jobvs") : 'V';          \
    VALUE sort_val = kw_values[2] != Qundef ? kw_values[2] : Qnil;                             \
    const char sort_ch = NIL_P(sort_val) ? 'N' : 'S';                                          \
                                                                                               \
    if (CLASS_OF(a_vnary) != tNAryClass) {                                                     \
      a_vnary = rb_funcall(tNAryClass, rb_intern("cast"), 1, a_vnary);                         \
    }                                                                                          \
    if (!RTEST(nary_check_contiguous(a_vnary))) {                                              \
      a_vnary = nary_dup(a_vnary);                                                             \
    }                                                                                          \
                                                                                               \
    narray_t* a_nary = NULL;                                                                   \
    GetNArray(a_vnary, a_nary);                                                                \
    if (NA_NDIM(a_nary) != 2) {                                                                \
      rb_raise(rb_eArgError, "input array must be 2-dimensional array");                       \
      return Qnil;                                                                             \
    }                                                                                          \
                                                                                               \
    size_t n = matrix_layout == LAPACK_ROW_MAJOR ? NA_SHAPE(a_nary)[0] : NA_SHAPE(a_nary)[1];  \
    size_t shape_w[1] = { n };                                                                 \
    size_t shape_vs[2] = { n, jobvs == 'N' ? 1 : n };                                          \
    ndfunc_arg_in_t ain[1] = { { OVERWRITE, 2 } };                                             \
    ndfunc_arg_out_t aout[4] = { { tNAryClass, 1, shape_w },                                   \
                                 { tNAryClass, 2, shape_vs },                                  \
                                 { numo_cInt32, 0 },                                           \
                                 { numo_cInt32, 0 } };                                         \
    ndfunc_t ndf = { _iter_##fLapackFunc, NO_LOOP | NDF_EXTRACT, 1, 4, ain, aout };            \
    struct _gees_option_##fLapackFunc opt = { matrix_layout, jobvs, sort_ch, NULL };           \
    const char* sort_str = NIL_P(sort_val) ? "" : StringValueCStr(sort_val);                   \
    if (NIL_P(sort_val)) {                                                                     \
      opt.select = _sort_nil_##fLapackFunc;                                                    \
    } else if (strcmp(sort_str, "lhp") == 0) {                                                 \
      opt.select = _sort_lhp_##fLapackFunc;                                                    \
    } else if (strcmp(sort_str, "rhp") == 0) {                                                 \
      opt.select = _sort_rhp_##fLapackFunc;                                                    \
    } else if (strcmp(sort_str, "iup") == 0) {                                                 \
      opt.select = _sort_iup_##fLapackFunc;                                                    \
    } else if (strcmp(sort_str, "ouc") == 0) {                                                 \
      opt.select = _sort_ouc_##fLapackFunc;                                                    \
    } else {                                                                                   \
      rb_raise(rb_eArgError, "invalid value for sort option");                                 \
      return Qnil;                                                                             \
    }                                                                                          \
    VALUE res = na_ndloop3(&ndf, &opt, 1, a_vnary);                                            \
    VALUE ret = rb_ary_concat(rb_ary_new3(1, a_vnary), res);                                   \
                                                                                               \
    RB_GC_GUARD(sort_val);                                                                     \
    RB_GC_GUARD(a_vnary);                                                                      \
    return ret;                                                                                \
  }

DEF_GEES_OPTION(dgees, LAPACK_D_SELECT2)
DEF_GEES_OPTION(sgees, LAPACK_S_SELECT2)
DEF_GEES_OPTION(zgees, LAPACK_Z_SELECT1)
DEF_GEES_OPTION(cgees, LAPACK_C_SELECT1)

DEF_GEES_SORT_FUNC(double, dgees)
DEF_GEES_SORT_FUNC(float, sgees)
DEF_GEES_SORT_FUNC_COMPLEX(
  lapack_complex_double, double, lapack_complex_double_real, lapack_complex_double_imag, zgees
)
DEF_GEES_SORT_FUNC_COMPLEX(
  lapack_complex_float, float, lapack_complex_float_real, lapack_complex_float_imag, cgees
)

DEF_LINALG_FUNC(double, numo_cDFloat, dgees)
DEF_LINALG_FUNC(float, numo_cSFloat, sgees)
DEF_LINALG_FUNC_COMPLEX(lapack_complex_double, numo_cDComplex, zgees)
DEF_LINALG_FUNC_COMPLEX(lapack_complex_float, numo_cSComplex, cgees)

#undef DEF_GEES_OPTION
#undef DEF_GEES_SORT_FUNC
#undef DEF_GEES_SORT_FUNC_COMPLEX
#undef DEF_LINALG_FUNC
#undef DEF_LINALG_FUNC_COMPLEX

void define_linalg_lapack_gees(VALUE mLapack) {
  rb_define_module_function(mLapack, "dgees", _linalg_lapack_dgees, -1);
  rb_define_module_function(mLapack, "sgees", _linalg_lapack_sgees, -1);
  rb_define_module_function(mLapack, "zgees", _linalg_lapack_zgees, -1);
  rb_define_module_function(mLapack, "cgees", _linalg_lapack_cgees, -1);
}
