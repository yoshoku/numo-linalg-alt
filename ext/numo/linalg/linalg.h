/**
 * Copyright (c) 2025-2026 Atsushi Tatsuma
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

#ifndef NUMO_LINALG_ALT_LINALG_H
#define NUMO_LINALG_ALT_LINALG_H 1

#include <string.h>

#include <ruby.h>

#include <numo/narray.h>
#include <numo/template.h>

#ifndef _DEFINED_SCOMPLEX
#define _DEFINED_SCOMPLEX 1
#endif
#ifndef _DEFINED_DCOMPLEX
#define _DEFINED_DCOMPLEX 1
#endif

#include <cblas.h>
#include <lapacke.h>

#include "extconf.h"
#ifdef HAVE_OPENBLAS_CONFIG_H
#include <openblas_config.h>
#endif

#include "blas/dot.h"
#include "blas/dot_sub.h"
#include "blas/gemm.h"
#include "blas/gemv.h"
#include "blas/nrm2.h"

#include "lapack/gbsv.h"
#include "lapack/gebal.h"
#include "lapack/gees.h"
#include "lapack/geev.h"
#include "lapack/gehrd.h"
#include "lapack/gelsd.h"
#include "lapack/geqrf.h"
#include "lapack/gerqf.h"
#include "lapack/gesdd.h"
#include "lapack/gesv.h"
#include "lapack/gesvd.h"
#include "lapack/getrf.h"
#include "lapack/getri.h"
#include "lapack/getrs.h"
#include "lapack/gges.h"
#include "lapack/hbevx.h"
#include "lapack/heev.h"
#include "lapack/heevd.h"
#include "lapack/heevr.h"
#include "lapack/hegv.h"
#include "lapack/hegvd.h"
#include "lapack/hegvx.h"
#include "lapack/hetrf.h"
#include "lapack/lange.h"
#include "lapack/orghr.h"
#include "lapack/orgqr.h"
#include "lapack/orgrq.h"
#include "lapack/pbsv.h"
#include "lapack/pbtrf.h"
#include "lapack/pbtrs.h"
#include "lapack/potrf.h"
#include "lapack/potri.h"
#include "lapack/potrs.h"
#include "lapack/sbevx.h"
#include "lapack/stevx.h"
#include "lapack/syev.h"
#include "lapack/syevd.h"
#include "lapack/syevr.h"
#include "lapack/sygv.h"
#include "lapack/sygvd.h"
#include "lapack/sygvx.h"
#include "lapack/sytrf.h"
#include "lapack/trtrs.h"
#include "lapack/unghr.h"
#include "lapack/ungqr.h"
#include "lapack/ungrq.h"

#endif /* NUMO_LINALG_ALT_LINALG_H */
