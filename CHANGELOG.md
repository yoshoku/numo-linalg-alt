## [[0.8.0](https://github.com/yoshoku/numo-linalg-alt/compare/v0.7.2...v0.8.0)] - 2026-02-06

- relax numo-narray-alt version constraint to >= 0.9.10, < 0.11.0.
- add `diagsvd` module function to Numo::Linalg.
- add `cholesky_banded` and `cho_solve_banded` module functions to Numo::Linalg.
- add `solve_banded` and `solveh_banded` module functions to Numo::Linalg.

## [[0.7.2](https://github.com/yoshoku/numo-linalg-alt/compare/v0.7.1...v0.7.2)] - 2026-01-16

- bump OpenBLAS from 0.3.30 to 0.3.31.

## [[0.7.1](https://github.com/yoshoku/numo-linalg-alt/compare/v0.7.0...v0.7.1)] - 2025-11-19

- set the required version of numo-narray-alt to 0.9.10 or higher.
- change require statement to explicitly load numo/narray/alt.
- fix to use require for compatibility with distributions installing extensions separately.

## [[0.7.0](https://github.com/yoshoku/numo-linalg-alt/compare/v0.6.0...v0.7.0)] - 2025-11-11

**Breaking changes**

improve error handling for LAPACK functions:

- add `LapackError` class under `Numo::Linalg` module.
  - This exception class is raised when invalid arguments are passed to a LAPACK function or
    when the algorithm does not execute successfully. In previous versions, `StandardError`
    was raised in these cases.
    ```ruby
    > Numo::Linalg.inv(Numo::DFloat[[3, 1], [9, 3]])
    /numo-linalg-alt/lib/numo/linalg.rb:418:in 'Numo::Linalg.inv': The matrix is singular, and
    the inverse matrix could not be computed. (Numo::Linalg::LapackError)
    ```
- change `solve`, `lu_fact`, `ldl`, and `qz` methods to issue a warning instead of raising an
  error when the algorithm completes but produces a result that may affect further computations,
  such as when the resulting matrix is sigular.
  ```ruby
  > Numo::Linalg.solve(Numo::DFloat.zeros(2, 2), Numo::DFloat.ones(2))
  the factorization has been completed, but the factor is singular, so the solution could not be computed.
  =>
  Numo::DFloat#shape=[2]
  [1, 1]
  ```
- change `det` method to return zero instead of raising an error when the input matrix is singular.
  ```ruby
  Numo::Linalg.det(Numo::DFloat[[1, 2], [2, 4]])
  => -0.0
  ```

## [[0.6.0](https://github.com/yoshoku/numo-linalg-alt/compare/v0.5.0...v0.6.0)] - 2025-11-02

- add `--with-blas` and `--with-lapacke` options for selecting backend libraries.
- add `logm`, `sqrtm`, and `signm` module functions to Numo::Linalg.
- add error handling for LAPACK functions such as gesv, getri, and potrs.
- FIX: ensure pinv returns contiguous array instead of NArray's view.

## [[0.5.0](https://github.com/yoshoku/numo-linalg-alt/compare/v0.4.1...v0.5.0)] - 2025-10-25

- FIX: correct Numo::Linalg::Lapack.xlange return type to Float for Numo::DComplex and Numo::SComplex.
- add `coshm`, `sinhm`, and `tanhm` module functions to Numo::Linalg.
- add `hessenberg` module function to Numo::Linalg.

## [[0.4.1](https://github.com/yoshoku/numo-linalg-alt/compare/v0.4.0...v0.4.1)] - 2025-10-19

- FIX: remove incorrect usage of RUBY_METHOD_FUNC macro: [#2](https://github.com/yoshoku/numo-linalg-alt/pull/2)
- add `matrix_balance` moudle function to Numo::Linalg.

## [[0.4.0](https://github.com/yoshoku/numo-linalg-alt/compare/v0.3.0...v0.4.0)] - 2025-10-16

- add `rq`, `qz`, and `tanm` module functions to Numo::Linalg.

## [[0.3.0](https://github.com/yoshoku/numo-linalg-alt/compare/v0.2.0...v0.3.0)] - 2025-10-06

- add `schur`, `cosm`, `sinm`, `orthogonal_procrustes`, and `polar` module functions to Numo::Linalg.
- fix version specifier for numo-narray-alt.

## [[0.2.0](https://github.com/yoshoku/numo-linalg-alt/compare/d010476...ea50089)] - 2025-09-29

- fork from [Numo::TinyLinalg main branch](https://github.com/yoshoku/numo-tiny_linalg/tree/d0104765c560e9664a868b7a3e2f3144bd32c428)
- rewrite native extensions with C programming language.
- implement module functions such as matmul, matrix_power, svdvals, orth, null_space, lu, lu_fact, lu_inv, lu_solve, ldl, cho_fact, cho_inv, eig, eigvals, eigvalsh, cond, slogdet, matrix_rank, lstsq, and expm that are not implemented in Numo::TinyLinalg.
