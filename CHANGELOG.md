## [[0.4.1](https://github.com/yoshoku/numo-linalg-alt/compare/v0.3.0...v0.4.0)] - 2025-10-19

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
