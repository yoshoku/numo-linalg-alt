# Numo::Linalg Alternative

[![Gem Version](https://badge.fury.io/rb/numo-linalg-alt.svg)](https://badge.fury.io/rb/numo-linalg-alt)
[![Build Status](https://github.com/yoshoku/numo-linalg-alt/actions/workflows/main.yml/badge.svg)](https://github.com/yoshoku/numo-linalg-alt/actions/workflows/main.yml)
[![BSD 3-Clause License](https://img.shields.io/badge/License-BSD%203--Clause-orange.svg)](https://github.com/yoshoku/numo-linalg-alt/blob/main/LICENSE.txt)
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://gemdocs.org/gems/numo-linalg-alt/)

Numo::Linalg Alternative (numo-linalg-alt) is an alternative to [Numo::Linalg](https://github.com/ruby-numo/numo-linalg).
Unlike Numo::Linalg, numo-linalg-alt depends on [Numo::NArray Alterntive](https://github.com/yoshoku/numo-narray-alt).
Please note that this gem was forked from [Numo::TinyLinalg](https://github.com/yoshoku/numo-tiny_linalg),
not Numo::Linalg, so its version numbering rule is not compatible with that of Numo::Linalg.

The project owner has the utmost respect for Numo::Linalg and its creator, Prof. [Masahiro Tanaka](https://github.com/masa16).
This project is in no way intended to adversely affect the development of the original Numo::Linalg.

## Installation

numo-linalg-alt uses [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) as the default backend library.
If BLAS/LAPACKE-related libraries and include files are not found during installation,
the gem will automatically download and build OpenBLAS from source.
This process can significantly increase installation time,
so pre-installing OpenBLAS is recommended.

Install the OpenBLAS.

macOS:

```sh
$ brew install openblas
```

Ubuntu:

```sh
$ sudo apt-get install libopenblas-dev liblapacke-dev
```

Install the gem and add to the application's Gemfile by executing.

macOS:

```sh
$ bundle config build.numo-linalg-alt "--with-opt-dir=/opt/homebrew/opt/openblas"
$ bundle add numo-linalg-alt
```

Ubuntu:

```sh
$ bundle add numo-linalg-alt
```

If bundler is not being used to manage dependencies, install the gem by executing.

macOS:

```sh
$ gem install numo-linalg-alt -- --with-opt-dir=/opt/homebrew/opt/openblas
```

Ubuntu:

```sh
$ gem install numo-linalg-alt
```

### Using alternative backend libraries

The `--with-blas` and `--with-lapacke` options allow you to specify which BLAS/LAPACKE libraries
to use as the backend. The following instructions are intended for Ubuntu.

#### BLIS

Install the BLIS:

```sh
$ sudo apt-get install libblis-dev liblapacke-dev
```

To use BLIS as the BLAS library, execute the following gem command.
The `--with-lapacke` option is not required as LAPACKE is automatically selected.

```sh
$ gem install numo-linalg-alt -- --with-blas=blis
```

#### Intel MKL

Install the Intel MKL:

```sh
sudo apt-get install intel-mkl
```

Run the following command to use Intel MKL's `mkl_lapacke.h` as `lapacke.h`:

```sh
sudo update-alternatives --install /usr/include/x86_64-linux-gnu/lapacke.h lapacke.h-x86_64-linux-gnu /usr/include/mkl/mkl_lapacke.h 10
```

To use Intel MKL as the BLAS/LAPACKE libraries, execute the following gem command.
The `--with-lapacke` option is not required as the `mkl_rt` library includes LAPACKE functions.

```sh
$ gem install numo-linalg-alt -- --with-blas=mkl_rt
```

## Documentation

- [API Documentation](https://gemdocs.org/gems/numo-linalg-alt/0.4.1/)
- [Comparison with scipy.linalg and numpy.linalg](https://github.com/yoshoku/numo-linalg-alt/wiki/Comparison-with-scipy.linalg-and-numpy.linalg)

## Usage

An example of singular value decomposition.

```ruby
require 'numo/linalg'

x = Numo::DFloat.new(5, 2).rand.dot(Numo::DFloat.new(2, 3).rand)
# =>
# Numo::DFloat#shape=[5,3]
# [[0.104945, 0.0284236, 0.117406],
#  [0.862634, 0.210945, 0.922135],
#  [0.324507, 0.0752655, 0.339158],
#  [0.67085, 0.102594, 0.600882],
#  [0.404631, 0.116868, 0.46644]]

s, u, vt = Numo::Linalg.svd(x, job: 'S')

z = u.dot(s.diag).dot(vt)
# =>
# Numo::DFloat#shape=[5,3]
# [[0.104945, 0.0284236, 0.117406],
#  [0.862634, 0.210945, 0.922135],
#  [0.324507, 0.0752655, 0.339158],
#  [0.67085, 0.102594, 0.600882],
#  [0.404631, 0.116868, 0.46644]]

puts (x - z).abs.max
# => 4.440892098500626e-16
```

## Development

preparation:

```shell
$ git clone https://github.com/yoshoku/numo-linalg-alt
$ cd numo-linalg-alt
$ bundle install
```

build and test:

```
$ bundle exec rake compile
$ bundle exec rake test
```

linter:

```shell
$ bundle exec rubocop
$ clang-format --dry-run --Werror --style=file ext/**/*.h ext/**/*.c
```

This project follows [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
Please run `npm install` to set up husky and commitlint for commit message validation:

```shell
$ npm install
```

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/yoshoku/numo-linalg-alt.
This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [code of conduct](https://github.com/yoshoku/numo-linalg-alt/blob/main/CODE_OF_CONDUCT.md).

## Code of Conduct

Everyone interacting in the Numo::Linalg Alternative project's codebases, issue trackers, chat rooms and mailing lists is expected to follow the [code of conduct](https://github.com/yoshoku/numo-linalg-alt/blob/main/CODE_OF_CONDUCT.md).

## License

The gem is available as open source under the terms of the [BSD-3-Clause License](https://opensource.org/licenses/BSD-3-Clause).
