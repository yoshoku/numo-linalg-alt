# Numo::Linalg Alternative

[![Gem Version](https://badge.fury.io/rb/numo-linalg-alt.svg)](https://badge.fury.io/rb/numo-linalg-alt)
[![Build Status](https://github.com/yoshoku/numo-linalg-alt/actions/workflows/main.yml/badge.svg)](https://github.com/yoshoku/numo-linalg-alt/actions/workflows/main.yml)
[![BSD 3-Clause License](https://img.shields.io/badge/License-BSD%203--Clause-orange.svg)](https://github.com/yoshoku/numo-linalg-alt/blob/main/LICENSE.txt)
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://gemdocs.org/gems/numo-linalg-alt/)

Numo::Linalg Alternative (numo-linalg-alt) is an alternative to [Numo::Linalg](https://github.com/ruby-numo/numo-linalg).
Unlike Numo::Linalg, numo-linalg-alt depends on [Numo::NArray Alterntive](https://github.com/yoshoku/numo-narray-alt).

Please note that this gem was forked from [Numo::TinyLinalg](https://github.com/yoshoku/numo-tiny_linalg),
not Numo::Linalg, and therefore it does not support changing backend libraries for BLAS and LAPACK.
In addition, the version numbering rule is not compatible with that of Numo::Linalg.

The project owner has the utmost respect for Numo::Linalg and its creator, Prof. [Masahiro Tanaka](https://github.com/masa16).
This project is in no way intended to adversely affect the development of the original Numo::Linalg.

## Installation

Unlike Numo::Linalg, numo-linalg-alt only supports [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS)
as a backend library.

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

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/yoshoku/numo-linalg-alt.
This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [code of conduct](https://github.com/yoshoku/numo-linalg-alt/blob/main/CODE_OF_CONDUCT.md).

## Code of Conduct

Everyone interacting in the Numo::Linalg Alternative project's codebases, issue trackers, chat rooms and mailing lists is expected to follow the [code of conduct](https://github.com/yoshoku/numo-linalg-alt/blob/main/CODE_OF_CONDUCT.md).

## License

The gem is available as open source under the terms of the [BSD-3-Clause License](https://opensource.org/licenses/BSD-3-Clause).
