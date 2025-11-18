# frozen_string_literal: true

require_relative 'lib/numo/linalg/version'

Gem::Specification.new do |spec|
  spec.name = 'numo-linalg-alt'
  spec.version = Numo::Linalg::VERSION
  spec.authors = ['yoshoku']
  spec.email = ['yoshoku@outlook.com']

  spec.summary = <<~MSG
    Numo::Linalg Alternative (numo-linalg-alt) is an alternative to Numo::Linalg.
  MSG
  spec.description = <<~MSG
    Numo::Linalg Alternative (numo-linalg-alt) is an alternative to Numo::Linalg.
    Unlike Numo::Linalg, numo-linalg-alt depends on Numo::NArray Alternative (numo-narray-alt).
  MSG
  spec.homepage = 'https://github.com/yoshoku/numo-linalg-alt'
  spec.license = 'BSD-3-Clause'

  spec.metadata['homepage_uri'] = spec.homepage
  spec.metadata['source_code_uri'] = spec.homepage
  spec.metadata['changelog_uri'] = "#{spec.homepage}/blob/main/CHANGELOG.md"
  spec.metadata['documentation_uri'] = "https://gemdocs.org/gems/numo-linalg-alt/#{spec.version}/"

  # Specify which files should be added to the gem when it is released.
  # The `git ls-files -z` loads the files in the RubyGem that have been added into git.
  spec.files = Dir.chdir(__dir__) do
    `git ls-files -z`.split("\x0")
                     .reject { |f| f.match(%r{\A(?:(?:test|doc|node_modules|pkg|tmp|bin|\.git|\.github|\.husky)/)}) }
                     .select { |f| f.match(/\.(?:rb|rbs|h|hpp|c|cpp|md|txt)$/) }
  end
  spec.files << 'vendor/tmp/.gitkeep'
  spec.bindir = 'exe'
  spec.executables = spec.files.grep(%r{\Aexe/}) { |f| File.basename(f) }
  spec.require_paths = ['lib']
  spec.extensions = ['ext/numo/linalg/extconf.rb']

  spec.add_dependency 'numo-narray-alt', '~> 0.9.9'

  # For more information and examples about making a new gem, check out our
  # guide at: https://bundler.io/guides/creating_gem.html
  spec.metadata['rubygems_mfa_required'] = 'true'
end
