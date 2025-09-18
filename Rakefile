# frozen_string_literal: true

require 'bundler/gem_tasks'
require 'rake/testtask'

Rake::TestTask.new(:test) do |t|
  t.libs << 'test'
  t.libs << 'lib'
  t.test_files = FileList['test/**/test_*.rb']
end

require 'rake/extensiontask'

task build: :compile # rubocop:disable Rake/Desc

desc 'Run clang-format'
task :'clang-format' do
  sh 'clang-format -style=file -Werror --dry-run ext/numo/linalg/*.cpp ext/numo/linalg/*.hpp'
end

Rake::ExtensionTask.new('linalg') do |ext|
  ext.ext_dir = 'ext/numo/linalg'
  ext.lib_dir = 'lib/numo/linalg'
end

task default: %i[clobber compile test]
