# frozen_string_literal: true

require 'bundler/gem_tasks'
require 'ruby_memcheck' if ENV['BUNDLE_WITH'] == 'memcheck'
require 'rake/testtask'

if ENV['BUNDLE_WITH'] == 'memcheck'
  test_config = lambda do |t|
    t.libs << 'test'
    t.libs << 'lib'
    t.test_files = FileList['test/**/test_*.rb']
  end
  Rake::TestTask.new(test: :compile, &test_config)
  namespace :test do
    RubyMemcheck::TestTask.new(valgrind: :compile, &test_config)
  end
else
  Rake::TestTask.new(:test) do |t|
    t.libs << 'test'
    t.libs << 'lib'
    t.test_files = FileList['test/**/test_*.rb']
  end
end

require 'rake/extensiontask'

task build: :compile # rubocop:disable Rake/Desc

desc 'Run clang-format'
task :'clang-format' do
  sh 'clang-format -style=file -Werror --dry-run ext/numo/linalg/**/*.c ext/numo/linalg/**/*.h'
end

Rake::ExtensionTask.new('linalg') do |ext|
  ext.ext_dir = 'ext/numo/linalg'
  ext.lib_dir = 'lib/numo/linalg'
end

task default: %i[clobber compile test]
