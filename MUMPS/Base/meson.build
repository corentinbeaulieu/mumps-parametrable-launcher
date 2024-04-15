project('MUMPS Parametrable Launch',
  'c',
  version: '0.0.1',
  default_options: [
    'prefix=' + meson.project_source_root(),
    'buildtype=debugoptimized',
    'warning_level=3',
    'c_std=c2x'
  ]
)

cc = meson.get_compiler('c')
prefix = get_option('prefix')

if cc.get_id() == 'intel' or cc.get_id() == 'intel-llvm'
  add_global_arguments('-qopenmp', language: 'c')
  add_global_link_arguments('-qopenmp', language: 'c')
else
  add_global_arguments('-fopenmp', language: 'c')
  add_global_link_arguments('-fopenmp', language: 'c')
endif

mpi = dependency('mpi', required: false)

metispath = get_option('metis-path') 
spralpath = get_option('spral-path')
mumpspath = get_option('mumps-path')

libmetis = metispath / 'lib'
libspral = spralpath / 'lib'
libmumps = mumpspath / 'lib'

incmetis = include_directories(metispath / 'include')
incspral = include_directories(spralpath / 'include')
incmumps = include_directories(mumpspath / 'include')

metis = declare_dependency(
  dependencies: cc.find_library('metis', dirs: libmetis, required: true),
  include_directories: incmetis
)

# Check if spral is modified to generate band matrix
cc.has_header_symbol(
  'spral.h',
  'spral_random_matrix_generate_band_long',
  include_directories:  incspral,
  required: true,
)

spral = declare_dependency(
  dependencies: cc.find_library('spral', dirs: libspral, required: true),
  include_directories: incspral 
)

dmumps_lib = cc.find_library('dmumps', dirs: libmumps, required: true)
zmumps_lib = cc.find_library('zmumps', dirs: libmumps, required: true)

mumpsdeps = declare_dependency(
  dependencies: [dmumps_lib, zmumps_lib],
  include_directories: incmumps
)

deps = [metis, spral, mumpsdeps]

# HACKY: intel-mpi isn't found on Ruche. 
# The workaround is to set the env variable CC=mpiicx to compile with mpi enabled
if mpi.found()
  deps += mpi
endif

if get_option('mkl')
  mkl = [
       cc.find_library('mkl_intel_lp64'),
       cc.find_library('mkl_intel_thread'),
       cc.find_library('mkl_core'),
  ]
  deps += mkl
endif

src = []
inc = include_directories('include', 'src')

subdir('src')

executable('mumps',
  sources: src,
  dependencies: deps,
  include_directories: inc,
  install: true
)

install_data(meson.project_source_root() / 'run_mumps.sh', 
  install_dir: prefix / 'bin',
  install_mode: 'r-xr-xr-x'
)

### Documentation Generation ###
docgen = get_option('doc')

if docgen
  doxygen = find_program('doxygen', required: false)
  if doxygen.found()
    cdata = configuration_data()
    cdata.set('PROJECT_NAME', meson.project_name())
    cdata.set('VERSION', meson.project_version())

    subdir('doc')
  endif
endif
