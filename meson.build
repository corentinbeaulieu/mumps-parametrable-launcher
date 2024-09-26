project(
  'MUMPS Parametrable Launcher',
  'c',
  version: '0.0.2',
  default_options: [
    'prefix=' + meson.project_source_root(),
    'buildtype=debugoptimized',
    'warning_level=0',
  ],
)

executable_name = 'mumps-parametrable-launcher'
deprecated_c2x = '-std=c2x'
preferred_c23 = '-std=c23'

cc = meson.get_compiler('c')
prefix = get_option('prefix')

if cc.get_id() == 'intel' or cc.get_id() == 'intel-llvm'
  add_global_arguments('-qopenmp', language: 'c')
  add_global_link_arguments('-qopenmp', language: 'c')
  if cc.version().split('.')[0].to_int() >= 2024
    add_global_arguments(preferred_c23, language: 'c')
  else
    add_global_arguments(deprecated_c2x, language: 'c')
  endif
else
  add_global_arguments('-fopenmp', language: 'c')
  add_global_arguments('-fanalyzer', language: 'c')
  add_global_link_arguments('-fopenmp', language: 'c')
  if cc.get_id() == 'clang'
    if cc.version().split('.')[0].to_int() >= 18
      add_global_arguments(preferred_c23, language: 'c')
    else
      add_global_arguments(deprecated_c2x, language: 'c')
    endif
  endif
  if cc.get_id() == 'gcc'
    if cc.version().split('.')[0].to_int() >= 14
      add_global_arguments(preferred_c23, language: 'c')
    else
      add_global_arguments(deprecated_c2x, language: 'c')
    endif
  endif
endif

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
  include_directories: incmetis,
)

# Check if spral is modified to generate band matrix
cc.has_header_symbol(
  'spral.h',
  'spral_random_matrix_generate_band_long',
  include_directories: incspral,
  required: true,
)

spral = declare_dependency(
  dependencies: cc.find_library('spral', dirs: libspral, required: true),
  include_directories: incspral,
)

dmumps_lib = cc.find_library('dmumps', dirs: libmumps, required: true)
zmumps_lib = cc.find_library('zmumps', dirs: libmumps, required: true)

mumpsdeps = declare_dependency(dependencies: [dmumps_lib, zmumps_lib], include_directories: incmumps)

# Define mandatory dependencies
deps = [metis, spral, mumpsdeps]

if get_option('mpi')
  deps += dependency('mpi', required: true)
endif

# Link with SCOTCH library if requested
if get_option('scotch')
  scotchpath = get_option('scotch-path')
  libscotch = scotchpath / 'lib'
  incscotch = include_directories(scotchpath / 'include')

  scotch = declare_dependency(
    dependencies: [
      cc.find_library('z', required: true),
      cc.find_library('scotch', dirs: libscotch, required: true),
      cc.find_library('ptscotch', dirs: libscotch, required: false),
      cc.find_library('ptesmumps', dirs: libscotch, required: false),
    ],
    include_directories: incscotch,
  )
  deps += scotch
endif

# Link with mkl if requested
if get_option('mkl')
  mkl = [
    cc.find_library('mkl_intel_lp64'),
    cc.find_library('mkl_intel_thread'),
    cc.find_library('mkl_core'),
  ]
  deps += mkl
endif

# Check if EAR is disponible to have energy measurements
ear_install_path = get_option('ear-path')
earlib = cc.find_library('ear_api', dirs: ear_install_path / 'lib', required: false)
if earlib.found()
  ear = declare_dependency(
    dependencies: earlib,
    include_directories: ear_install_path / 'include',
  )
  deps += ear
  add_global_arguments('-DUSE_EAR', language: 'c')
endif

# Define subdirectories and build targets
src = []
inc = include_directories('include', 'src')

subdir('src')

executable(
  executable_name,
  sources: src,
  dependencies: deps,
  include_directories: inc,
  install: true,
)

if get_option('cluster')
  launch_command = '''salloc -N 1 -n "$num_proc" -c "$num_threads" --job-name=mumps_run -p cpu_short --mem=32G --time=00:05:00 $ear_options --exclude="$node_list" \
             srun'''
else
  launch_command = 'mpirun -np "$num_proc" '
endif
# Add the launching scripts to the install directory
if get_option('spack')
  script_config = configuration_data(
    {
      'MUMPS_EXEC_CALL': executable_name,
      'RUN_MUMPS_CALL': 'run_mumps.sh',
      'LAUNCH_CMD': launch_command,
    },
  )
else
  script_config = configuration_data(
    {
      'MUMPS_EXEC_CALL': './' + executable_name,
      'RUN_MUMPS_CALL': './run_mumps.sh',
      'LAUNCH_CMD': launch_command,
    },
  )
endif

configure_file(
  input: meson.project_source_root() / 'scripts' / 'run_mumps.sh.in',
  configuration: script_config,
  output: 'run_mumps.sh',
  install: true,
  install_dir: prefix / 'bin',
  install_mode: 'rwxr-xr-x',
)

configure_file(
  input: meson.project_source_root() / 'scripts' / 'run_mumps_analysis_perf.sh.in',
  configuration: script_config,
  output: 'run_mumps_analysis_perf.sh',
  install: true,
  install_dir: prefix / 'bin',
  install_mode: 'rwxr-xr-x',
)
configure_file(
  input: meson.project_source_root() / 'scripts' / 'run_mumps_analysis_energy.sh.in',
  configuration: script_config,
  output: 'run_mumps_analysis_energy.sh',
  install: true,
  install_dir: prefix / 'bin',
  install_mode: 'rwxr-xr-x',
)
configure_file(
  input: meson.project_source_root() / 'scripts' / 'run_mumps_factorization_perf.sh.in',
  configuration: script_config,
  output: 'run_mumps_factorization_perf.sh',
  install: true,
  install_dir: prefix / 'bin',
  install_mode: 'rwxr-xr-x',
)
configure_file(
  input: meson.project_source_root() / 'scripts' / 'run_mumps_factorization_energy.sh.in',
  configuration: script_config,
  output: 'run_mumps_factorization_energy.sh',
  install: true,
  install_dir: prefix / 'bin',
  install_mode: 'rwxr-xr-x',
)

# Documentation generation
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
