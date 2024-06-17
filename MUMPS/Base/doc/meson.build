cdata.set('TOP_SRCDIR', meson.project_source_root())
cdata.set('TOP_BUILDDIR', meson.project_build_root())

doxyfile = configure_file(
  input: 'Doxyfile.in',
  output: 'Doxyfile',
  configuration: cdata,
  install: false,
)

datadir = meson.project_source_root() / 'doc'

_ = custom_target(
  'docs',
  input: doxyfile,
  output: 'html',
  command: [doxygen, doxyfile],
  install: true,
  install_dir: datadir,
)
