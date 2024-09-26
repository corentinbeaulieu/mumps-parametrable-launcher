# MUMPS Parametrable Launcher

This repository contains a wrapper around [MUMPS library](https://mumps-solver.org) to factorize generated
or given matrices with a selection of parameters and returns metrics about the execution.

## Installation

### Quick start

You can use spack to install the dependencies with the following command with [this repository](<>).

### Building from sources

#### Dependencies

You need the following

- C compiler with support for C23 or C2x standard
- Meson ([link](https://mesonbuild.com/SimpleStart.html#installing-meson))
- MUMPS ([link](https://mumps-solver.org/index.php?page=dwnld))
- SPRAL ([modified version](https://github.com/corentinbeaulieu/spral-band-matrix-generator))
- MPI
- METIS
- SCOTCH and PTSCOTCH (optionnal)
- Doxygen (optionnal)

> \[!WARNING\]
> You need a modified version of spral with band matrix generation.
> Please find it [here](https://github.com/corentinbeaulieu/spral-band-matrix-generator)

The experiment rely on [SPRAL](https://github.com/ralna/spral) ([documentation](https://www.numerical.rl.ac.uk/spral/doc/latest/C/))
to generate sparse random matrix.

#### Meson

To build the `mumps-parametrable-launcher` executable

```
$ meson setup builddir --prefix=<install_dir>
```

Default paths for the dependency may be wrong, you can modify them with the following option

```
$ meson setup builddir -Dmetis-path=<metis-root-path> -Dspral-path=<spral-root-path>\
    -Dmumps-path=<mumps-root-path>
```

If you are using INTEL's MKL as your BLAS/LAPACK implementation, we must enable it through `-Dmkl=true` option.
If you want to generate the *Doxygen* documentation, please use `-Ddoc=true` option.
If you are using a SLURM based system, you can use the `-Dcluster=true` option to have generate the scripts with
SLURM's commands in it (`salloc`, `srun`...).

> \[!WARNING\]
> If MPI cannot be found, you can use specify `-Dmpi=false` and use `CC=<mpi compiler wrapper>` as a workaround

If you want to use SCOTCH as an ordering library, consider the following

```
$ meson setup builddir -Dscotch=true -Dscotch-path=<scotch-root-path>
```

Then, you can compile the project

```
$ meson compile -C builddir
```

or directly install it

```
$ meson install -C builddir
```

The necessary executables to run an experiment should be located in a `bin` directory in the install directory you specified
(default to the project root directory).

## Usage

The mumps executable has the following usage

```
USAGE: ./mumps-parametrable-launcher -i input_file PAR ICNTL_13 ICNTL_16 ordering
       ./mumps-parametrable-launcher datatype N bandwidth density symmetry_type PAR ICNTL_13 ICNTL_16 ordering
with
     datatype       0 (real) or 1 (complex)
     N              width/height of the generated matrix
     bandwidth      maximal upper/lower bandwidth of the generated matrix
     density        density of the bandwidth (total matrix with -g)
     symmetry_type  0 (unsymmetric), 1 (positive_definite), 2 (symmetric)
     ordering       ordering solution (0 default, 1 MeTiS, 2 PORD, 3 SCOTCH, 4 PT-SCOTCH)

Options:
    -h	        print this help and exit
    -s seed 	seed for random generation
    -i file     mtx file describing the matrix
    -a          do the analysis stage
    -f          do the factorization stage
    -r          enable MUMPS resolution stage
    -g          consider global matrix density instead of band ones
    -w          print the matrix in MTX format and exit
```

By default, if neither of the `afr` options are passed, the program calls MUMPS analysis and factorization
(equivalent to `-af`).

Upon installation, helper scripts are installed as well in the installation directory. To run an experiment, we use the `run_mumps.sh` script as followed

```
./run_mumps.sh job synthetic n bandwidth density symmetry num_proc num_threads par inctl13 ordering
```

or, to read a MTX format matrix to factorize

```
./run_mumps_file.sh input num_proc num_threads par icntl_13 ordering
```

with

- **job** Control which stage executed and metric returned ("ap" == analysis performance, "fe" == factorization energy...)
- **synthetic** Control whether we use the generator or a matrix from the [dataset](#Dataset)
- **n** Rank of the matrix
- **bandwidth** Maximal upper/lower bandwidth of the matrix
- **density** Global density of nnz in the matrix $\\left(\\frac{nnz}{n^2}\\right)$
- **symmetry** Type of symmetry of the matrix (please see `mumps` executable usage for further details)
- **num_proc** Number of MPI ranks to run with
- **num_theads** Number of OpenMP threads to run with
- **par** Value of the PAR MUMPS parameter (Userguide p. 26)
- **inctl13** Value of the _INCTL(13)_ MUMPS parameter (Userguide p. 75)
- **ordering** Which ordering solution to use (change _ICNTL(7)_, _ICNTL(28)_ and _ICNTL(29)_)
  - **0** Use automatic MUMPS choice (_ICNTL(7)_ = 7)
  - **1** Use MeTiS (_ICNTL(7)_ = 5)
  - **2** Use PORD  (_ICNTL(7)_ = 4)
  - **3** Use SCOTCH (_ICNTL(7)_ = 3)
  - **4** Use PT-SCOTCH (_ICNTL(27)_ = 2 _ICNTL(29)_ = 1)

The `run_mumps_analysis_perf.sh` `run_mumps_analysis_energy.sh` `run_mumps_factorization_perf.sh` `run_mumps_factorization_energy.sh`
scripts (installed withe the program) are helper scripts to common job/synthetic choices (always synthetic).

<details>
<summary> Repository structure </summary>

In this folder, we have the following files/directories

- `config.json`: the configuration file for the ML-KAPS experiment.
- `src/`: directory with the sources of a program calling MUMPS resolution on it.
- `include/`: directory with the include files needed to compile the project.
- `scripts/`: directory with the template launching scripts modified during the building of the project.
- `doc/`: directory with the generated documentation of this project (please see below for generation).
- `meson.build`: meson configuration to build and install the project

</details>
