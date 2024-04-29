# ML-KAPS Autotuning Experiments

This repository keeps experiments to autotune sparse matrix algebra using ML-KAPS.
For now, it contains scripts and programs to use ML-KAPS on the [MUMPS library](https://mumps-solver.org).

## MUMPS

### Base
This directory contains the files for a basic ML-KAPS experiment on MUMPS.
In this folder, we have the following files/directories
- `config.json`: the configuration file for the ML-KAPS experiment.
- `src/`: directory with the sources of a program calling MUMPS resolution on it.
- `include/`: directory with the include files needed to compile the project.
- `doc/`: directory with the generated documentation of this project (please see below for generation).
- `meson.build`: meson configuration to build and install the project 
- `run_mumps.sh`: script to call the previous program with specific parameters. This is the script called by MLKAPS.
- `run_mlkaps.batch`: a slurm batch script to launch the ML-KAPS experiment (**make sure to modify the paths**).

#### Installation
##### Configuration
To build the `mumps` executable 
```sh
$ meson setup builddir --prefix=<install_dir>
```
Default paths for the dependency may be wrong, you can modify them with the following option
```sh
$ meson setup builddir -Dmetis-path=<metis-root-path> -Dspral-path=<spral-root-path>\
    -Dmumps-path=<mumps-root-path>
```

If you are using INTEL's MKL as your BLAS/LAPACK implementation, we must enable it through `-Dmkl=true` option.
If you want to generate the *Doxygen* documentation, please use `-Ddoc=true` option.

> [!WARNING]
> If MPI cannot be found, you can use `CC=<mpi compiler wrapper>` as a workaround

If you want to use SCOTCH as an ordering library, consider the following 
```sh
$ meson setup builddir -Dscotch=true -Dscotch-path=<scotch-root-path>
```

##### Compilation

```sh
$ meson compile -C builddir
```

##### Installation

```sh
$ meson install -C builddir
```

The necessary executables to run an experiment should be located in a `bin` directory in the install directory you specified
(default to the project root directory).

#### Usage
The mumps executable has the following usage
```
USAGE: ./mumps -i input_file PAR ICNTL_13 ICNTL_16 ordering
       ./mumps data_type N bandwidth density symmetry_type PAR ICNTL_13 ICNTL_16 ordering
with
     data_type      0 (real) or 1 (complex)
     N              width/height of the generated matrix
     bandwidth      maximal upper/lower bandwidth of the generated matrix
     density        density of the bandwidth (total matrix with -g)
     symmetry_type  0 (unsymmetric), 1 (positive_definite), 2 (symmetric)
     ordering       ordering solution (0 default, 1 MeTiS, 2 PORD, 3 SCOTCH, 4 PT-SCOTCH)

Options:
    -h	        print this help and exit
    -s seed 	seed for random generation
    -a          do the analysis stage
    -f          do the factorization stage
    -r          enable MUMPS resolution stage
    -g          consider global matrix density instead of band ones
```
By default, if neither of the `afr` options are passed, the program calls MUMPS analysis and factorization
(equivalent to `-af`).

To run an experiment on SLURM based system, we use the `run_mumps.sh` script as followed
```
./run_mumps.sh n bandwidth density symmetry num_proc num_threads par inctl13 ordering
```
with
- **n** Rank of the matrix
- **bandwidth** Maximal upper/lower bandwidth of the matrix
- **density** Global density of nnz in the matrix ($\frac{nnz}{n^2}$)
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


### SPRAL
> [!WARNING]
> You need a modified version of spral with band matrix generation. 
> Please find it [here](https://github.com/corentinbeaulieu/spral-band-matrix-generator)

The experiment rely on [SPRAL](https://github.com/ralna/spral) ([documentation](https://www.numerical.rl.ac.uk/spral/doc/latest/C/))
to generate sparse random matrix.



## Dataset

We use the matrix used in this [paper](https://hal.science/hal-03536031v1/document).
For now, we only use the real ones that can be retrieved from [MatrixMarket](https://sparse.tamu.edu/).

<details>
<summary> Matrix links </summary>

This matrix denoted with * aren't used until we can figure out a way to overcome the memory issue.

2. ss: https://sparse.tamu.edu/VLSI/ss
3. nlpkkt80: https://sparse.tamu.edu/Schenk/nlpkkt80 *
4. Serena: https://sparse.tamu.edu/Janna/Serena
5. Geo_1438: https://sparse.tamu.edu/Janna/Geo_1438
7. ML_Geer: https://sparse.tamu.edu/Janna/ML_Geer *
8. Transport: https://sparse.tamu.edu/Janna/Transport
9. Bump_2911: https://sparse.tamu.edu/Janna/Bump_2911
11. vas_stokes_1M: https://sparse.tamu.edu/VLSI/vas_stokes_1M
12. Hook_1498: https://sparse.tamu.edu/Janna/Hook_1498
13. Queen_4147: https://sparse.tamu.edu/Janna/Queen_4147 *
14. dielFilterV2real: https://sparse.tamu.edu/Dziekonski/dielFilterV2real
15. Flan_1565: https://sparse.tamu.edu/Janna/Flan_1565 *
18. PFlow_742: https://sparse.tamu.edu/Janna/PFlow_742
19. Cube_Coup_dt0: https://sparse.tamu.edu/Janna/Cube_Coup_dt0 *
23. Long_Coup_dt0: https://sparse.tamu.edu/Janna/Long_Coup_dt0 *

</details>
