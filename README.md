# ML-KAPS Autotuning Experiments

This repository keeps experiments to autotune sparse matrix algebra using ML-KAPS.
For now, it contains scripts and programs to use ML-KAPS on the [MUMPS library](https://mumps-solver.org).

## MUMPS

### Base
This directory contains the files for a basic ML-KAPS experiment on MUMPS.
In this folder, we have the following files:
- `config.json`: the configuration file for the ML-KAPS experiment.
- `mumps.c`: source file of a program reading a mtx formatted matrix and calling MUMPS resolution on it.
- `Makefile`: makefile to compile and link the previous program (**make sure to modify the paths**).
- `run_mumps.sh`: script to call the previous program with specific parameters. This is the script called by MLKAPS.
- `run_mlkaps.batch`: a slurm batch script to launch the ML-KAPS experiment (**make sure to modify the paths**).

#### Build
To build the `mumps` executable 
```sh
$ make
```
Make sure you have modified the paths in the `Makefile` to reflect the installation directory on your machine.

(We are testing the integration of meson to build the project which will simplify the build process
as no modification will be necessary

#### Usage
The mumps executable has the following usage
```
USAGE: ./mumps -f input_file PAR ICNTL_13 ICNTL_16
       ./mumps data_type N nnz symmetry_type PAR ICNTL_13 ICNTL_16
with
     data_type      0 (real) or 1 (complex)
     N              width/height of the generated matrix
     nnz            number of non-zero (must be < N*N + 1)
     symmetry_type  0 (unsymmetric), 1 (positive_definite), 2 (symmetric)

Options:
	-h	print this help and exit
	-s seed	seed for random generation
```

To run an experiment on SLURM based system, we use the `run_mumps.sh` script as followed
```
./run_mumps.sh n nnz symmetry num_proc num_threads par inctl13
```
with
- **n** Rank of the matrix
- **nnz** Number of non-zero elements in the generated matrix 
- **symmetry** Type of symmetry of the matrix (please see `mumps` executable usage for further details)
- **num_proc** Number of MPI ranks to run with
- **num_theads** Number of OpenMP threads to run with
- **par** Value of the PAR MUMPS parameter (Userguide p. 26)
- **inctl13** Value of the INCTL(13) MUMPS parameter (Userguide p. 75)


### SPRAL
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
