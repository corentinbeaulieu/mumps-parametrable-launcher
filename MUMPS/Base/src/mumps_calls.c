/** @file mumps_calls.c
 *  @brief Definition of the MUMPS lauching related functions
 */

#include "mumps_calls.h"
#include <stdlib.h>

static int experiment_real (const matrix_t a, const int par, const int icntl_13,
                            const int icntl_16, const bool resolve) {

    DMUMPS_STRUC_C dmumps_par;

    DMUMPS_REAL *rhs = nullptr;

    if (resolve == true) {
        rhs = (typeof(rhs)) malloc(a.n * sizeof(*rhs));
        generate_rhs(a.n, a.nnz, a.irn, a.d_array, rhs);
    }

    // Initialize MUMPS
    dmumps_par.comm_fortran = USE_COMM_WORLD;
    dmumps_par.job          = JOB_INIT;
    // Activate logs & metrics only on root rank
    dmumps_par.ICNTL(2)     = -1;

    // This parameter controls the involvement of the root rank in the factorization
    // and solve phase
    dmumps_par.par       = par;
    // This parameter controls the parallelism of factorization of the root node
    dmumps_par.ICNTL(13) = icntl_13;
    // This parameter explicitly request a number of OMP threads
    dmumps_par.ICNTL(16) = icntl_16;

    dmumps_c(&dmumps_par);

    // Define the system
    dmumps_par.n   = a.n;
    dmumps_par.nnz = a.nnz;
    dmumps_par.irn = a.irn;
    dmumps_par.jcn = a.jcn;
    dmumps_par.a   = a.d_array;
    dmumps_par.rhs = rhs;
    dmumps_par.sym = (int) a.spec;

    // Enables residual computation and printing
    dmumps_par.ICNTL(11) = 2;
    dmumps_par.ICNTL(13) = icntl_13;
    // This parameter controls the memory relaxation of the MUMPS during
    // factorisation. We need to increase it as some matrix needs large size of
    // temporary memory
    dmumps_par.ICNTL(14) = 30;
    dmumps_par.ICNTL(16) = icntl_16;

    // Launch MUMPS for Analysis, Factorisation
    dmumps_par.job = resolve ? 6 : 4;
    dmumps_c(&dmumps_par);
    if (dmumps_par.infog[0] != 0) {
        return EXIT_FAILURE;
    }

    // Deinitialize MUMPS
    dmumps_par.job = JOB_END;
    dmumps_c(&dmumps_par);

    if (rhs != nullptr) {
        free(rhs);
    }

    return EXIT_SUCCESS;
}

static int experiment_complex (const matrix_t a, const int par, const int icntl_13,
                               const int icntl_16, const bool resolve) {

    ZMUMPS_STRUC_C zmumps_par;

    ZMUMPS_COMPLEX *rhs = nullptr;
    if (resolve == true) {
        rhs = (typeof(rhs)) malloc(a.n * sizeof(*rhs));
        generate_rhs(a.n, a.nnz, a.irn, a.z_array, rhs);
    }

    // Initialize MUMPS
    zmumps_par.comm_fortran = USE_COMM_WORLD;
    zmumps_par.job          = JOB_INIT;
    // Activate logs & metrics only on root rank
    zmumps_par.ICNTL(2)     = -1;

    zmumps_par.par       = par;
    zmumps_par.ICNTL(13) = icntl_13;
    zmumps_par.ICNTL(16) = icntl_16;

    zmumps_c(&zmumps_par);

    // Define the system
    zmumps_par.n   = a.n;
    zmumps_par.nnz = a.nnz;
    zmumps_par.irn = a.irn;
    zmumps_par.jcn = a.jcn;
    zmumps_par.a   = a.z_array;
    zmumps_par.rhs = rhs;
    zmumps_par.sym = (int) a.spec;

    // Enables residual computation and printing
    zmumps_par.ICNTL(11) = 2;
    zmumps_par.ICNTL(13) = icntl_13;
    // This parameter controls the memory relaxation of the MUMPS during
    // factorisation. We need to increase it as some matrix needs large size of
    // temporary memory
    zmumps_par.ICNTL(14) = 25;
    zmumps_par.ICNTL(16) = icntl_16;

    // Launch MUMPS for Analysis, Factorisation and Resolution
    zmumps_par.job = resolve ? 6 : 4;
    zmumps_c(&zmumps_par);
    if (zmumps_par.infog[0] != 0) {
        return EXIT_FAILURE;
    }

    // Deinitialize MUMPS
    zmumps_par.job = JOB_END;
    zmumps_c(&zmumps_par);

    free(rhs);

    return EXIT_SUCCESS;
}

int run_experiment (const matrix_t a, const MUMPS_INT par, const int icntl_13,
                    const int icntl_16, const bool resolve) {

    if (a.type == real) {
        return experiment_real(a, par, icntl_13, icntl_16, resolve);
    }
    else {
        return experiment_complex(a, par, icntl_13, icntl_16, resolve);
    }
}
