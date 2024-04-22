/** @file mumps_calls.c
 *  @brief Definition of the MUMPS lauching related functions
 */

#include "mumps_calls.h"
#include <stdlib.h>

/**
 * @brief Launch MUMPS on a real matrix
 *
 * @param[in] a                 Real matrix to factorize
 * @param[in] par               MUMPS PAR parameter controlling the involvement of the
 * root process in factorization
 * @param[in] icntl_13          MUMPS ICNTL(13) parameter controlling the factorization
 * of the root node
 * @param[in] icntl_16          MUMPS ICNTL(16) parameter controlling the number of
 * threads
 * @param[in] resolve           Generate a rhs and do the resolution stage if true only
 *   factorize otherwise
 * @param[in] partition_agent   Which ordering library to use (see refer to @ref
 * partition_agent_t)
 *
 * @return    EXIT_SUCCESS on success, EXIT_FAILURE otherwise
 */
static int experiment_real (const matrix_t a, const int par, const int icntl_13,
                            const int icntl_16, const bool resolve,
                            const partition_agent_t partition_agent) {

    DMUMPS_STRUC_C dmumps_par;

    DMUMPS_REAL *rhs = nullptr;

    if (resolve == true) {
        rhs = (typeof(rhs)) malloc(a.n * sizeof(*rhs));
        generate_rhs(a.n, a.nnz, a.irn, a.d_array, rhs);
    }

    // Define the system
    dmumps_par.n   = a.n;
    dmumps_par.nnz = a.nnz;
    dmumps_par.irn = a.irn;
    dmumps_par.jcn = a.jcn;
    dmumps_par.a   = a.d_array;
    dmumps_par.rhs = rhs;
    dmumps_par.sym = (int) a.spec;

    dmumps_par.ICNTL(2) = -1;
    dmumps_par.ICNTL(5) = 0;

    // Initialize MUMPS
    dmumps_par.comm_fortran = USE_COMM_WORLD;
    dmumps_par.job          = JOB_INIT;
    // Activate logs & metrics only on root rank
    dmumps_par.ICNTL(2)     = -1;
    dmumps_par.ICNTL(5)     = 0;
    // Memory relaxation
    dmumps_par.ICNTL(14)    = 30;

    // Sequential ordering as default
    dmumps_par.ICNTL(28) = 1;
    switch (partition_agent) {
        case Automatic:
            dmumps_par.ICNTL(7)  = 7; // Automatic choice of sequential ordering
            dmumps_par.ICNTL(28) = 0; // Automatic choice between seq or par ordering
            dmumps_par.ICNTL(29) = 0; // Automatic choice of parallel ordering
            break;
        case Metis:
            dmumps_par.ICNTL(7) = 5;
            break;
        case Pord:
            dmumps_par.ICNTL(7) = 4;
            break;
        case Scotch:
            dmumps_par.ICNTL(7) = 3;
            break;
        case PTScotch:
            dmumps_par.ICNTL(28) = 2;
            dmumps_par.ICNTL(29) = 1;
            break;
    }

    // This parameter controls the involvement of the root rank in the factorization
    // and solve phase
    dmumps_par.par       = par;
    // This parameter controls the parallelism of factorization of the root node
    dmumps_par.ICNTL(13) = icntl_13;
    // This parameter explicitly request a number of OMP threads
    dmumps_par.ICNTL(16) = icntl_16;

    dmumps_c(&dmumps_par);

    // This parameter controls the involvement of the root rank in the factorization
    // and solve phase
    dmumps_par.par = par;
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

    // Define Ordonning strategy
    dmumps_par.ICNTL(28) = 1;
    switch (partition_agent) {
        case Automatic:
            dmumps_par.ICNTL(7)  = 7; // Automatic choice of sequential ordering
            dmumps_par.ICNTL(28) = 0; // Automatic choice between seq or par ordering
            dmumps_par.ICNTL(29) = 0; // Automatic choice of parallel ordering
            break;
        case Metis:
            dmumps_par.ICNTL(7) = 5;
            break;
        case Pord:
            dmumps_par.ICNTL(7) = 4;
            break;
        case Scotch:
            dmumps_par.ICNTL(7) = 3;
            break;
        case PTScotch:
            dmumps_par.ICNTL(28) = 2;
            dmumps_par.ICNTL(29) = 1;
            break;
    }
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

/**
 * @brief Launch MUMPS on a complex matrix
 *
 * @param[in] a                 Complex matrix to factorize
 * @param[in] par               MUMPS PAR parameter controlling the involvement of the
 * root process in factorization
 * @param[in] icntl_13          MUMPS ICNTL(13) parameter controlling the factorization
 * of the root node
 * @param[in] icntl_16          MUMPS ICNTL(16) parameter controlling the number of
 * threads
 * @param[in] resolve           Generate a rhs and do the resolution stage if true only
 * factorize otherwise
 * @param[in] partition_agent   Which ordering library to use (see refer to @ref
 * partition_agent_t)
 *
 * @return    EXIT_SUCCESS on success, EXIT_FAILURE otherwise
 */
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

/**
 * @brief Launch MUMPS on a matrix
 *
 * This function calls @ref run_experiment_real or @ref run_experiment_complex
 * based on the type of the elements of the matrix
 *
 * @param[in] a                 Matrix to factorize
 * @param[in] par               MUMPS PAR parameter controlling the involvement of the
 * root process in factorization
 * @param[in] icntl_13          MUMPS ICNTL(13) parameter controlling the factorization
 * of the root node
 * @param[in] icntl_16          MUMPS ICNTL(16) parameter controlling the number of
 * threads
 * @param[in] resolve           Generate a rhs and do the resolution stage if true only
 * factorize otherwise
 * @param[in] partition_agent   Which ordering library to use (see refer to @ref
 * partition_agent_t)
 *
 * @return    EXIT_SUCCESS on success, EXIT_FAILURE otherwise
 */
int run_experiment (const matrix_t a, const MUMPS_INT par, const int icntl_13,
                    const int icntl_16, const bool resolve,
                    const partition_agent_t partition_agent) {

    if (a.type == real) {
        return experiment_real(a, par, icntl_13, icntl_16, resolve, partition_agent);
    }
    else {
        return experiment_complex(a, par, icntl_13, icntl_16, resolve);
    }
}
