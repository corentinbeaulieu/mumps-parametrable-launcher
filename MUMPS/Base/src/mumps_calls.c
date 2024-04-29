/** @file mumps_calls.c
 *  @brief Definition of the MUMPS lauching related functions
 */

#include "mumps_calls.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Fills the dmumps structure for MUMPS internal use
 *
 * @param[inout] info Caracteristic of the experiment containing MUMPS struct (see @ref
 * mumps_t)
 *
 * @return EXIT_SUCCES on success, EXIT_FAILURE otherwise
 */
static int _mumps_fill_struct_real (mumps_t info[static 1]) {

    // Define the system
    info->mumps_struct.dmumps_par.n   = info->a.n;
    info->mumps_struct.dmumps_par.nnz = info->a.nnz;
    info->mumps_struct.dmumps_par.irn = info->a.irn;
    info->mumps_struct.dmumps_par.jcn = info->a.jcn;
    info->mumps_struct.dmumps_par.a   = info->a.d_array;
    info->mumps_struct.dmumps_par.sym = (int) info->a.spec;

    // Activate logs & metrics only on root rank
    info->mumps_struct.dmumps_par.ICNTL(2)  = -1;
    info->mumps_struct.dmumps_par.ICNTL(5)  = 0;
    info->mumps_struct.dmumps_par.ICNTL(11) = 2;
    // Memory relaxation
    info->mumps_struct.dmumps_par.ICNTL(14) = 30;
    // Sequential ordering as default
    info->mumps_struct.dmumps_par.ICNTL(28) = 1;

    // Select ordering
    switch (info->partition_agent) {
        case Automatic:
            info->mumps_struct.dmumps_par.ICNTL(7) =
                7; // Automatic choice of sequential ordering
            info->mumps_struct.dmumps_par.ICNTL(28) =
                0; // Automatic choice between seq or par ordering
            info->mumps_struct.dmumps_par.ICNTL(29) =
                0; // Automatic choice of parallel ordering
            break;
        case Metis:
            info->mumps_struct.dmumps_par.ICNTL(7) = 5;
            break;
        case Pord:
            info->mumps_struct.dmumps_par.ICNTL(7) = 4;
            break;
        case Scotch:
            info->mumps_struct.dmumps_par.ICNTL(7) = 3;
            break;
        case PTScotch:
            info->mumps_struct.dmumps_par.ICNTL(28) = 2;
            info->mumps_struct.dmumps_par.ICNTL(29) = 1;
            break;
        default:
            return EXIT_FAILURE;
            break;
    }

    // This parameter controls the involvement of the root rank in the factorization
    // and solve phase
    info->mumps_struct.dmumps_par.par       = info->par;
    // This parameter controls the parallelism of factorization of the root node
    info->mumps_struct.dmumps_par.ICNTL(13) = info->icntl_13;
    // This parameter explicitly request a number of OMP threads
    info->mumps_struct.dmumps_par.ICNTL(16) = info->icntl_16;


    return EXIT_SUCCESS;
}

/**
 * @brief Fills the zmumps structure for MUMPS internal use
 *
 * @param[inout] info Caracteristic of the experiment containing MUMPS struct (see @ref
 * mumps_t)
 *
 * @return EXIT_SUCCES on success, EXIT_FAILURE otherwise
 */
static int _mumps_fill_struct_complex (mumps_t info[static 1]) {

    // Define the system
    info->mumps_struct.zmumps_par.n   = info->a.n;
    info->mumps_struct.zmumps_par.nnz = info->a.nnz;
    info->mumps_struct.zmumps_par.irn = info->a.irn;
    info->mumps_struct.zmumps_par.jcn = info->a.jcn;
    info->mumps_struct.zmumps_par.a   = info->a.z_array;
    info->mumps_struct.zmumps_par.sym = (int) info->a.spec;

    // Activate logs & metrics only on root rank
    info->mumps_struct.zmumps_par.ICNTL(2)  = -1;
    info->mumps_struct.zmumps_par.ICNTL(5)  = 0;
    info->mumps_struct.zmumps_par.ICNTL(11) = 2;
    // Memory relaxation
    info->mumps_struct.zmumps_par.ICNTL(14) = 30;
    // Sequential ordering as default
    info->mumps_struct.zmumps_par.ICNTL(28) = 1;

    // Select ordering
    switch (info->partition_agent) {
        case Automatic:
            info->mumps_struct.zmumps_par.ICNTL(7) =
                7; // Automatic choice of sequential ordering
            info->mumps_struct.zmumps_par.ICNTL(28) =
                0; // Automatic choice between seq or par ordering
            info->mumps_struct.zmumps_par.ICNTL(29) =
                0; // Automatic choice of parallel ordering
            break;
        case Metis:
            info->mumps_struct.zmumps_par.ICNTL(7) = 5;
            break;
        case Pord:
            info->mumps_struct.zmumps_par.ICNTL(7) = 4;
            break;
        case Scotch:
            info->mumps_struct.zmumps_par.ICNTL(7) = 3;
            break;
        case PTScotch:
            info->mumps_struct.zmumps_par.ICNTL(28) = 2;
            info->mumps_struct.zmumps_par.ICNTL(29) = 1;
            break;
        default:
            return EXIT_FAILURE;
            break;
    }

    // This parameter controls the involvement of the root rank in the factorization
    // and solve phase
    info->mumps_struct.zmumps_par.par       = info->par;
    // This parameter controls the parallelism of factorization of the root node
    info->mumps_struct.zmumps_par.ICNTL(13) = info->icntl_13;
    // This parameter explicitly request a number of OMP threads
    info->mumps_struct.zmumps_par.ICNTL(16) = info->icntl_16;


    return EXIT_SUCCESS;
}

/**
 * @brief Initializes the double variant of the MUMPS solver
 *
 * @param[inout] info Caracteristics of the experiment (see @ref mumps_t)
 *
 * @return EXIT_SUCCESS on success, EXIT_FAILURE otherwise
 */
static int _mumps_init_real (mumps_t info[static 1]) {
    int ret = EXIT_SUCCESS;

    ret = _mumps_fill_struct_real(info);

    // Initialize MUMPS
    info->mumps_struct.dmumps_par.comm_fortran = USE_COMM_WORLD;
    info->mumps_struct.dmumps_par.job          = JOB_INIT;

    if (ret == EXIT_SUCCESS) {
        dmumps_c(&info->mumps_struct.dmumps_par);
        if (info->mumps_struct.dmumps_par.infog[0] != 0) {
            ret = EXIT_FAILURE;
        }
    }

    return ret;
}

/**
 * @brief Initializes the double complex variant of the MUMPS solver
 *
 * @param[inout] info Caracteristics of the experiment (see @ref mumps_t)
 *
 * @return EXIT_SUCCESS on success, EXIT_FAILURE otherwise
 */
static int _mumps_init_complex (mumps_t info[static 1]) {
    int ret = EXIT_SUCCESS;

    ret = _mumps_fill_struct_complex(info);

    // Initialize MUMPS
    info->mumps_struct.zmumps_par.comm_fortran = USE_COMM_WORLD;
    info->mumps_struct.zmumps_par.job          = JOB_INIT;

    if (ret == EXIT_SUCCESS) {
        zmumps_c(&info->mumps_struct.zmumps_par);
        if (info->mumps_struct.zmumps_par.infog[0] != 0) {
            ret = EXIT_FAILURE;
        }
    }

    return ret;
}

/**
 * @brief Initializes MUMPS internals based on the experiment characteristics
 *
 * @param[inout] info  Caracteristics of the experiment (see @ref mumps_t)
 *
 * @return      EXIT_SUCCESS on success, EXIT_FAILURE otherwise
 */
int mumps_init (mumps_t info[static 1]) {
    int ret = EXIT_SUCCESS;

    switch (info->a.type) {
        case real:
            ret = _mumps_init_real(info);
            break;
        case complex_number:
            ret = _mumps_init_complex(info);
            break;
        default:
            ret = EXIT_FAILURE;
            break;
    }

    return ret;
}

/**
 * @brief Launches the Analysis stage of MUMPS for real numbers matrix
 *
 * @param[inout] info Characteristics of the experiment (see @ref mumps_t)
 *
 * @return EXIT_SUCCESS on success, EXIT_FAILURE otherwise
 */
static int _mumps_run_ana_real (mumps_t info[static 1]) {
    int ret = EXIT_SUCCESS;

    ret                               = _mumps_fill_struct_real(info);
    info->mumps_struct.dmumps_par.job = JOB_ANA;

    dmumps_c(&info->mumps_struct.dmumps_par);
    if (info->mumps_struct.dmumps_par.infog[0] != 0) {
        ret = EXIT_FAILURE;
    }

    return ret;
}

/**
 * @brief Launches the Analysis stage of MUMPS for complex numbers matrix
 *
 * @param[inout] info Characteristics of the experiment (see @ref mumps_t)
 *
 * @return EXIT_SUCCESS on success, EXIT_FAILURE otherwise
 */
static int _mumps_run_ana_complex (mumps_t info[static 1]) {
    int ret = EXIT_SUCCESS;

    ret                               = _mumps_fill_struct_complex(info);
    info->mumps_struct.zmumps_par.job = JOB_ANA;

    zmumps_c(&info->mumps_struct.zmumps_par);
    if (info->mumps_struct.zmumps_par.infog[0] != 0) {
        ret = EXIT_FAILURE;
    }

    return ret;
}

/**
 * @brief Runs MUMPS analysis phase based on the experiment characteristics
 *
 * The function calls either @ref _mumps_run_ana_real or @ref _mumps_run_ana_complex
 * based on the type of the elements in the matrix to analyze.
 *
 * @param[inout] info  Caracteristics of the experiment (see @ref mumps_t)
 *
 * @return      EXIT_SUCCESS on success, EXIT_FAILURE otherwise
 */
int mumps_run_ana (mumps_t info[static 1]) {
    int ret = EXIT_SUCCESS;

    switch (info->a.type) {
        case real:
            ret = _mumps_run_ana_real(info);
            break;
        case complex_number:
            ret = _mumps_run_ana_complex(info);
            break;
        default:
            ret = EXIT_FAILURE;
            break;
    }

    return ret;
}

/**
 * @brief Launches the Factorisation stage of MUMPS for real numbers matrix
 *
 * @param[inout] info Characteristics of the experiment (see @ref mumps_t)
 *
 * @return EXIT_SUCCESS on success, EXIT_FAILURE otherwise
 */
static int _mumps_run_facto_real (mumps_t info[static 1]) {
    int ret = EXIT_SUCCESS;

    info->mumps_struct.dmumps_par.job = JOB_FACTO;

    dmumps_c(&info->mumps_struct.dmumps_par);
    if (info->mumps_struct.dmumps_par.infog[0] != 0) {
        ret = EXIT_FAILURE;
    }

    return ret;
}

/**
 * @brief Launches the Factorisation stage of MUMPS for complex numbers matrix
 *
 * @param[inout] info Characteristics of the experiment (see @ref mumps_t)
 *
 * @return EXIT_SUCCESS on success, EXIT_FAILURE otherwise
 */
static int _mumps_run_facto_complex (mumps_t info[static 1]) {
    int ret = EXIT_SUCCESS;

    info->mumps_struct.zmumps_par.job = JOB_FACTO;

    zmumps_c(&info->mumps_struct.zmumps_par);
    if (info->mumps_struct.zmumps_par.infog[0] != 0) {
        ret = EXIT_FAILURE;
    }

    return ret;
}

/**
 * @brief Runs MUMPS factorisation phase based on the experiment characteristics
 *
 * The function calls either @ref _mumps_run_facto_real or @ref _mumps_run_facto_complex
 * based on the type of the elements in the matrix to factorize.
 *
 * @param[inout] info  Caracteristics of the experiment (see @ref mumps_t)
 *
 * @return      EXIT_SUCCESS on success, EXIT_FAILURE otherwise
 */
int mumps_run_facto (mumps_t info[static 1]) {
    int ret = EXIT_SUCCESS;

    switch (info->a.type) {
        case real:
            ret = _mumps_run_facto_real(info);
            break;
        case complex_number:
            ret = _mumps_run_facto_complex(info);
            break;
        default:
            ret = EXIT_FAILURE;
            break;
    }

    return ret;
}

/**
 * @brief Launches the Resolution stage of MUMPS for real numbers matrix
 *
 * @param[inout] info Characteristics of the experiment (see @ref mumps_t)
 *
 * @warning A rhs must be provided in @p info for this stage to complete
 *
 * @return EXIT_SUCCESS on success, EXIT_FAILURE otherwise
 */
static int _mumps_run_res_real (mumps_t info[static 1]) {
    int ret = EXIT_SUCCESS;

    const DMUMPS_REAL *rhs = (typeof(rhs)) malloc(info->a.n * sizeof(*rhs));

    info->mumps_struct.dmumps_par.job = JOB_RES;

    dmumps_c(&info->mumps_struct.dmumps_par);
    if (info->mumps_struct.dmumps_par.infog[0] != 0) {
        ret = EXIT_FAILURE;
    }

    free((void *) rhs);

    return ret;
}

/**
 * @brief Launches the Resolution stage of MUMPS for complex numbers matrix
 *
 * @param[inout] info Characteristics of the experiment (see @ref mumps_t)
 *
 * @warning A rhs must be provided in @p info for this stage to complete
 *
 * @return EXIT_SUCCESS on success, EXIT_FAILURE otherwise
 */
static int _mumps_run_res_complex (mumps_t info[static 1]) {
    int ret = EXIT_SUCCESS;

    info->mumps_struct.zmumps_par.job = JOB_RES;

    zmumps_c(&info->mumps_struct.zmumps_par);
    if (info->mumps_struct.zmumps_par.infog[0] != 0) {
        ret = EXIT_FAILURE;
    }

    return ret;
}

/**
 * @brief Runs MUMPS resolution phase based on the experiment characteristics
 *
 * The function calls either @ref _mumps_run_res_real or @ref _mumps_run_res_complex
 * based on the type of the elements in the system to resolve.
 *
 * @param[inout] info  Caracteristics of the experiment (see @ref mumps_t)
 *
 * @return      EXIT_SUCCESS on success, EXIT_FAILURE otherwise
 */
int mumps_run_res (mumps_t info[static 1]) {
    int ret = EXIT_SUCCESS;

    switch (info->a.type) {
        case real:
            ret = _mumps_run_res_real(info);
            break;
        case complex_number:
            ret = _mumps_run_res_complex(info);
            break;
        default:
            ret = EXIT_FAILURE;
            break;
    }

    return ret;
}

/**
 * @brief Frees the internals of a double precision real MUMPS instance
 *
 * @param[inout] info Characteristics of the experiment (see @ref mumps_t)
 *
 * @return EXIT_SUCCESS on success, EXIT_FAILURE otherwise
 */
static int _mumps_finalize_real (mumps_t info[static 1]) {
    int ret = EXIT_SUCCESS;

    // Deinitialize MUMPS
    info->mumps_struct.dmumps_par.job = JOB_END;
    dmumps_c(&info->mumps_struct.dmumps_par);
    if (info->mumps_struct.dmumps_par.infog[0] != 0) {
        ret = EXIT_FAILURE;
    }

    return ret;
}

/**
 * @brief Frees the internals of a double precision complex MUMPS instance
 *
 * @param[inout] info Characteristics of the experiment (see @ref mumps_t)
 *
 * @return EXIT_SUCCESS on success, EXIT_FAILURE otherwise
 */
static int _mumps_finalize_complex (mumps_t info[static 1]) {
    int ret = EXIT_SUCCESS;

    // Deinitialize MUMPS
    info->mumps_struct.zmumps_par.job = JOB_END;
    zmumps_c(&info->mumps_struct.zmumps_par);
    if (info->mumps_struct.zmumps_par.infog[0] != 0) {
        ret = EXIT_FAILURE;
    }

    return ret;
}

/**
 * @brief Frees MUMPS internals based on the experiment characteristics
 *
 * The function calls either @ref _mumps_finalize_real or @ref _mumps_finalize_complex
 * based on the type of the elements in the system.
 *
 * @param[inout] info  Caracteristics of the experiment (see @ref mumps_t)
 *
 * @return      EXIT_SUCCESS on success, EXIT_FAILURE otherwise
 */
int mumps_finalize (mumps_t info[static 1]) {
    int ret = EXIT_SUCCESS;

    switch (info->a.type) {
        case real:
            ret = _mumps_finalize_real(info);
            break;
        case complex_number:
            ret = _mumps_finalize_complex(info);
            break;
        default:
            ret = EXIT_FAILURE;
            break;
    }

    return ret;
}

/**
 * @brief Saves the internals of a double precision real MUMPS instance
 *
 * @param[in] info     Characteristics of the experiment (see @ref mumps_t)
 * @param[in] nb_char  Length of @p exp_name
 * @param[in] exp_name Name of the experiment. Used to name saved files.
 *
 * @return EXIT_SUCCESS on success, EXIT_FAILURE otherwise
 */
static int _mumps_save_real (mumps_t info[static 1], const size_t nb_char,
                             const char exp_name[static nb_char]) {
    int ret = EXIT_SUCCESS;

    strncpy(info->mumps_struct.dmumps_par.save_dir, "Saved_analysis", 15);
    strncpy(info->mumps_struct.dmumps_par.save_prefix, exp_name, nb_char);
    info->mumps_struct.dmumps_par.job = JOB_SAVE;

    dmumps_c(&info->mumps_struct.dmumps_par);
    if (info->mumps_struct.dmumps_par.infog[0] != 0) {
        ret = EXIT_FAILURE;
    }

    return ret;
}

/**
 * @brief Saves the internals of a double precision complex MUMPS instance
 *
 * @param[in] info     Characteristics of the experiment (see @ref mumps_t)
 * @param[in] nb_char  Length of @p exp_name
 * @param[in] exp_name Name of the experiment. Used to name saved files.
 *
 * @return EXIT_SUCCESS on success, EXIT_FAILURE otherwise
 */
static int _mumps_save_complex (mumps_t info[static 1], const size_t nb_char,
                                const char exp_name[static nb_char]) {
    int ret = EXIT_SUCCESS;

    strncpy(info->mumps_struct.dmumps_par.save_dir, "Saved_analysis", 15);
    strncpy(info->mumps_struct.zmumps_par.save_prefix, exp_name, nb_char);
    info->mumps_struct.zmumps_par.job = JOB_SAVE;

    zmumps_c(&info->mumps_struct.zmumps_par);
    if (info->mumps_struct.zmumps_par.infog[0] != 0) {
        ret = EXIT_FAILURE;
    }

    return ret;
}

/**
 * @brief Saves the current MUMPS internals for latter restore
 *
 * The function calls either @ref _mumps_save_real or @ref _mumps_save_complex
 * based on the type of the elements in the system.
 *
 * @param[in] info     Caracteristics of the experiment (see @ref mumps_t)
 * @param[in] nb_char  Size of @p exp_name
 * @param[in] exp_name Unique name of the experiment. Used to name the files.
 *
 * @return      EXIT_SUCCESS on success, EXIT_FAILURE otherwise
 */
int mumps_save (mumps_t info[static 1]) {
    int ret = EXIT_SUCCESS;

    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    char name[256] = { 0 };
    snprintf(name, 256, "exp-%d-%ld-%d-%d-%dx%d-%d", info->a.n, info->a.nnz,
             info->a.spec, info->partition_agent, size, info->icntl_16, info->par);

    switch (info->a.type) {
        case real:
            ret = _mumps_save_real(info, 256, name);
            break;
        case complex_number:
            ret = _mumps_save_complex(info, 256, name);
            break;
        default:
            ret = EXIT_FAILURE;
            break;
    }

    return ret;
}

/**
 * @brief Restores a saved double precision real MUMPS instance from a prior save
 *
 * @param[inout] info  Caracteristics of the experiment (see @ref mumps_t)
 * @param[in] nb_char  Size of @p exp_name
 * @param[in] exp_name Unique name of the experiment. Used to name the files.
 *
 * @return      EXIT_SUCCESS on success, EXIT_FAILURE otherwise
 */
static int _mumps_restore_real (mumps_t info[static 1], const size_t nb_char,
                                const char exp_name[static nb_char]) {
    int ret = EXIT_SUCCESS;

    strncpy(info->mumps_struct.dmumps_par.save_dir, "Saved_analysis", 15);
    strncpy(info->mumps_struct.dmumps_par.save_prefix, exp_name, nb_char);
    info->mumps_struct.dmumps_par.job = JOB_RESTORE;

    dmumps_c(&info->mumps_struct.dmumps_par);
    if (info->mumps_struct.dmumps_par.infog[0] != 0) {
        ret = EXIT_FAILURE;
    }

    return ret;
}

/**
 * @brief Restores a saved double precision complex MUMPS instance from a prior save
 *
 * @param[inout] info  Caracteristics of the experiment (see @ref mumps_t)
 * @param[in] nb_char  Size of @p exp_name
 * @param[in] exp_name Unique name of the experiment. Used to name the files.
 *
 * @return      EXIT_SUCCESS on success, EXIT_FAILURE otherwise
 */
static int _mumps_restore_complex (mumps_t info[static 1], const size_t nb_char,
                                   const char exp_name[static nb_char]) {
    int ret = EXIT_SUCCESS;

    strncpy(info->mumps_struct.dmumps_par.save_dir, "Saved_analysis", 15);
    strncpy(info->mumps_struct.zmumps_par.save_prefix, exp_name, nb_char);
    info->mumps_struct.zmumps_par.job = JOB_RESTORE;

    zmumps_c(&info->mumps_struct.zmumps_par);
    if (info->mumps_struct.zmumps_par.infog[0] != 0) {
        ret = EXIT_FAILURE;
    }

    return ret;
}

/**
 * @brief Restores a saved MUMPS state from a prior save
 *
 * The function calls either @ref _mumps_restore_real or @ref _mumps_restore_complex
 * based on the type of the elements in the system.
 *
 * @param[inout] info  Caracteristics of the experiment (see @ref mumps_t)
 *
 * @return      EXIT_SUCCESS on success, EXIT_FAILURE otherwise
 */
int mumps_restore (mumps_t info[static 1]) {
    int ret = EXIT_SUCCESS;

    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    char name[256] = { 0 };
    snprintf(name, 256, "exp-%d-%ld-%d-%d-%dx%d-%d", info->a.n, info->a.nnz,
             info->a.spec, info->partition_agent, size, info->icntl_16, info->par);


    switch (info->a.type) {
        case real:
            ret = _mumps_fill_struct_real(info);
            if (ret != EXIT_SUCCESS) {
                break;
            }
            ret = _mumps_restore_real(info, 256, name);
            break;
        case complex_number:
            ret = _mumps_fill_struct_complex(info);
            if (ret != EXIT_SUCCESS) {
                break;
            }
            ret = _mumps_restore_complex(info, 256, name);
            break;
        default:
            ret = EXIT_FAILURE;
            break;
    }

    return ret;
}
