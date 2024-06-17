/** @file mumps_calls.h
 *
 *  @brief Contains the MUMPS related launching functions
 */

#pragma once
#include "utils.h"


/** @brief Phase MUMPS should do */
enum job_e : int {
    JOB_INIT          = -1, /*!< Initialize MUMPS internals */
    JOB_END           = -2, /*!< Free all MUMPS internals */
    JOB_ANA           = 1,  /*!< Perform the analysis phase only */
    JOB_FACTO         = 2,  /*!< Perform the factorisation phase only */
    JOB_RES           = 3,  /*!< Perform the resolution phase only */
    JOB_ANA_FACTO     = 4,  /*!< Perform analysis and factorisation phase */
    JOB_FACT_RES      = 5,  /*!< Perform factorisation and resolution phase */
    JOB_ANA_FACTO_RES = 6,  /*!< Perform analysis, factorisation and resolution phase */
    JOB_SAVE          = 7,  /*!< Save current state */
    JOB_RESTORE       = 8,  /*!< Restore past state */
};

/** @brief Possible ordering strategies allowed */
typedef enum : unsigned char {
    Automatic = 0, /*!< MUMPS choose the ordering */
    Metis,         /*!< Use MeTiS */
    Pord,          /*!< Use Pord */
    Scotch,        /*!< Use SCOTCH */
    PTScotch,      /*!< Use PT-SCOTCH */
} partition_agent_t;

/** @brief Experiment characteristics */
typedef struct {
    union {
        DMUMPS_STRUC_C dmumps_par; /*!< MUMPS structure for double precision floats */
        ZMUMPS_STRUC_C zmumps_par; /*!< MUMPS structure for double precision complex */
    } mumps_struct;                /*!< MUMPS structure */
    matrix_t a;                    /*!< Matrix structure (see @ref matrix_t) */
    double *restrict rhs;          /*!< Right Hand Side for resolution if needed */
    int               par;         /*!< Value of the PAR parameter */
    int               icntl_13;    /*!< Value of ICNTL(13) parameter */
    int               icntl_16;    /*!< Value of ICNTL(16) parameter */
    partition_agent_t partition_agent; /*!< Which ordering library to use (see @ref
                                          partition_agent_t) */
} mumps_t;

int mumps_init (mumps_t info[static 1]);

int mumps_run_ana (mumps_t info[static 1]);

int mumps_run_facto (mumps_t info[static 1]);

int mumps_run_res (mumps_t info[static 1]);

int mumps_finalize (mumps_t info[static 1]);

int mumps_save (mumps_t info[static 1], const size_t nb_char,
                const char name[restrict static nb_char]);

int mumps_restore (mumps_t info[static 1], const size_t nb_char,
                   const char name[restrict static nb_char]);

// The following is at the end to disable some weird doxygen behaviour
/** @def USE_COMM_WORLD
 * @brief Special MUMPS value to use MPI_COMM_WORLD communicator
 * */
#define USE_COMM_WORLD (-987'654)

// constexpr int USE_COMM_WORLD = -987'654;
