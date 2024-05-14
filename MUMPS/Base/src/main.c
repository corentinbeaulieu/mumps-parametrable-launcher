/** @mainpage MUMPS Parametrable launcher
 *
 * This project aims at providing a parametrable executable to measure mumps
 * execution with different parameters. The main goal is to use machine learning methods
 * to predict the best parameters to speedup the resolution of a linear system with
 * MUMPS
 */

/** @file main.c
 *
 *  Contains the main function of the program which
 *  - check the inputs
 *  - whether read a matrix or generate it
 *  - call mumps to factorize it
 */
#define _GNU_SOURCE
#include <err.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <mpi.h>
#include <omp.h>

#ifdef USE_EAR
#include <ear.h>
#endif

#include "mumps_calls.h"


int main (int argc, char *argv[]) {

    matrix_t a;
    int      ierr, rank, size;

    // Initialize MPI
    ierr = MPI_Init(&argc, &argv);
    if (ierr != MPI_SUCCESS) {
        perror("ERROR");
        return EXIT_FAILURE;
    }
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (ierr != MPI_SUCCESS) {
        perror("ERROR");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (ierr != MPI_SUCCESS) {
        perror("ERROR");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    char opt      = -1;
    bool readfile = false;
    int  state    = 0;
    bool analysis = false;
    bool facto    = false;
    bool resolve  = false;
    bool global   = false;

    // Get the options
    while ((opt = getopt(argc, argv, "ighfars:")) != -1) {
        switch (opt) {
            case 'h': // Print help and exit
                print_help(argv[0]);
                return EXIT_SUCCESS;
            case 'i': // Don't use the generator
                readfile = true;
                break;
            case 'a': // Do the Analysis stage in MUMPS
                analysis = true;
                break;
            case 'f': // Do the Factorization stage in MUMPS
                facto = true;
                break;
            case 'r': // Do the Resolve stage in MUMPS
                resolve = true;
                break;
            case 'g': // Density is global instead of bandwide
                global = true;
                break;
            case 's': // Provide a seed for the generator
                state = (int) atoi(optarg);
                break;
            default:
                print_help(argv[0]);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    if (facto == false && resolve == true) {
        fprintf(stderr, "ERROR: The factorization must be performed in order to "
                        "resolve the system\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (analysis == false && facto == false && resolve == false) {
        analysis = true;
        facto    = true;
    }

    char name[256] = { 0 };

    if (readfile == true) {

        if ((argc - optind) != 5) {
            print_help(argv[0]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        // Read and parse the input matrix
        parse_error_t perr = parseMTX(argv[optind], &a);
        switch (perr) {
            case NotMatrixMarket:
                fprintf(stderr, "Not a Matrix Market input!\n");
                return perr;
            case Unknown:
                fprintf(stderr, "Second description term must be matrix\n");
                return perr;
            case UnknownType:
                fprintf(stderr, "Only real/complex_number numbers are supported\n");
                return perr;
            case UnknownFormat:
                fprintf(stderr, "Only coordinate format is allowed\n");
                return perr;
            case UnknownSpecificity:
                fprintf(stderr,
                        "The form of the matrix must be general or symmetric\n");
                return perr;

            case FileError:
                err(perr, "ERROR: %s", argv[optind]);
                return perr;

            case MallocError:
                perror("ERROR");
                return perr;

            case Success:
                break;
        }

        char *prev, *curr;
        curr = strtok(argv[optind], "/");
        while (curr != NULL) {
            prev = curr;
            curr = strtok(NULL, "/");
        }
        snprintf(name, 128, "exp-%s", strtok(prev, "."));

        fprintf(stderr, "[LOG] %s\n", name);

        optind++;
    }
    else {

        if ((argc - optind) != 9) {
            print_help(argv[0]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        a.type = (typeof(a.type)) atoi(argv[optind++]);
        a.n    = (MUMPS_INT) atoi(argv[optind++]);

        const double bandwidth_ratio = (typeof(bandwidth_ratio)) atof(argv[optind++]);
        if ((bandwidth_ratio > 1.0) || (bandwidth_ratio <= 0.0)) {
            fprintf(stderr, "\x1b[31mERROR\x1b[0m Wrong bandwidth provided\n");
            print_help(argv[0]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        const MUMPS_INT bandwidth = (MUMPS_INT) (bandwidth_ratio * (double) a.n);

        const double density = (typeof(density)) atof(argv[optind++]);
        if ((density > 1.0) || (density <= 0.0)) {
            fprintf(stderr, "\x1b[31mERROR\x1b[0m Wrong density provided\n");
            print_help(argv[0]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        a.spec = (typeof(a.spec)) atoi(argv[optind++]);


        int64_t               *ptr         = nullptr;
        enum spral_matrix_type matrix_type = SPRAL_MATRIX_UNSPECIFIED;

        const int64_t max_nnz_band = bandwidth * (2lu * a.n - bandwidth - 1lu) + a.n;

        // The generator returns an error (-3) if nnz is too high in regard of n.
        // The maximal value of nnz depends on whether the matrix is symmetric or not
        int64_t max_nnz = global
                              ? a.n * a.n //*< Number of elements in the dense matrix
                              : max_nnz_band; //*< Number of elements in the dense band

        // Translate input parameters into spral enum
        switch (a.spec) {
            case Unsymmetric:
                matrix_type = SPRAL_MATRIX_REAL_UNSYM;
                break;
            case Symmetric:
                max_nnz     = (max_nnz + a.n) / 2;
                matrix_type = SPRAL_MATRIX_REAL_SYM_INDEF;
                break;
            case SymmetricPositiveDefinite:
                max_nnz     = (max_nnz + a.n) / 2;
                matrix_type = SPRAL_MATRIX_REAL_SYM_PSDEF;
                break;
        }
        a.nnz = (typeof(a.nnz)) (density * (double) max_nnz);

        if (a.n > a.nnz) {
            a.nnz = a.n + 1;
        }
        if (a.nnz > max_nnz_band) {
            a.nnz = max_nnz_band - 1;
        }

        if (rank == 0) {
            // Allocate the various vectors
            ptr = (typeof(ptr)) malloc((a.n + 1) * sizeof(*ptr));

            a.irn = (typeof(a.irn)) malloc(a.nnz * sizeof(*a.irn));
            a.jcn = (typeof(a.jcn)) malloc(a.nnz * sizeof(*a.jcn));

            if (a.type == real) {
                a.d_array = (typeof(a.d_array)) malloc(a.nnz * sizeof(*a.d_array));

                ierr = spral_random_matrix_generate_band_long(
                    &state, matrix_type, a.n, a.n, a.nnz, bandwidth, ptr, a.irn,
                    (double *) a.d_array,
                    SPRAL_RANDOM_MATRIX_FINDEX | SPRAL_RANDOM_MATRIX_NONSINGULAR
                        | SPRAL_RANDOM_MATRIX_SORT);
            }
            // FIXME: complex_number matrix generation not supported by spral
            else {
                matrix_type = -matrix_type;
                a.z_array   = (typeof(a.z_array)) malloc(a.nnz * sizeof(*a.z_array));

                ierr = 0;
                ierr = spral_random_matrix_generate_band_long(
                    &state, matrix_type, a.n, a.n, a.nnz, bandwidth, ptr, a.irn,
                    (double *) a.z_array,
                    SPRAL_RANDOM_MATRIX_FINDEX | SPRAL_RANDOM_MATRIX_NONSINGULAR
                        | SPRAL_RANDOM_MATRIX_SORT);
            }

            // Check the generator return code
            if (ierr != 0) {
                fprintf(stderr,
                        "\x1b[31mERROR\x1b[0m The matrix generation failed\tERROR "
                        "CODE: %d\n",
                        ierr);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }

            conversion_CSC_to_COO(a.nnz, a.n, ptr, a.jcn);
            free(ptr);

        }
        snprintf(name, 128, "exp-%d-%lf-%lf-%d", a.n, bandwidth_ratio, density,
                a.spec);
    }

    mumps_t info = {
        .a               = a,
        .rhs             = nullptr,
        .par             = (int) atoi(argv[optind]),
        .icntl_13        = (MUMPS_INT) atoi(argv[optind + 1]),
        .icntl_16        = (MUMPS_INT) atoi(argv[optind + 2]),
        .partition_agent = (int) atoi(argv[optind + 3]),
    };

    {
        char buf[128] = {0};
        snprintf(buf, 128, "_%dx%d-%d-%d-%d", size, info.icntl_13,
                 info.par, info.icntl_16, info.partition_agent);
        strncat(name, buf, 128);
    }

    if(mumps_init(&info) != EXIT_SUCCESS) goto cleanup;

#ifdef USE_EAR
    // EAR energy measurement setup
    bool ear_ok = true;
    unsigned long e_mj = 0, e_mj_start = 0, e_mj_end = 0;
    unsigned long t_ms = 0, t_ms_start = 0, t_ms_end = 0;

    if (ear_connect() != EAR_SUCCESS) {
        ear_ok = false;
    }
    if ((ear_ok == true) && (ear_energy(&e_mj_start, &t_ms_start) != EAR_SUCCESS)) {
        ear_ok = false;
    }
#endif

    if ((analysis == true) || (readfile == true)) {
        if(mumps_run_ana(&info) != EXIT_SUCCESS) goto cleanup_full;

#ifdef USE_EAR
        if ((ear_ok == true) && (ear_energy(&e_mj_end, &t_ms_end) != EAR_SUCCESS)) {
            ear_ok = false;
        }

        if (ear_ok == true) {
            ear_energy_diff(e_mj_start, e_mj_end, &e_mj, t_ms_start, t_ms_end, &t_ms);
            if (rank == 0) {
                printf("ANALYSIS EAR ENERGY = %16lu mJ, TIME = %16lu ms, POWER = %16.6lf W\n",
                        e_mj,
                        t_ms,
                        (double) e_mj / (double) t_ms);
            }
        }
        if ((ear_ok == true) && (ear_energy(&e_mj_start, &t_ms_start) != EAR_SUCCESS)) {
            ear_ok = false;
        }
#endif

    }
    else if (mumps_restore(&info, 256, name) != EXIT_SUCCESS) {
        if(mumps_run_ana(&info) != EXIT_SUCCESS) goto cleanup_full;
    }

    mumps_save(&info, 256, name);

    if (facto == true) {
        if(mumps_run_facto(&info) != EXIT_SUCCESS) goto cleanup_full;

#ifdef USE_EAR
        if ((ear_ok == true) && (ear_energy(&e_mj_end, &t_ms_end) != EAR_SUCCESS)) {
            ear_ok = false;
        }

        if (ear_ok == true) {
            ear_energy_diff(e_mj_start, e_mj_end, &e_mj, t_ms_start, t_ms_end, &t_ms);
            if (rank == 0) {
                printf("FACTORIZATION EAR ENERGY = %16lu mJ, TIME = %16lu ms, POWER = %16.6lf W\n",
                        e_mj,
                        t_ms,
                        (double) e_mj / (double) t_ms);
            }
        }
        if ((ear_ok == true) && (ear_energy(&e_mj_start, &t_ms_start) != EAR_SUCCESS)) {
            ear_ok = false;
        }
#endif

    }

    if (resolve == true) {
        mumps_run_res(&info);

#ifdef USE_EAR
        if ((ear_ok == true) && (ear_energy(&e_mj_end, &t_ms_end) != EAR_SUCCESS)) {
            ear_ok = false;
        }

        if (ear_ok == true) {
            ear_energy_diff(e_mj_start, e_mj_end, &e_mj, t_ms_start, t_ms_end, &t_ms);
            if (rank == 0) {
                printf("RESOLVE EAR ENERGY = %16lu mJ, TIME = %16lu ms, POWER = %16.6lf W\n",
                        e_mj,
                        t_ms,
                        (double) e_mj / (double) t_ms);
            }
        }
        if ((ear_ok == true) && (ear_energy(&e_mj_start, &t_ms_start) != EAR_SUCCESS)) {
            ear_ok = false;
        }
#endif

    }
cleanup_full:
    mumps_finalize(&info);

cleanup:
    ierr = MPI_Finalize();
    if (ierr != MPI_SUCCESS) {
        perror("ERROR");
        return EXIT_FAILURE;
    }

    if (rank == 0) {
        free(a.irn);
        free(a.jcn);
        if (a.type == real) {
            free(a.d_array);
        }
        else {
            free(a.z_array);
        }
    }

    return 0;
}
