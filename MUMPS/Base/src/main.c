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
#include <stdlib.h>
#include <unistd.h>

#include <mpi.h>
#include <omp.h>

#include "mumps_calls.h"


int main (int argc, char *argv[]) {

    matrix_t a;
    int      ierr, rank;

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
            case MallocError:
                perror("ERROR");
                return perr;

            case Success:
                break;
        }
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
    }

    mumps_t info = {
        .a               = a,
        .rhs             = nullptr,
        .par             = (int) atoi(argv[optind]),
        .icntl_13        = (MUMPS_INT) atoi(argv[optind + 1]),
        .icntl_16        = (MUMPS_INT) atoi(argv[optind + 2]),
        .partition_agent = (int) atoi(argv[optind + 3]),
    };


    mumps_init(&info);
    if (analysis == true) {
        mumps_run_ana(&info);
    }
    else if (mumps_restore(&info) != EXIT_SUCCESS) {
        mumps_run_ana(&info);
    }

    if (facto == true) {
        mumps_run_facto(&info);
    }
    else {
        mumps_save(&info);
    }

    if (resolve == true) {
        mumps_run_res(&info);
    }
    mumps_finalize(&info);

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
