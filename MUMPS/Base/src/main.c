/** @mainpage This project aims at providing a parametrable executable to measure mumps
 * execution with different parameters. The main goal is to use machine learning methods
 * to predict the best parameters to speedup the resolution of a linear system with
 * MUMPS
 */

/** @file main.c
 *  @brief Contains the main function of the program which
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
    if (ierr != 0) {
        perror("ERROR");
        return EXIT_FAILURE;
    }
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (ierr != 0) {
        perror("ERROR");
        return EXIT_FAILURE;
    }

    char opt      = -1;
    bool readfile = false;
    int  state    = 0;
    bool resolve  = false;

    // Get the options
    while ((opt = getopt(argc, argv, "fhrs:")) != -1) {
        switch (opt) {
            case 'h': // Print help and exit
                print_help(argv[0]);
                return EXIT_SUCCESS;
            case 'f': // Don't use the generator
                readfile = true;
                break;
            case 'r':
                resolve = true;
                break;
            case 's': // Provide a seed for the generator
                state = (int) atoi(optarg);
                break;
            default:
                print_help(argv[0]);
                return EXIT_FAILURE;
        }
    }

    if (readfile == true) {

        if ((argc - optind) != 4) {
            if (rank == MPI_ROOT) {
                print_help(argv[0]);
            }
            return EXIT_FAILURE;
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

        if ((argc - optind) != 8) {
            print_help(argv[0]);
            return EXIT_FAILURE;
        }


        a.type                    = (typeof(a.type)) atoi(argv[optind++]);
        a.n                       = (MUMPS_INT) atoi(argv[optind++]);
        const MUMPS_INT bandwidth = (typeof(bandwidth)) atoi(argv[optind++]);
        const double    density   = (typeof(density)) atof(argv[optind++]);
        if ((density > 1.0) || (density <= 0.0)) {
            print_help(argv[0]);
            return EXIT_FAILURE;
        }
        a.spec = (typeof(a.spec)) atoi(argv[optind++]);


        int64_t               *ptr         = nullptr;
        enum spral_matrix_type matrix_type = SPRAL_MATRIX_UNSPECIFIED;

        // The generator returns an error (-3) if nnz is too high in regard of n.
        // The maximal value of nnz depends on whether the matrix is symmetric or not
        int64_t max_nnz = bandwidth * (2lu * a.n - bandwidth - 1lu)
                          + a.n; //*< Number of elements in the dense band

        // Translate input parameters into spral enum
        switch (a.spec) {
            case Unsymmetric:
                matrix_type = SPRAL_MATRIX_REAL_UNSYM;
                break;
            case Symmetric:
                max_nnz     = (max_nnz + a.n) / 2;
                matrix_type = SPRAL_MATRIX_REAL_SYM_INDEF;
                break;
            case SymmetricDefinePositive:
                max_nnz     = (max_nnz + a.n) / 2;
                matrix_type = SPRAL_MATRIX_REAL_SYM_PSDEF;
                break;
        }
        a.nnz = (typeof(a.nnz)) (density * (double) max_nnz);

        if (a.n > a.nnz) {
            a.nnz = a.n + 1;
        }

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
            fprintf(
                stderr,
                "\x1b[31mERROR\x1b[0m The matrix generation failed\tERROR CODE: %d\n",
                ierr);
            return EXIT_FAILURE;
        }

        conversion_CSC_to_COO(a.nnz, a.n, ptr, a.jcn);
        free(ptr);
    }

    const int       par      = (int) atoi(argv[optind++]);
    const MUMPS_INT icntl_13 = (MUMPS_INT) atoi(argv[optind++]);
    const MUMPS_INT icntl_16 = (MUMPS_INT) atoi(argv[optind++]);


    run_experiment(a, par, icntl_13, icntl_16, resolve);

    ierr = MPI_Finalize();
    if (ierr != 0) {
        perror("ERROR");
        return EXIT_FAILURE;
    }

    free(a.irn);
    free(a.jcn);
    if (a.type == real) {
        free(a.d_array);
    }
    else {
        free(a.z_array);
    }

    return 0;
}
