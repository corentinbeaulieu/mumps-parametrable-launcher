
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <mpi.h>
#include <omp.h>

#include <dmumps_c.h>
#include <zmumps_c.h>

#include <spral_random_matrix.h>

#define JOB_INIT       -1
#define JOB_END        -2
#define USE_COMM_WORLD -987'654

#define ICNTL(i) icntl[(i) -1]

/** Parsing errors **/
typedef enum {
    Success = 0,
    NotMatrixMarket,
    Unknown,
    UnknownFormat,
    UnknownType,
    UnknownSpecificity,
    FileError,
    MallocError
} parse_error_t;

/** MUMPS values for symmetry **/
typedef enum : int {
    Unsymmetric             = 0,
    SymmetricDefinePositive = 1,
    Symmetric               = 2
} specificity_t;

/** Generic matrix for both double and double complex **/
typedef struct {
    union {
        DMUMPS_REAL    *d_array;
        ZMUMPS_COMPLEX *z_array;
    };
    enum : unsigned char {
        real    = 0,
        complex = 1
    } type;
} matrix_t;

/** Parses a matrix formatted with the MatrixMarket format and stores it in well-formed
 *  arrays for MUMPS. Note that only a subpart of the format is supported (the one we
 *  need currently)
 */
parse_error_t parseMTX (MUMPS_INT *n, MUMPS_INT8 *nnz, MUMPS_INT **irn, MUMPS_INT **jcn,
                        matrix_t *a, specificity_t *spec, char *input_file) {

    FILE *const fd = fopen(input_file, "r");
    if (fd == NULL) {
        return FileError;
    }

    char  *buffer = NULL;
    char   verification[32], matrix[32], format[32], type[32], specificity[32];
    size_t buffer_size = 0;

    getline(&buffer, &buffer_size, fd);

    sscanf(buffer, "%s %s %s %s %s", verification, matrix, format, type, specificity);

    // Verify the format is supported
    if (strcmp(verification + 2, "MatrixMarket") != 0) {
        return NotMatrixMarket;
    }

    if (strcmp(matrix, "matrix") != 0) {
        return Unknown;
    }

    if (strcmp(format, "coordinate") != 0) {
        return UnknownFormat;
    }

    if (strcmp(type, "real") == 0) {
        a->type = real;
    }
    else if (strcmp(type, "complex") == 0) {
        a->type = complex;
    }
    else {
        return UnknownType;
    }

    if (strcmp(specificity, "general") == 0) {
        *spec = Unsymmetric;
    }
    else if (strcmp(specificity, "symmetric") == 0) {
        *spec = Symmetric;
    }
    else {
        return UnknownSpecificity;
    }

    // Go pass the comments and notes
    getline(&buffer, &buffer_size, fd);
    while (buffer[0] == '%') {
        getline(&buffer, &buffer_size, fd);
    }

    // Get the matrix characteristics
    size_t tmp;
    sscanf(buffer, "%d %lu %ld", n, &tmp, nnz);

    *irn = (int *) malloc(*nnz * sizeof(int));
    *jcn = (int *) malloc(*nnz * sizeof(int));
    if (a->type == real) {
        a->d_array = (DMUMPS_REAL *) malloc(*nnz * sizeof(DMUMPS_REAL));
        if (*irn == NULL || *jcn == NULL || a->d_array == NULL) {
            return MallocError;
        }

        // Get the elements of the matrix
        MUMPS_INT8 i = 0;
        while (i < *nnz && getline(&buffer, &buffer_size, fd) > 0) {
            sscanf(buffer, "%d %d %lf", *irn + i, *jcn + i, a->d_array + i);
            i++;
        }
    }
    else {
        a->z_array = (ZMUMPS_COMPLEX *) malloc(*nnz * sizeof(ZMUMPS_COMPLEX));
        if (*irn == NULL || *jcn == NULL || a->z_array == NULL) {
            return MallocError;
        }

        // Get the elements of the matrix
        MUMPS_INT8 i = 0;
        while (i < *nnz && getline(&buffer, &buffer_size, fd) > 0) {
            sscanf(buffer, "%d %d %lf %lf", *irn + i, *jcn + i, &(a->z_array[i].r),
                   &(a->z_array[i].i));
            i++;
        }
    }

    fclose(fd);
    free(buffer);

    return Success;
}

/** Generates a rhs that will give the desired x upon resolution
 *  WARNING: work in progress
 */
void dgenerate_rhs (MUMPS_INT n, MUMPS_INT8 nnz, MUMPS_INT irn[static nnz],
                    DMUMPS_REAL a[static nnz], DMUMPS_REAL rhs[static n]) {

    memset(rhs, 0, n * sizeof(double));

    // We want to have the unit vector as solution
    for (MUMPS_INT8 k = 0; k < nnz; k++) {
        const MUMPS_INT i = irn[k];

        rhs[i] += a[k];
    }
}

void zgenerate_rhs (MUMPS_INT n, MUMPS_INT8 nnz, MUMPS_INT irn[static nnz],
                    ZMUMPS_COMPLEX a[static nnz], ZMUMPS_COMPLEX rhs[static n]) {

    memset(rhs, 0, n * sizeof(ZMUMPS_COMPLEX));

    // We want to have the unit vector as solution
    for (MUMPS_INT8 k = 0; k < nnz; k++) {
        const MUMPS_INT i = irn[k];

        rhs[i].r += a[k].r;
        rhs[i].i += a[k].i;
    }
}

// clang-format off
#define generate_rhs(n, nnz, irn, a, rhs)                                       \
    _Generic(a, DMUMPS_REAL *: dgenerate_rhs, ZMUMPS_COMPLEX *: zgenerate_rhs)  \
        (n, nnz, irn, a, rhs)
// clang-format on


/** Converts a matrix in the CSC into the COO format.
 *  It is needed to do the transition between the spral generator and MUMPS.
 *  This is a naive implementation which only translates the column ptr vector into
 *  a column coordinate one.
 *  We assume allocation has been made before calling the method.
 */
static void conversion_CSC_to_COO (const ssize_t nnz, const ssize_t n,
                                   const int64_t ptr[static n],
                                   MUMPS_INT     jcn[static nnz]) {
    size_t j = 1;
    for (ssize_t i = 1; i <= nnz; i++) {
        while (i >= ptr[j]) {
            j++;
        }

        jcn[i - 1] = j;
    }
}


/** Helper function to print the help on wrong usage or user request
 */
static void print_help (const char program_name[]) {

    printf("\x1b[33mUSAGE:\x1b[0m %s -f input_file PAR ICNTL_13 ICNTL_16\n"
           "       %s data_type N nnz symmetry_type PAR ICNTL_13 ICNTL_16\n"
           "with\n"
           "     data_type      0 (real) or 1 (complex)\n"
           "     N              width/height of the generated matrix\n"
           "     nnz            number of non-zero (must be < N*N + 1)\n"
           "     symmetry_type  0 (unsymmetric), 1 (positive_definite), 2 (symmetric)\n"
           "\nOptions:\n\th\tprint this help and exit\n"
           "\ts seed\tseed for random generation\n",
           program_name, program_name);
}

int main (int argc, char *argv[]) {

    char opt      = -1;
    bool readfile = false;
    int  state    = 0;

    // Get the options
    while ((opt = getopt(argc, argv, "fhs:")) != -1) {
        switch (opt) {
            case 'h': // Print help and exit
                print_help(argv[0]);
                return EXIT_SUCCESS;
            case 'f': // Don't use the generator
                readfile = true;
                break;
            case 's': // Provide a seed for the generator
                state = (int) atoi(optarg);
                break;
            default:
                print_help(argv[0]);
                return EXIT_FAILURE;
        }
    }

    MUMPS_INT     n;
    MUMPS_INT8    nnz;
    MUMPS_INT    *irn;
    MUMPS_INT    *jcn;
    matrix_t      a;
    specificity_t spec;
    int           ierr, rank;

    if (readfile == true) {

        if ((argc - optind) != 4) {
            print_help(argv[0]);
            return EXIT_FAILURE;
        }

        // Read and parse the input matrix
        parse_error_t perr = parseMTX(&n, &nnz, &irn, &jcn, &a, &spec, argv[optind]);
        switch (perr) {
            case NotMatrixMarket:
                fprintf(stderr, "Not a Matrix Market input!\n");
                return perr;
            case Unknown:
                fprintf(stderr, "Second description term must be matrix\n");
                return perr;
            case UnknownType:
                fprintf(stderr, "Only real/complex numbers are supported\n");
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
                return EXIT_FAILURE;

            case Success:
                break;
        }
        optind++;
    }
    else {

        if ((argc - optind) != 7) {
            print_help(argv[0]);
            return EXIT_FAILURE;
        }

        a.type = (typeof(a.type)) atoi(argv[optind++]);
        n      = (MUMPS_INT) atoi(argv[optind++]);
        nnz    = (MUMPS_INT8) strtol(argv[optind++], nullptr, 10);
        spec   = (typeof(spec)) atoi(argv[optind++]);

        enum spral_matrix_type matrix_type = SPRAL_MATRIX_UNSPECIFIED;
        // The generator returns an error (-3) if nnz is too high in regard of n.
        // The maximal value of nnz depends on whether the matrix is symmetric or not
        int64_t                max_nnz     = 0;

        // Translate input parameters into spral enum
        switch (spec) {
            case Unsymmetric:
                max_nnz     = n * n;
                matrix_type = SPRAL_MATRIX_REAL_UNSYM;
                break;
            case Symmetric:
                max_nnz     = ((n * n) + n) / 2;
                matrix_type = SPRAL_MATRIX_REAL_SYM_INDEF;
                break;
            case SymmetricDefinePositive:
                max_nnz     = ((n * n) + n) / 2;
                matrix_type = SPRAL_MATRIX_REAL_SYM_PSDEF;
                break;
        }

        // The change of nnz is a bit hacky. A solution would be to take a density
        // instead of nnz and compute then compute it.
        if (nnz > max_nnz) {
            nnz = max_nnz - 1;
        }
        if (n > nnz) {
            nnz = n + 1;
        }

        // Allocate the various vectors
        int64_t *ptr = (typeof(ptr)) malloc((n + 1) * sizeof(*ptr));

        irn = (typeof(irn)) malloc(nnz * sizeof(*irn));
        jcn = (typeof(jcn)) malloc(nnz * sizeof(*jcn));

        if (a.type == real) {
            a.d_array = (typeof(a.d_array)) malloc(nnz * sizeof(*a.d_array));

            ierr = spral_random_matrix_generate_long(
                &state, matrix_type, n, n, nnz, ptr, irn, (double *) a.d_array,
                SPRAL_RANDOM_MATRIX_FINDEX | SPRAL_RANDOM_MATRIX_NONSINGULAR |
                    SPRAL_RANDOM_MATRIX_SORT);
        }
        // FIXME: complex matrix generation not supported by spral
        else {
            matrix_type = -matrix_type;
            a.z_array   = (typeof(a.z_array)) malloc(nnz * sizeof(*a.z_array));

            ierr = 0;
            ierr = spral_random_matrix_generate_long(
                &state, matrix_type, n, n, nnz, ptr, irn, (double *) a.d_array,
                SPRAL_RANDOM_MATRIX_FINDEX | SPRAL_RANDOM_MATRIX_NONSINGULAR |
                    SPRAL_RANDOM_MATRIX_SORT);
        }

        // Check the generator return code
        if (ierr != 0) {
            fprintf(
                stderr,
                "\x1b[31mERROR\x1b[0m The matrix generation failed\tERROR CODE: %d\n",
                ierr);
            return EXIT_FAILURE;
        }

        conversion_CSC_to_COO(nnz, n, ptr, jcn);
        free(ptr);
    }

    DMUMPS_STRUC_C dmumps_par;
    ZMUMPS_STRUC_C zmumps_par;

    const int       par      = (int) atoi(argv[optind++]);
    const MUMPS_INT icntl_13 = (MUMPS_INT) atoi(argv[optind++]);
    const MUMPS_INT icntl_16 = (MUMPS_INT) atoi(argv[optind++]);

    if (a.type == real) {

        /* DMUMPS_REAL *rhs = malloc(n * sizeof(DMUMPS_REAL)); */

        /* generate_rhs(n, nnz, irn, a.d_array, rhs); */

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

        // Initialize MUMPS
        dmumps_par.comm_fortran = USE_COMM_WORLD;
        dmumps_par.job          = JOB_INIT;
        // Activate logs & metrics only on root rank
        zmumps_par.ICNTL(2)     = -1;

        // This parameter controls the involvement of the root rank in the factorization
        // and solve phase
        dmumps_par.par       = par;
        // This parameter controls the parallelism of factorization of the root node
        dmumps_par.ICNTL(13) = icntl_13;
        // This parameter explicitly request a number of OMP threads
        dmumps_par.ICNTL(16) = icntl_16;

        dmumps_c(&dmumps_par);

        // Define the system
        if (rank == 0) {
            dmumps_par.n   = n;
            dmumps_par.nnz = nnz;
            dmumps_par.irn = irn;
            dmumps_par.jcn = jcn;
            dmumps_par.a   = a.d_array;
            /* dmumps_par.rhs = rhs; */
            dmumps_par.sym = (int) spec;
        }

        // Enables residual computation and printing
        dmumps_par.ICNTL(11) = 2;
        dmumps_par.ICNTL(13) = icntl_13;
        // This parameter controls the memory relaxation of the MUMPS during
        // factorisation. We need to increase it as some matrix needs large size of
        // temporary memory
        dmumps_par.ICNTL(14) = 30;
        dmumps_par.ICNTL(16) = icntl_16;

        // Launch MUMPS for Analysis, Factorisation
        dmumps_par.job = 4;
        dmumps_c(&dmumps_par);
        if (dmumps_par.infog[0] != 0) {
            return EXIT_FAILURE;
        }

        // Deinitialize MUMPS
        dmumps_par.job = JOB_END;
        dmumps_c(&dmumps_par);

        /* free(rhs); */
    }
    else {
        /* ZMUMPS_COMPLEX *rhs = malloc(n * sizeof(ZMUMPS_COMPLEX)); */

        /* generate_rhs(n, nnz, irn, a.z_array, rhs); */

        const MUMPS_INT icntl_13 = (MUMPS_INT) atoi(argv[3]);
        const MUMPS_INT icntl_16 = (MUMPS_INT) atoi(argv[4]);


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
        if (rank == 0) {
            zmumps_par.n   = n;
            zmumps_par.nnz = nnz;
            zmumps_par.irn = irn;
            zmumps_par.jcn = jcn;
            zmumps_par.a   = a.z_array;
            /* zmumps_par.rhs = rhs; */
            zmumps_par.sym = (int) spec;
        }

        // Enables residual computation and printing
        zmumps_par.ICNTL(11) = 2;
        zmumps_par.ICNTL(13) = icntl_13;
        // This parameter controls the memory relaxation of the MUMPS during
        // factorisation. We need to increase it as some matrix needs large size of
        // temporary memory
        zmumps_par.ICNTL(14) = 25;
        zmumps_par.ICNTL(16) = icntl_16;

        // Launch MUMPS for Analysis, Factorisation and Resolution
        zmumps_par.job = 4;
        zmumps_c(&zmumps_par);
        if (dmumps_par.infog[0] != 0) {
            return EXIT_FAILURE;
        }

        // Deinitialize MUMPS
        zmumps_par.job = JOB_END;
        zmumps_c(&zmumps_par);

        /* free(rhs); */
    }

    ierr = MPI_Finalize();
    if (ierr != 0) {
        perror("ERROR");
        return EXIT_FAILURE;
    }

    free(irn);
    free(jcn);
    if (a.type == real) {
        free(a.d_array);
    }
    else {
        free(a.z_array);
    }

    return 0;
}
