
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>
#include <omp.h>
#include <string.h>

#include <dmumps_c.h>
#include <zmumps_c.h>

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
        d,
        z
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
        a->type = d;
    }
    else if (strcmp(type, "complex") == 0) {
        a->type = z;
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
    if (a->type == d) {
        a->d_array = (DMUMPS_REAL *) malloc(*nnz * sizeof(DMUMPS_REAL));
        if (*irn == NULL || *jcn == NULL || a->d_array == NULL) {
            return MallocError;
        }

        // Get the elements of the matrix
        MUMPS_INT8 i = 0;
        while (getline(&buffer, &buffer_size, fd) > 0 && i < *nnz) {
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
        while (getline(&buffer, &buffer_size, fd) > 0 && i < *nnz) {
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

int main (int argc, char *argv[]) {

    if (argc != 5) {
        fprintf(stderr, "USAGE: %s input_file PAR ICNTL_13 ICNTL_16\n", argv[0]);
        return EXIT_FAILURE;
    }


    MUMPS_INT     n;
    MUMPS_INT8    nnz;
    MUMPS_INT    *irn;
    MUMPS_INT    *jcn;
    matrix_t      a;
    specificity_t spec;

    parse_error_t perr = parseMTX(&n, &nnz, &irn, &jcn, &a, &spec, argv[1]);
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
            fprintf(stderr, "The form of the matrix must be general or symmetric\n");
            return perr;

        case FileError:
        case MallocError:
            perror("ERROR: ");
            return EXIT_FAILURE;

        case Success:
            break;
    }


    int            ierr, rank;
    DMUMPS_STRUC_C dmumps_par;
    ZMUMPS_STRUC_C zmumps_par;

    const int       par      = (int) atoi(argv[2]);
    const MUMPS_INT icntl_13 = (MUMPS_INT) atoi(argv[3]);
    const MUMPS_INT icntl_16 = (MUMPS_INT) atoi(argv[4]);


    if (a.type == d) {

        DMUMPS_REAL *rhs = malloc(n * sizeof(DMUMPS_REAL));

        generate_rhs(n, nnz, irn, a.d_array, rhs);

        // Initialize MPI
        ierr = MPI_Init(&argc, &argv);
        if (ierr != 0) {
            perror("ERROR: ");
            return EXIT_FAILURE;
        }
        ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (ierr != 0) {
            perror("ERROR: ");
            return EXIT_FAILURE;
        }

        // Initialize MUMPS
        dmumps_par.comm_fortran = USE_COMM_WORLD;
        dmumps_par.job          = JOB_INIT;

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
            dmumps_par.rhs = rhs;
            dmumps_par.sym = (int) spec;
        }

        // Enables residual computation and printing
        dmumps_par.ICNTL(11) = 2;
        dmumps_par.ICNTL(13) = icntl_13;
        // This parameter controls the memory relaxation of the MUMPS during
        // factorisation. We need to increase it as some matrix needs large size of
        // temporary memory
        dmumps_par.ICNTL(14) = 25;
        dmumps_par.ICNTL(16) = icntl_16;

        // Launch MUMPS for Analysis, Factorisation and Resolution
        dmumps_par.job = 6;
        dmumps_c(&dmumps_par);

        // Deinitialize MUMPS
        dmumps_par.job = JOB_END;
        dmumps_c(&dmumps_par);

        free(rhs);
    }
    else {
        ZMUMPS_COMPLEX *rhs = malloc(n * sizeof(ZMUMPS_COMPLEX));

        generate_rhs(n, nnz, irn, a.z_array, rhs);

        const MUMPS_INT icntl_13 = (MUMPS_INT) atoi(argv[3]);
        const MUMPS_INT icntl_16 = (MUMPS_INT) atoi(argv[4]);


        // Initialize MPI
        ierr = MPI_Init(&argc, &argv);
        if (ierr != 0) {
            perror("ERROR: ");
            return EXIT_FAILURE;
        }
        ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (ierr != 0) {
            perror("ERROR: ");
            return EXIT_FAILURE;
        }

        // Initialize MUMPS
        zmumps_par.comm_fortran = USE_COMM_WORLD;
        zmumps_par.job          = JOB_INIT;

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
            zmumps_par.rhs = rhs;
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
        zmumps_par.job = 6;
        zmumps_c(&zmumps_par);

        // Deinitialize MUMPS
        zmumps_par.job = JOB_END;
        zmumps_c(&zmumps_par);

        free(rhs);
    }

    // Print the solution for verification
    // TODO: Automate the verification (residual, forward error if possible...)
    // The residual can be computed by mumps and returned with ICNTL(11) = 2
    /* if(rank == 0) { */
    /*     fputs("Solution is : (", stdout); */
    /*     for(size_t i = 0; i < n; i++) { */
    /*         printf("%8.2lf, ", rhs[i]); */
    /*     } */
    /*     puts(")\n"); */
    /* } */

    ierr = MPI_Finalize();
    if (ierr != 0) {
        perror("ERROR: ");
        return EXIT_FAILURE;
    }

    free(irn);
    free(jcn);
    free(a.d_array);

    return 0;
}
