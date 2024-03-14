
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <dmumps_c.h>
#define JOB_INIT -1
#define JOB_END -2
#define USE_COMM_WORLD -987654

#define ICNTL(i) icntl[(i)-1]

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
typedef enum {
    Unsymmetric = 0,
    SymmetricDefinePositive = 1,
    Symmetric = 2
} specificity_t;

/** Parses a matrix formatted with the MatrixMarket format and stores it in well-formed arrays for MUMPS
 *  Note that only a subpart of the format is supported (the one we need currently)
 */
parse_error_t parseMTX (MUMPS_INT *n, MUMPS_INT8 *nnz, MUMPS_INT **irn, MUMPS_INT **jcn, 
                double **a, specificity_t *spec, char *input_file) {

    FILE *const fd = fopen(input_file, "r");
    if (fd == NULL) {
        return FileError;
    }

    char *buffer = NULL;
    char verification[32], matrix[32], format[32], type[32], specificity[32];
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

    if (strcmp(type, "real") != 0) {
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
    while(buffer[0] == '%') getline(&buffer, &buffer_size, fd);

    // Get the matrix characteristics
    size_t tmp;
    sscanf(buffer, "%u %lu %lu", n, &tmp, nnz);

    *irn = (int *) malloc(*nnz * sizeof(int));
    *jcn = (int *) malloc(*nnz * sizeof(int));
    *a = (double *) malloc(*nnz * sizeof(double));

    if (*irn == NULL || *jcn == NULL || *a == NULL) return MallocError;
    
    // Get the elements of the matrix
    size_t i = 0;
    while(getline(&buffer, &buffer_size, fd) > 0 && i < *nnz) {
        sscanf(buffer, "%u %u %lf", *irn + i, *jcn + i, *a + i);
        i++;
    }

    fclose(fd);
    free(buffer);

    return Success;
}

/** Generates a rhs in order to have the desired x upon resolving the system
 *  This function aims to ease the verification of the result
 *  WARNING: work in progress
 */
void generate_rhs (MUMPS_INT n, MUMPS_INT8 nnz, MUMPS_INT irn[static nnz],
                   double a[static n], double rhs[static n]) {

    memset(rhs, 0, n * sizeof(double));

    // We want to have the unit vector as solution
    for (size_t k = 0; k < nnz; k++) {
        const size_t i = irn[k];

        rhs[i] += a[k];
    }
}

int main (int argc, char *argv[]) {

    if(argc != 3) {
        fprintf(stderr, "USAGE: %s input_file ICNTL_13\n", argv[0]);
        return EXIT_FAILURE;
    }


    MUMPS_INT n;
    MUMPS_INT8 nnz;
    MUMPS_INT *irn;
    MUMPS_INT *jcn;
    double *a;
    specificity_t spec;

    parse_error_t perr = parseMTX(&n, &nnz, &irn, &jcn, &a, &spec,argv[1]);
    switch(perr) {
        case NotMatrixMarket:
            fprintf(stderr, "Not a Matrix Market input!\n");
            return perr;
        case Unknown:
            fprintf(stderr, "Second description term must be matrix\n");
            return perr;
        case UnknownType:
            fprintf(stderr, "Only real numbers are supported\n");
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

    double *rhs = malloc(n * sizeof(double));

    generate_rhs(n, nnz, irn, a, rhs);

    int ierr, rank;
    DMUMPS_STRUC_C dmumps_par;

    const MUMPS_INT icntl_13 = (MUMPS_INT) atoi(argv[2]);

    // This parameter controls the involvement of the root rank in the factorization
    dmumps_par.ICNTL(13) = icntl_13;

    // Initialize MPI
    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialize MUMPS
    dmumps_par.comm_fortran = USE_COMM_WORLD;
    dmumps_par.par = 1;
    dmumps_par.job = JOB_INIT;

    dmumps_c(&dmumps_par);

    // Define the system 
    if(rank == 0) {
        dmumps_par.n = n;
        dmumps_par.nnz = nnz;
        dmumps_par.irn = irn;
        dmumps_par.jcn = jcn;
        dmumps_par.a = a;
        dmumps_par.rhs = rhs;
        dmumps_par.sym = spec;
    }

    dmumps_par.ICNTL(13) = icntl_13;
    // This parameter controls the memory relaxation of the MUMPS during factorisation
    // We need to increase it as some matrix needs large size of temporary memory
    dmumps_par.ICNTL(14) = 25;

    // Launch MUMPS for Analysis, Factorisation and Resolution
    dmumps_par.job = 6;
    dmumps_c(&dmumps_par);

    // Deinitialize MUMPS
    dmumps_par.job = JOB_END;
    dmumps_c(&dmumps_par);

    // Print the solution for verification
    // TODO: Automate the verification (residual, forward error if possible...)
    // The residual can be computed by mumps and returned with ICNTL(11) = 2
    if(rank == 0) {
        fputs("Solution is : (", stdout);
        for(size_t i = 0; i < n; i++) {
            printf("%8.2lf, ", rhs[i]);
        }
        puts(")\n");
    }

    ierr = MPI_Finalize();

    free(irn);
    free(jcn);
    free(a);
    free(rhs);

    return 0;
}
