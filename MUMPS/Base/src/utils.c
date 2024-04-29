/** @file utils.c
 *  @brief Definition of the common functions
 */
#define _GNU_SOURCE
#include "utils.h"
#include <stdlib.h>
#include <string.h>

/**
 * @brief Parses a matrix formatted with the MatrixMarket format and stores it in
 * well-formed matrix @p a for MUMPS.
 *
 * @warning Note that only a subpart of the format is supported (the one we need
 * currently)
 *
 * @param[in]  input_file     MatrixMarket file to parse
 * @param[out] a              Reprensation of the read matrix
 *
 * @return                    An error code (see @ref parse_error_t)
 */
parse_error_t parseMTX (const char *const input_file, matrix_t a[static 1]) {

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
    else if (strcmp(type, "complex_number") == 0) {
        a->type = complex_number;
    }
    else {
        return UnknownType;
    }

    if (strcmp(specificity, "general") == 0) {
        a->spec = Unsymmetric;
    }
    else if (strcmp(specificity, "symmetric") == 0) {
        a->spec = Symmetric;
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
    sscanf(buffer, "%d %lu %ld", &(a->n), &tmp, &(a->nnz));

    a->irn = (typeof(a->irn)) malloc(a->nnz * sizeof(*(a->irn)));
    a->jcn = (typeof(a->jcn)) malloc(a->nnz * sizeof(*(a->jcn)));
    if (a->type == real) {
        a->d_array = (typeof(a->d_array)) malloc(a->nnz * sizeof(*(a->d_array)));
        if (a->irn == NULL || a->jcn == NULL || a->d_array == NULL) {
            return MallocError;
        }

        // Get the elements of the matrix
        MUMPS_INT8 i = 0;
        while (i < a->nnz && getline(&buffer, &buffer_size, fd) > 0) {
            sscanf(buffer, "%d %d %lf", a->irn + i, a->jcn + i, a->d_array + i);
            i++;
        }
    }
    else {
        a->z_array = (ZMUMPS_COMPLEX *) malloc(a->nnz * sizeof(*(a->z_array)));
        if (a->irn == NULL || a->jcn == NULL || a->z_array == NULL) {
            return MallocError;
        }

        // Get the elements of the matrix
        MUMPS_INT8 i = 0;
        while (i < a->nnz && getline(&buffer, &buffer_size, fd) > 0) {
            sscanf(buffer, "%d %d %lf %lf", a->irn + i, a->jcn + i, &(a->z_array[i].r),
                   &(a->z_array[i].i));
            i++;
        }
    }

    fclose(fd);
    free(buffer);

    return Success;
}


/**
 * @brief Generate a vector with double precision floats
 *
 * It is supposed to return a rhs which gives a unit vector as x upon resolution of the
 *  linear system.
 *
 * @warning This function is broken
 * @warning It generates a vector but not with the desired properties
 *
 * @warning @p rhs must be allocated before calling this function
 *
 * @param[in]  n          Size of the vector / matrix
 * @param[in]  nnz        Number of non-zeroes elem in the matrix
 * @param[in]  irn        Row coordinates of the matrix elements
 * @param[in]  values     Values of the elements of the matrix
 * @param[out] rhs        The Right Hand Side generated
 */
void dgenerate_rhs (const MUMPS_INT n, const MUMPS_INT8 nnz,
                    const MUMPS_INT   irn[restrict static nnz],
                    const DMUMPS_REAL values[restrict static nnz],
                    DMUMPS_REAL       rhs[restrict static n]) {

    memset(rhs, 0, n * sizeof(double));

    // We want to have the unit vector as solution
    for (MUMPS_INT8 k = 0; k < nnz; k++) {
        const MUMPS_INT i = irn[k];

        rhs[i] += values[k];
    }
}


/**
 * @brief Generate a vector with double precision complex floats
 *
 * It is supposed to return a rhs which gives a unit vector as x upon resolution of the
 *  linear system.
 *
 * @warning This function is broken
 * @warning It generates a vector but not with the desired properties
 *
 * @warning @p rhs must be allocated before calling this function
 *
 * @param[in]  n          Size of the vector / matrix
 * @param[in]  nnz        Number of non-zeroes elem in the matrix
 * @param[in]  irn        Row coordinates of the matrix elements
 * @param[in]  values     Values of the elements of the matrix
 * @param[out] rhs        The Right Hand Side generated
 */
void zgenerate_rhs (const MUMPS_INT n, const MUMPS_INT8 nnz,
                    const MUMPS_INT      irn[restrict static nnz],
                    const ZMUMPS_COMPLEX values[restrict static nnz],
                    ZMUMPS_COMPLEX       rhs[restrict static n]) {

    memset(rhs, 0, n * sizeof(ZMUMPS_COMPLEX));

    // We want to have the unit vector as solution
    for (MUMPS_INT8 k = 0; k < nnz; k++) {
        const MUMPS_INT i = irn[k];

        rhs[i].r += values[k].r;
        rhs[i].i += values[k].i;
    }
}

/**
 * @brief Converts a matrix in CSC format into COO format.
 *
 * It is needed to do the transition between the spral generator and MUMPS.
 *
 * @warning We assume allocation has been made before calling the method.
 *
 * @param[in]  nnz      Number of non-zeroes elements
 * @param[in]  n        Size of the matrix
 * @param[in]  ptr      CSC columns pointer
 * @param[out] jcn      COO columns indices
 */
void conversion_CSC_to_COO (const ssize_t nnz, const ssize_t n,
                            const int64_t ptr[restrict static n + 1],
                            MUMPS_INT     jcn[restrict static nnz]) {

    for (ssize_t j = 0; j < n; j++) {
        for (ssize_t k = ptr[j]; k < ptr[j + 1]; k++) {
            jcn[k - 1] = j + 1;
        }
    }
}


/**
 * @brief Helper function to print the help on wrong usage or user request
 *
 * @param[in]  program_name Name of the current program (argv[0])
 */
void print_help (const char program_name[]) {

    printf(
        "\x1b[33mUSAGE:\x1b[0m %s -i input_file PAR ICNTL_13 ICNTL_16\n"
        "       %s data_type N bandwith density symmetry_type PAR ICNTL_13 ICNTL_16 "
        "ordering\n"
        "with\n"
        "     data_type      0 (real) or 1 (complex_number)\n"
        "     N              width/height of the generated matrix\n"
        "     bandwidth      width of the upper/lower band\n"
        "     density        density of non-zeroes of the band (must be > 1/N and < 1\n"
        "     symmetry_type  0 (unsymmetric), 1 (positive_definite), 2 (symmetric)\n"
        "\nOptions:\n\t-h\tprint this help and exit\n"
        "\t-s seed\tseed for random generation\n"
        "\t-a\tdo the analysis phase (default: -af)\n"
        "\t-f\tdo the factorisation phase (default: -af)\n"
        "\t-r\tdo the resolving phase (default: -af)\n"
        "\t-g\tuse global matrix density\n\n",
        program_name, program_name);
}
