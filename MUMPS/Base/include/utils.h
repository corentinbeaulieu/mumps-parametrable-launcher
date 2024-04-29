/** @file utils.h
 *  @brief Contains common functions and structures for the project
 */
#pragma once

/** @brief Needed for @p getline function */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include <dmumps_c.h>
#include <zmumps_c.h>

#include <spral.h>

/** @brief Macro to easily convert C indexing into MUMPS fortran ones */
#define ICNTL(i) icntl[(i) -1]

/**
 * @brief Errors that can be encountered while parsing a MatrixMarket file
 */
typedef enum {
    Success = 0,     /*!< No error */
    NotMatrixMarket, /*!< The file is not a MatrixMarket formatted one */
    Unknown,         /*!< Unknown MatrixMarket value*/
    UnknownFormat, /*!< The format in which the matrix is compressed is unrecognized */
    UnknownType,   /*!< The type of the elements of the matrix is unrecognized */
    UnknownSpecificity, /*!< The specificity of the matrix is unrecognized */
    FileError,  /*!< An error occured in a file manipulation function (fopen...) */
    MallocError /*!< Malloc failed */
} parse_error_t;

/** @brief MUMPS values for symmetry */
typedef enum : int {
    Unsymmetric               = 0, /*!< Not symmetric */
    SymmetricPositiveDefinite = 1, /*!< Symmetric Positive define */
    Symmetric                 = 2  /*!< General symmetric */
} specificity_t;

/** @brief Generic matrix description */
typedef struct {
    MUMPS_INT8 nnz;          /*!< Number of non-zeroes elements */
    MUMPS_INT *restrict irn; /*!< Rows coordinates */
    MUMPS_INT *restrict jcn; /*!< Colomns coordinates */
    /** Values array */
    union {
        DMUMPS_REAL *restrict d_array;    /*!< Double variant */
        ZMUMPS_COMPLEX *restrict z_array; /*!< Double complex variant */
    };
    MUMPS_INT     n;    /*!< Number of row/column in the matrix */
    specificity_t spec; /*!< Global shape of the matrix (see @ref specificity_t) */
    /** Type definition of the element of the matrix */
    enum : unsigned char {
        real           = 0, /*!< Elements are double precision real */
        complex_number = 1  /*!< Elements are double precision complex */
    } type;
} matrix_t;


parse_error_t parseMTX (const char *const input_file, matrix_t a[static 1]);

void dgenerate_rhs (const MUMPS_INT n, const MUMPS_INT8 nnz,
                    const MUMPS_INT   irn[restrict static nnz],
                    const DMUMPS_REAL values[restrict static nnz],
                    DMUMPS_REAL       rhs[restrict static n]);
void zgenerate_rhs (const MUMPS_INT n, const MUMPS_INT8 nnz,
                    const MUMPS_INT      irn[restrict static nnz],
                    const ZMUMPS_COMPLEX values[restrict static nnz],
                    ZMUMPS_COMPLEX       rhs[restrict static n]);

// clang-format off
/**
 * @brief Generic macro to invoke the correct rhs generation function
 */
#define generate_rhs(n, nnz, irn, values, rhs)                                       \
    _Generic(values, DMUMPS_REAL *: dgenerate_rhs, ZMUMPS_COMPLEX *: zgenerate_rhs)  \
        (n, nnz, irn, values, rhs)
// clang-format on

void conversion_CSC_to_COO (const ssize_t nnz, const ssize_t n,
                            const int64_t ptr[restrict static n + 1],
                            MUMPS_INT     jcn[restrict static nnz]);


void print_help (const char program_name[]);
