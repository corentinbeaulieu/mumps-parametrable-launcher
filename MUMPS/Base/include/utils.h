/** @file utils.h
 *  @brief Contains common functions and structures for the project
 */
#pragma once

#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include <dmumps_c.h>
#include <zmumps_c.h>

#include <spral.h>

#define JOB_INIT       -1
#define JOB_END        -2
#define USE_COMM_WORLD -987'654

#define ICNTL(i) icntl[(i) -1]

/**
 * @brief Errors that can be encountered while parsing a MatrixMarket file
 */
typedef enum {
    Success = 0,     /**< No error */
    NotMatrixMarket, /**< The file is not a MatrixMarket formatted one */
    Unknown,         /**< */
    UnknownFormat, /**< The format in which the matrix is compressed is unrecognized */
    UnknownType,   /**< The type of the elements of the matrix is unrecognized */
    UnknownSpecificity, /**< The specificity of the matrix is unrecognized */
    FileError,  /**< An error occured in a file manipulation function (fopen...) */
    MallocError /**< Malloc failed */
} parse_error_t;

/** MUMPS values for symmetry */
typedef enum : int {
    Unsymmetric             = 0, /**< Not symmetric */
    SymmetricDefinePositive = 1, /**< Symmetric Positive define */
    Symmetric               = 2  /**< General symmetric */
} specificity_t;

/** Generic matrix */
typedef struct {
    MUMPS_INT8 nnz; /**< Number of non-zeroes elements */
    MUMPS_INT *irn; /**< Rows coordinates */
    MUMPS_INT *jcn; /**< Colomns coordinates */
    /** Values array */
    union {
        DMUMPS_REAL    *d_array; /**< Double variant */
        ZMUMPS_COMPLEX *z_array; /**< Double complex variant */
    };
    MUMPS_INT     n;    /**< Number of row/column in the matrix */
    specificity_t spec; /**< Global shape of the matrix (see @ref specifity_t) */
    /** Type definition of the element of the matrix */
    enum : unsigned char {
        real           = 0, /**< Elements are double precision real */
        complex_number = 1  /**< Elements are double precision complex */
    } type;
} matrix_t;


parse_error_t parseMTX (const char *const input_file, matrix_t a[static 1]);

void dgenerate_rhs (const MUMPS_INT n, const MUMPS_INT8 nnz,
                    const MUMPS_INT   irn[static nnz],
                    const DMUMPS_REAL values[static nnz],
                    DMUMPS_REAL rhs[static n]);
void zgenerate_rhs (const MUMPS_INT n, const MUMPS_INT8 nnz,
                    const MUMPS_INT      irn[static nnz],
                    const ZMUMPS_COMPLEX values[static nnz],
                    ZMUMPS_COMPLEX       rhs[static n]);

// clang-format off
#define generate_rhs(n, nnz, irn, values, rhs)                                       \
    _Generic(values, DMUMPS_REAL *: dgenerate_rhs, ZMUMPS_COMPLEX *: zgenerate_rhs)  \
        (n, nnz, irn, values, rhs)
// clang-format on

void conversion_CSC_to_COO (const ssize_t nnz, const ssize_t n,
                            const int64_t ptr[static n + 1], MUMPS_INT jcn[static nnz]);


void print_help (const char program_name[]);
