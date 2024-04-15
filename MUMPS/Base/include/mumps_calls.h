/** @file mumps_calls.h
 *
 *  @brief Contains the MUMPS related launching functions
 */

#pragma once
#include "utils.h"

int run_experiment (const matrix_t a, const int par, const int icntl_13,
                    const int icntl_16, const bool resolve);
