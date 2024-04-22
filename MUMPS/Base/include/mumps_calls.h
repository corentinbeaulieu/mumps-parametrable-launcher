/** @file mumps_calls.h
 *
 *  @brief Contains the MUMPS related launching functions
 */

#pragma once
#include "utils.h"

/**
 * @brief Possible ordering strategies allowed
 */
typedef enum : unsigned char {
    Automatic = 0, //*< MUMPS choose the ordering
    Metis,         //*< Use MeTiS
    Pord,          //*< Use Pord
    Scotch,        //*< Use SCOTCH
    PTScotch,      //*< Use PT-SCOTCH
} partition_agent_t;

int run_experiment (const matrix_t a, const int par, const int icntl_13,
                    const int icntl_16, const bool resolve,
                    const partition_agent_t partition_agent);
