#!/bin/bash

# Check the inputs

if [[ $# != 11 ]]; then
    printf >&2 "\e[33mUSAGE:\e[0m %s job synthetic n bandwidth density symmetry num_proc num_threads par inctl13 ordering\n" "$0"
    exit 1
fi

job=$1
synthetic=$2
n=$3
bandwidth=$4
density=$5
symmetry=$6
num_proc=$7
num_threads=$8
par=$9
icntl_13=${10}
ordering=${11}

## FIXME: complex support is broken for the moment. The generator does not support complex matrix.
##   We have to find a way to construct complex matrix while keeping the non-singularity property.
#
# type=$1
# case $type in
#     real)
#         type_int=0
#         ;;
#     complex)
#         type_int=1
#         ;;
#     *)
#         printf >&2 "\e[31mERROR\e[0m %s is a not a valid datatype (possible values: real complex)\n" "$type"
#         exit 1
#         ;;
# esac
# type="real"
type_int=0

case "$job" in
"ap")
    stage="Analysis"
    ;;
"ae")
    stage="Analysis"
    ;;
"fp")
    stage="Factorization"
    ;;
"fe")
    stage="Factorization"
    ;;
*)
    printf >&2 "\e[31mERROR\e[0m Wrong job value: %s (possible: ap, ae, fp, fe)\n" "$job"
    exit 2
    ;;
esac

if [[ "$synthetic" != "True" && "$synthetic" != "False" ]]; then
    printf >&2 "\e[31mERROR\e[0m synthetic must be True or False (given: %s)\n" "$synthetic"
    exit 2
fi

if [[ $n == "" || $(printf "%s" "$n" | grep -E -vo "[1-9][0-9]*") != "" ]]; then
    printf >&2 "\e[31mERROR\e[0m %s is a not a valid integer for N\n" "$n"
    exit 3
fi

if [[ $(echo "$bandwidth" | grep -E -vo "0.[0-9]*") != "" ]]; then
    printf >&2 "\e[31mERROR\e[0m %s is a not a valid real for bandwidth\n" "$bandwidth"
    exit 4
fi

if [[ $(echo "$density" | grep -E -vo "0.[0-9]*") != "" ]]; then
    printf >&2 "\e[31mERROR\e[0m %s is a not a valid real for density\n" "$density"
    exit 5
fi

if [[ $symmetry != "0" && $symmetry != "1" && $symmetry != "2" && $symmetry != "0.0" && $symmetry != "1.0" && $symmetry != "2.0" ]]; then
    printf >&2 "\e[31mERROR\e[0m %s is a wrong symmetry kind\n" "$symmetry"
    exit 6
fi

if [[ $(echo "$num_threads" | grep -E -o "[1-9][0-9]*") == "" ]]; then
    printf >&2 "\e[31mERROR\e[0m %s is a wrong number of threads\n" "$num_threads"
    exit 7
fi

if [[ $(echo "$num_proc" | grep -E -o "[1-9][0-9]*") == "" ]]; then
    printf >&2 "\e[31mERROR\e[0m %s is a wrong number of MPI ranks\n" "$num_threads"
    exit 8
fi

if [[ $(echo "$par" | grep -E -o "(0|1)") == "" ]]; then
    printf >&2 "\e[31mERROR\e[0m %s is not a proper PAR parameter (0 or 1)\n" "$par"
    exit 9
fi

if [[ $(echo "$icntl_13" | grep -E -o "([1-9][0-9]*)|-1|0") == "" ]]; then
    printf >&2 "\e[31mERROR\e[0m %s is not a proper ICNTL 13 parameter (-1, 0 or >0)\n" "$icntl_13"
    exit 10
fi

case "$ordering" in
"metis")
    ordering_int=1
    ;;
"pord")
    ordering_int=2
    ;;
"scotch")
    ordering_int=3
    ;;
"ptscotch")
    ordering_int=4
    ;;
*)
    printf >&2 "\e[31mERROR\e[0m %s is not a proper ordering parameter (metis, pord, scotch, ptscotch)\n" "$ordering"
    exit 11
    ;;
esac

matrix="${WORKDIR}/Dataset/"
rand=$RANDOM
outputfile="${WORKDIR}/tmp/tmp${rand}.out"

# EAR salloc options
ear_options="--ear=on --ear-user-db=eardata --ear-cpufreq=2100000"
node_list="node[001-021,025-079,081-110,112-125,127-130,132-137,139-212,215]"

if [[ "$synthetic" == "True" ]]; then
    # Log the inputs
    printf "%s %d %s %s %d %d %d %d %d %s " "$job" "$n" "$bandwidth" "$density" "$symmetry" "$num_proc" "$num_threads" "$par" "$icntl_13" "$ordering" >>launch.log

    # Run Analysis stage
    if [[ "$stage" == "Analysis" ]]; then
        # Launch the MUMPS executable with the given matrix size and symmetry
        KMP_AFFINITY=scatter OMP_NESTED=TRUE MKL_NUM_THREADS="$num_threads" OMP_NUM_THREADS="$num_threads" \
            salloc -N 1 -n "$num_proc" -c "$num_threads" --job-name=mumps_run -p cpu_short --mem=32G --time=00:05:00 "$ear_options" --exclude="$node_list" \
            srun ./mumps "-ag" "$type_int" "$n" "$bandwidth" "$density" "$symmetry" "$par" "$icntl_13" "$num_threads" "$ordering_int" \
            1>"$outputfile" 2>>err.out

    # Run Factorization stage
    else
        # Launch the MUMPS executable with the given matrix size and symmetry
        KMP_AFFINITY=scatter OMP_NESTED=TRUE MKL_NUM_THREADS="$num_threads" OMP_NUM_THREADS="$num_threads" \
            salloc -N 1 -n "$num_proc" -c "$num_threads" --job-name=mumps_run -p cpu_short --mem=32G --time=00:05:00 "$ear_options" --exclude="$node_list" \
            srun ./mumps "-fg" "$type_int" "$n" "$bandwidth" "$density" "$symmetry" "$par" "$icntl_13" "$num_threads" "$ordering_int" \
            1>"$outputfile" 2>>err.out
    fi

# Use Matrix in the Dataset
else
    max_n=$(grep -A 5 "KERNEL_INPUTS" ./config.json | tail -n 1 | grep -E -o "[0-9]*")
    div=$(echo "$n / $max_n" | bc -l)
    # Select the matrix according to the inputs
    if [[ $symmetry == "0" ]]; then
        selected=$(echo "($div * 4) / 1" | bc)
        case "$selected" in
        0)
            matrix=$matrix"8-Transport.mtx"
            ;;
        1)
            matrix=$matrix"2-ss.mtx"
            ;;
        2)
            matrix=$matrix"11-vas_stokes_1M.mtx"
            ;;
        3)
            matrix=$matrix"7-ML_Geer.mtx"
            ;;
        *)
            exit 12
            ;;
        esac

    else
        selected=$(echo "($div * 11) / 1" | bc)
        case "$selected" in
        0)
            matrix=$matrix"3-nlpkkt80.mtx"
            ;;
        1)
            matrix=$matrix"18-PFlow_742.mtx"
            ;;
        2)
            matrix=$matrix"14-dielFilterV2real.mtx"
            ;;
        3)
            matrix=$matrix"12-Hook_1489.mtx"
            ;;
        4)
            matrix=$matrix"5-Geo_1438.mtx"
            ;;
        5)
            matrix=$matrix"4-Serena.mtx"
            ;;
        6)
            matrix=$matrix"9-Bump_2911.mtx"
            ;;
        7)
            matrix=$matrix"23-Long_Coup_dt0.mtx"
            ;;
        8)
            matrix=$matrix"15-Flan_1565.mtx"
            ;;
        9)
            matrix=$matrix"19-Cube_Coup_dt0.mtx"
            ;;
        10)
            matrix=$matrix"13-Queen_4147.mtx"
            ;;
        *)
            exit 13
            ;;
        esac
    fi

    # Log the inputs
    printf "%s %s %d %d %d %d %s " "$job" "$matrix" "$num_proc" "$num_threads" "$par" "$icntl_13" "$ordering" >>launch.log

    # Run Analysis stage
    if [[ "$stage" == "Analysis" ]]; then
        # Launch the MUMPS executable with the given matrix size and symmetry
        KMP_AFFINITY=scatter OMP_NESTED=TRUE MKL_NUM_THREADS="$num_threads" OMP_NUM_THREADS="$num_threads" \
            salloc -N 1 -n "$num_proc" -c "$num_threads" --job-name=mumps_run -p cpu_short --mem=96G --time=00:10:00 "$ear_options" --exclude="$node_list" \
            srun ./mumps "-ai" "$matrix" "$par" "$icntl_13" "$num_threads" "$ordering_int" \
            1>"$outputfile" 2>>err.out

    # Run Factorization stage
    else
        # Launch the MUMPS executable with the given matrix size and symmetry
        KMP_AFFINITY=scatter OMP_NESTED=TRUE MKL_NUM_THREADS="$num_threads" OMP_NUM_THREADS="$num_threads" \
            salloc -N 1 -n "$num_proc" -c "$num_threads" --job-name=mumps_run -p cpu_short --mem=96G --time=00:10:00 "$ear_options" --exclude="$node_list" \
            srun ./mumps "-fi" "$matrix" "$par" "$icntl_13" "$num_threads" "$ordering_int" \
            1>"$outputfile" 2>>err.out
    fi
fi

exec_error=$(grep -E "\*\* ERROR RETURN \*\* FROM DMUMPS INFO\(1\)=" "$outputfile" | cut -d '=' -f2)

# Check for an error
if [[ $exec_error != "" ]]; then
    if [[ $(echo "$exec_error" | grep -E -v -- "-7[0-9]") != "" ]]; then
        printf >&2 "\e[31mERROR\e[0m An error occured during the execution\n"
        printf "\n" >>launch.log
        exit 14
    fi
fi

# Retrieve the times mesures by MUMPS
analysis_time=$(grep "Elapsed time in analysis driver" "$outputfile" | grep -E -o "[0-9]*\.[0-9]*")
analysis_energy=$(grep "ANALYSIS EAR" "$outputfile" | cut -d ',' -f1 | grep -E -o "[0-9]*")
factorization_time=$(grep "Elapsed time in factorization driver" "$outputfile" | grep -E -o "[0-9]*\.[0-9]*")
factorization_energy=$(grep "FACTORIZATION EAR" "$outputfile" | cut -d ',' -f1 | grep -E -o "[0-9]*")

# Select the value to return
case "$job" in
"ap")
    result=$analysis_time
    ;;
"ae")
    result=$analysis_energy
    ;;
"fp")
    result=$factorization_time
    ;;
"fe")
    result=$factorization_energy
    ;;
esac

printf "%s\n" "$result" >>launch.log
if [[ -z "$result" ]]; then
    printf >&2 "\e[31mERROR\e[0m An error occured during the execution\n"
    exit 15
fi

# Retrieve the result
echo "$result"

rm -f "$outputfile"
