#!/bin/bash

# Check the inputs

if [[ $# != 9 ]]; then
    printf >&2 "\e[33mUSAGE:\e[0m %s n bandwidth density symmetry num_proc num_threads par inctl13 ordering\n" "$0"
    exit 1
fi

n=$1
bandwidth=$2
density=$3
symmetry=$4
num_proc=$5
num_threads=$6
par=$7
icntl_13=$8
ordering=$9

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
type_int=0

if [[ $n == "" || $(printf "%s" "$n" | grep -E -vo "[1-9][0-9]*") != "" ]]; then
    printf >&2 "\e[31mERROR\e[0m %s is a not a valid integer for N\n" "$n"
    exit 1
fi

if [[ $(echo $bandwidth | grep -E -vo "[1-9][0-9]*") != "" ]]; then
    printf >&2 "\e[31mERROR\e[0m %s is a not a valid integer for bandwidth\n" "$bandwidth"
    exit 1
fi

if [[ $(echo $bandwidth | grep -E -vo "0.[0-9]*") != "" ]]; then
    printf >&2 "\e[31mERROR\e[0m %s is a not a valid real for density\n" "$density"
    exit 1
fi

if [[ $symmetry != "0" && $symmetry != "1" && $symmetry != "2" ]]; then
    printf >&2 "\e[31mERROR\e[0m %s is a wrong symmetry kind\n" "$symmetry"
    exit 1
fi

if [[ $(echo "$num_threads" | grep -E -o "[1-9][0-9]*") == "" ]]; then
    printf >&2 "\e[31mERROR\e[0m %s is a wrong number of threads\n" "$num_threads"
    exit 1
fi

if [[ $(echo "$num_proc" | grep -E -o "[1-9][0-9]*") == "" ]]; then
    printf >&2 "\e[31mERROR\e[0m %s is a wrong number of MPI ranks\n" "$num_threads"
    exit 1
fi

if [[ $(echo "$par" | grep -E -o "(0|1)") == "" ]]; then
    printf >&2 "\e[31mERROR\e[0m %s is not a proper PAR parameter (0 or 1)\n" "$par"
    exit 1
fi

if [[ $(echo "$icntl_13" | grep -E -o "([1-9][0-9]*)|-1|0") == "" ]]; then
    printf >&2 "\e[31mERROR\e[0m %s is not a proper ICNTL 13 parameter (-1, 0 or >0)\n" "$icntl_13"
    exit 1
fi

if [[ $(echo "$ordering" | grep -E -o "[0-4]") == "" ]]; then
    printf >&2 "\e[31mERROR\e[0m %s is not a proper ordering parameter ([0; 4])\n" "$ordering"
    exit 1
fi

rand=$RANDOM
outputfile="tmp${rand}.out"

# Log the inputs
printf "%s %d %d %d %d %d %d %d " "$type" "$n" "$nnz" "$symmetry" "$num_proc" "$num_threads" "$par" "$icntl_13" >> launch.log

# Launch the MUMPS executable with the given matrix size and symmetry
KMP_AFFINITY=scatter OMP_NESTED=TRUE MKL_NUM_THREADS="$num_threads" OMP_NUM_THREADS="$num_threads"\
    salloc -N 1 -n "$num_proc" -c "$num_threads" --job-name=mumps_run -p cpu_short --mem=128G --time=00:05:00\
    srun ./mumps\
    "$type_int" "$n" "$bandwidth" "$density" "$symmetry" "$par" "$icntl_13" "$num_threads" "$ordering"\
    1> "$outputfile" 2>> err.out

OMP_NESTED=TRUE OMP_NUM_THREADS="$num_threads" \
    mpirun -np "$num_proc" --mca accelerator rocm ./mumps "0" "$n" "$bandwidth" "$density" "$symmetry" "$par" "$icntl_13" "$num_threads" 1> "$outputfile" 2>> err.out

exec_error=$(grep -E -o "ERROR RETURN" "$outputfile")

# Check for an error
if [[ $exec_error != "" ]]; then
    printf >&2 "\e[31mERROR\e[0m An error occured during the execution\n"
    printf "\n" >> launch.log
    exit 1
fi

# Retrieve the times mesures by MUMPS
analysis_time=$(grep "Elapsed time in analysis driver" "$outputfile" | grep -E -o "[0-9]*\.[0-9]*")
factorization_time=$(grep "Elapsed time in factorization driver" "$outputfile" | grep -E -o "[0-9]*\.[0-9]*")

# Compute the total time and return it to mlkaps
total=$(echo "scale=4; $analysis_time + $factorization_time" | bc)
printf "%s\n" "$total" >> launch.log
echo "$analysis_time,$factorization_time"

rm -f "$outputfile"
