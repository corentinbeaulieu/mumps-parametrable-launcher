#!/bin/bash

nnz=$1
symmetric=$2
num_proc=$3
num_threads=$4
par=$5
icntl_13=$6

# Check the inputs
if [[ $nnz == "" || $(printf "$nnz" | grep -E -vo "[1-9][0-9]*") != "" ]]; then
    >&2 printf "\e[31mERROR\e[0m $nnz is a not a valid integer\n"
    exit 1
fi

if [[ $symmetric != "0" && $symmetric != "1" ]]; then
    >&2 printf "\e[31mERROR\e[0m $symmetric is a wrong symmetry kind\n"
    exit 1
fi

if [[ $(echo $num_threads | grep -E -o "[1-9][0-9]*") == "" ]]; then
    >&2 printf "\e[31mERROR\e[0m $num_threads is a wrong number of threads\n"
    exit 1
fi

if [[ $(echo $num_proc | grep -E -o "[1-9][0-9]*") == "" ]]; then
    >&2 printf "\e[31mERROR\e[0m $num_threads is a wrong number of MPI ranks\n"
    exit 1
fi

if [[ $(echo $par | grep -E -o "(0|1)") == "" ]]; then
    >&2 printf "\e[31mERROR\e[0m $par is not a proper PAR parameter (0 or 1)\n"
    exit 1
fi

if [[ $(echo $icntl_13 | grep -E -o "([1-9][0-9]*)|-1|0") == "" ]]; then
    >&2 printf "\e[31mERROR\e[0m $icntl_13 is not a proper ICNTL 13 parameter (-1, 0 or >0)\n"
    exit 1
fi


# Matrix properties
#    Name    |     nnz     | symmetric |          name 
# -----------|-------------|-----------|-------------------------
# nlpkkt     |  14,883,536 | true      |  3-nlpkkt80.mtx
# Pflow      |  18,940,627 | true      | 18-PFlow_742.mtx
# dielFilter |  24,848,204 | true      | 14-dielFilterV2real.mtx
# Hook       |  31,207,734 | true      | 12-Hook_1489.mtx
# Geo        |  32,297,325 | true      |  5-Geo_1438.mtx
# Serena     |  32,961,525 | true      |  4-Serena.mtx
# Bump       |  34,767,207 | true      |  9-Bump_2911.mtx
# Long       |  44,279,572 | true      | 23-Long_Coup_dt0.mtx
# Flan       |  59,485,419 | true      | 15-Flan_1565.mtx
# Cube       |  64,685,452 | true      | 19-Cube_Coup_dt0.mtx
# Queen      | 166,823,197 | true      | 13-Queen_4147.mtx
# -----------|-------------|-----------|-------------------------
# Transport  |  23,500,731 | false     |  8-Transport.mtx
# ss         |  34,753,577 | false     |  2-ss.mtx
# vas_stokes |  34,767,207 | false     | 11-vas_stokes_1M.mtx
# ML_Geer    | 110,879,972 | false     |  7-ML_Geer.mtx

matrix=$WORKDIR"/Dataset/"

# Select the matrix according to the inputs
if [[ $symmetric == "0" ]]; then 
    if [[ $nnz -lt 29127154 ]]; then
        matrix=$matrix"8-Transport.mtx"
    elif [[ $nnz -lt 34760392 ]]; then
        matrix=$matrix"2-ss.mtx"
    elif [[ $nnz -lt 72823589 ]]; then
        matrix=$matrix"11-vas_stokes_1M.mtx"
    else
        matrix=$matrix"7-ML_Geer.mtx"
    fi

else
    if [[ $nnz -lt 16912081 ]]; then
        matrix=$matrix"3-nlpkkt80.mtx"
    elif [[ $nnz -lt 21894415 ]]; then
        matrix=$matrix"18-PFlow_742.mtx"
    elif [[ $nnz -lt 28027969 ]]; then
        matrix=$matrix"14-dielFilterV2real.mtx"
    elif [[ $nnz -lt 31752529 ]]; then
        matrix=$matrix"12-Hook_1489.mtx"
    elif [[ $nnz -lt 32629425 ]]; then
        matrix=$matrix"5-Geo_1438.mtx"
    elif [[ $nnz -lt 33864366 ]]; then
        matrix=$matrix"4-Serena.mtx"
    elif [[ $nnz -lt 39523389 ]]; then
        matrix=$matrix"9-Bump_2911.mtx"
    elif [[ $nnz -lt 51882495 ]]; then
        matrix=$matrix"23-Long_Coup_dt0.mtx"
    elif [[ $nnz -lt 62085435 ]]; then
        matrix=$matrix"15-Flan_1565.mtx"
    elif [[ $nnz -lt 115754324 ]]; then
        matrix=$matrix"19-Cube_Coup_dt0.mtx"
    else
        matrix=$matrix"13-Queen_4147.mtx"
    fi

fi

outputfile="tmp${RANDOM}.out"

# Log the inputs and the selection
printf "$nnz $symmetric $matrix $num_proc $num_threads $icntl_13" >> launch.log

# Launch the MUMPS executable on the selected matrix
KMP_AFFINITY=scatter MKL_NUM_THREADS=$num_threads OMP_NUM_THREADS=$num_threads\
    salloc -N 1 -n $num_proc -c $num_threads --job-name=mumps_run -p cpu_short --mem=42G --time=00:40:00\
    srun -N 1 -n $num_proc ./mumps\
    $matrix $par $icntl_13 $num_threads 1> $outputfile 2> err.out

# Retrieve the times mesures by MUMPS
analysis_time=$(grep "Elapsed time in analysis driver" $outputfile | grep -E -o "[0-9]*\.[0-9]*")
factorization_time=$(grep "Elapsed time in factorization driver" $outputfile | grep -E -o "[0-9]*\.[0-9]*")
solve_time=$(grep "Elapsed time in solve driver" $outputfile | grep -E -o "[0-9]*\.[0-9]*")

# Check for an error
if [[ -z $analysis_time || -z $factorization_time || -z $solve_time ]]; then
    >&2 printf "\e[31mERROR\e[0m the execution went wrong\n"
    printf "\n" >> launch.log
    exit 1
fi

# Compute the total time and return it to mlkaps
total=$(echo "scale=4; $analysis_time + $factorization_time + $solve_time" | bc)
printf "$total\n" >> launch.log
echo $total

rm -f $outputfile
