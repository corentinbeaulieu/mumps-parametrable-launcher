
CC=mpicc

LMETIS=-L/gpfs/users/beaulieuc/Installs/metis/lib
LAPACK=-L$(MKLROOT)/lib/intel64 -lmkl_lapack95_lp64
SCALAP=-L$(MKLROOT)/lib/intel64 -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64
LIBPAR=-Wl,--no-as-needed $(SCALAP) $(LAPACK) -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm
LIBBLAS=-lmkl_blas95_lp64
LIBBLACS=-lmkl_blacs_intelmpi_lp64
LORDERINGS=${LMETIS}
LMUMPS=-L/gpfs/users/beaulieuc/Installs/MUMPS/MUMPS_5.6.2/lib -ldmumps -lmumps_common
IFLAGS=-I$(MKLROOT)/include -I/gpfs/users/beaulieuc/Installs/MUMPS/MUMPS_5.6.2/include 
LDFLAGS=${LORDERINGS} ${LMUMPS} ${LIBPAR} ${LIBBLAS} ${LIBBLACS} -lpthread

.PHONY: all clean

all: mumps

mumps: mumps.c
	${CC} -std=c11 ${IFLAGS} $< -o $@ ${LDFLAGS}

clean:
	rm -f mumps

