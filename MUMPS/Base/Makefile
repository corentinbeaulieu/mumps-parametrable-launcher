
CC=mpiicx

LMETIS=-L/gpfs/users/beaulieuc/Installs/metis/lib -lmetis
LSPRAL=-L/gpfs/users/beaulieuc/Installs/spral/INSTALL/lib -lspral
LAPACK=-L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core
SCALAP=-L$(MKLROOT)/lib/intel64 -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64
LIBPAR= $(SCALAP) $(LAPACK) -lm -ldl
LIBBLAS=-lmkl_blas95_lp64
LIBBLACS=-lmkl_blacs_intelmpi_lp64
LORDERINGS=${LMETIS} -lpord
LMUMPS=-L/gpfs/users/beaulieuc/Installs/MUMPS/MUMPS_5.6.2/lib -ldmumps -lzmumps -lmumps_common
IFLAGS=-I$(MKLROOT)/include -I$(MKLROOT)/include/intel64 -I/gpfs/users/beaulieuc/Installs/MUMPS/MUMPS_5.6.2/include -I/gpfs/users/beaulieuc/Installs/spral/INSTALL/include
LDFLAGS=${LORDERINGS} ${LMUMPS} ${LIBPAR} ${LIBBLAS} ${LIBBLACS} ${LSPRAL} -lpthread
WFLAGS=-Wall -Wextra -Wpedantic

.PHONY: all clean

all: mumps

mumps: mumps.c
	${CC} -O1 -std=c2x ${WFLAGS} ${IFLAGS} $< -o $@ ${LDFLAGS} -qopenmp

clean:
	rm -f mumps

