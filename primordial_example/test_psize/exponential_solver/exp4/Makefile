CC=g++
NVCC=nvcc
CXXFLAGS= -fopenmp -Wextra -O3 -std=c++11 -g
CUDAFLAGS= -m64 -Xptxas -O3 -std=c++11 -arch=sm_60 -Xcompiler -fopenmp --ftz=false --prec-div=true --prec-sqrt=true --fmad=false -g -dc
LIBS = -lstdc++ -lcuda -lcudart -L/opt/intel/compilers_and_libraries_2016.2.181/linux/mkl/lib/intel64 -lm -lmkl_rt -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -ldl -lmkl_mc -lmkl_def -lhdf5 -lhdf5_hl
 
LIBDIRS = -L/usr/local/cuda/samples/common/lib/linux/x86_64//linux -L/usr/local/cuda/lib64
INCDIRS =-I/usr/local/cuda/samples/common/inc -I/usr/local/cuda/include -I../dengo_files


SRC = dydt.cu gpu_memory.cu jacob.cu mechanism.cu sparse_multiplier.cu \
      solver_generic/complexInverse.o solver_generic/inverse.o \
      exp4/exp4

DENGO_DIR = dengo_files
EXP4_DIR  = exp4_integrator
UTIL_DIR  = utilities
SOLV_DIR  = solver_generic

all:
	mkdir -p out
	+$(MAKE) -C $(DENGO_DIR)
	+$(MAKE) -C $(EXP4_DIR)
	+$(MAKE) -C $(SOLV_DIR)
	+$(MAKE) -C $(UTIL_DIR)
	$(NVCC) $(wildcard out/*.o) $(LIBDIRS) $(LIBS) -dlink -o exp4-int-gpu-dlink.o 
	gcc -fPIC -fopenmp -Xlinker -rpath /usr/local/cuda/lib64 $(wildcard out/*.o) exp4-int-gpu-dlink.o  $(LIBDIRS) $(LIBS) -lfftw3 -o exp4-int-gpu 
clean:
	rm -f out/*o
