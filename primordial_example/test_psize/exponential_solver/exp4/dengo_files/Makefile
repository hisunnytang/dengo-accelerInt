CC=g++
NVCC=nvcc
CXXFLAGS= -fopenmp -Wextra -O3 -std=c++11 -g
CUDAFLAGS= -m64 -Xptxas -O3 -std=c++11 -arch=sm_60 -Xcompiler -fopenmp --ftz=false --prec-div=true --prec-sqrt=true --fmad=false -g -dc
LIBS = -lstdc++ -lcuda -lcudart -L/opt/intel/compilers_and_libraries_2016.2.181/linux/mkl/lib/intel64 -lm -lmkl_rt -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -ldl -lmkl_mc -lmkl_def -lhdf5 -lhdf5_hl
 
LIBDIRS  = -L/usr/local/cuda/samples/common/lib/linux/x86_64//linux -L/usr/local/cuda/lib64
INCDIRS  = -I. -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda/include
INCDIRS += -I../exp4_integrator -I../utilities -I../dengo_files

SOURCES =$(wildcard *.cu)
SOURCES+=$(wildcard *.C)
TMP     = $(patsubst %.cu, %.cu.o, $(SOURCES))
OBJECTS = $(patsubst %.C, %.o, $(TMP))


../out/%.cu.o: %.cu  
	$(NVCC) $(INCDIRS) $(CUDAFLAGS) -c $< -o $@ $(LIBDIRS) $(LIBS) -DEXP4 -DSCALE_INPUT 

../out/%.o: %.C  
	$(CC) $(INCDIRS) $(CXXFLAGS) -c $< -o $@ $(LIBDIRS) $(LIBS)



OUT=$(addprefix ../out/, $(OBJECTS) )

all: $(OUT)
	echo $(OUT)
clean:
	rm -f ../out/*o
