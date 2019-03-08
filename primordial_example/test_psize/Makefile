CC=g++
NVCC=nvcc
CXXFLAGS= -fopenmp -Wextra -O3 -std=c++11 -g
CUDAFLAGS= -m64 -Xptxas -O3 -std=c++11 -arch=sm_60 -Xcompiler -fopenmp --ftz=false --prec-div=true --prec-sqrt=true --fmad=false -g -dc -c
LIBS = -lopenblas -lpthread -lcudart -lcublas -lhdf5 -lhdf5_hl
LIBDIRS = -L/usr/local/cuda/samples/common/lib/linux/x86_64//linux -L/usr/local/cuda/lib64
INCDIRS =-I/usr/local/cuda/samples/common/inc -I/usr/local/cuda/include
INCDIRS+=-I/home/kwoksun2/sundials-3.2.1/instdir/include


%.o: %.cu  
	$(NVCC) $(INCDIRS) $(CUDAFLAGS) -c $< -o $@ $(LIBDIRS) $(LIBS) -DRKC

my_dlink.o:
	$(NVCC) $(INCDIRS) -arch=sm_60 dydt.o gpu_memory.o solver_generic.o solver_interface.o rkc.o rkc_init.o rkc_props.o mechanism.o check_memory_usuage.o $(LIBDIR) $(LIBS) -dlink -o my_dlink.o

test: dydt.o rkc.o rkc_props.o rkc_init.o solver_generic.o check_memory_usuage.o gpu_memory.o rkc_init.o  solver_interface.o mechanism.o my_dlink.o
	$(CC) -fPIC -fopenmp -O3 -Xlinker -rpath /usr/local/cuda/lib64 dydt.o gpu_memory.o solver_generic.o solver_interface.o rkc.o rkc_init.o rkc_props.o mechanism.o check_memory_usuage.o  my_dlink.o $(LIBDIRS) $(LIBS) -o test 

clean:
	rm -f *o
	rm -f test
