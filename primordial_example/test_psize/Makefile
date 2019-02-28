CC=g++
NVCC=nvcc
CXXFLAGS= -fopenmp -Wextra -O3 -std=c++11 -g
CUDAFLAGS= -m64 -Xptxas -O3 -v -std=c++11 -arch=sm_60 -Xcompiler -fopenmp --ftz=false --prec-div=true --prec-sqrt=true --fmad=false --profile -lineinfo -dc -c
LIBS = -lopenblas -lpthread -lcudart -lcublas -lhdf5 -lhdf5_hl
LIBDIRS = -L/usr/local/cuda/samples/common/lib/linux/x86_64//linux -L/usr/local/cuda/lib64
INCDIRS =-I/usr/local/cuda/samples/common/inc -I/usr/local/cuda/include
INCDIRS+=-I/home/kwoksun2/sundials-3.2.1/instdir/include


%.cu.o: %.cu  
	$(NVCC) $(INCDIRS) $(CUDAFLAGS) -c $< -o $@ $(LIBDIRS) $(LIBS) -DRKC

my_dlink.o:
	$(NVCC) $(INCDIRS) -arch=sm_60 dydt.cu.o gpu_memory.cu.o solver_generic.cu.o solver_interface.cu.o rkc.cu.o rkc_init.cu.o rkc_props.cu.o mechanism.cu.o check_memory_usuage.cu.o $(LIBDIR) $(LIBS) -dlink -o my_dlink.o

test: dydt.cu.o rkc.cu.o rkc_props.cu.o rkc_init.cu.o solver_generic.cu.o check_memory_usuage.cu.o gpu_memory.cu.o rkc_init.cu.o  solver_interface.cu.o mechanism.cu.o my_dlink.o
	$(CC) -fPIC -fopenmp -Xlinker -rpath /usr/local/cuda/lib64 dydt.cu.o gpu_memory.cu.o solver_generic.cu.o solver_interface.cu.o rkc.cu.o rkc_init.cu.o rkc_props.cu.o mechanism.cu.o check_memory_usuage.cu.o  my_dlink.o $(LIBDIRS) $(LIBS) -o test 

clean:
	rm -f *o
	rm -f test
