CC=g++
NVCC=nvcc
CXXFLAGS= -fopenmp -Wextra -O3 -std=c++11 -g
CUDAFLAGS= -m64 -Xptxas -O3 -std=c++11 -arch=sm_60 -Xcompiler -fopenmp --ftz=false --prec-div=true --prec-sqrt=true --fmad=false -g -dc
LIBS = -lstdc++ -lcuda -lcudart -L/opt/intel/compilers_and_libraries_2016.2.181/linux/mkl/lib/intel64 -lm -lmkl_rt -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -ldl -lmkl_mc -lmkl_def -lhdf5 -lhdf5_hl
 
LIBDIRS = -L/usr/local/cuda/samples/common/lib/linux/x86_64//linux -L/usr/local/cuda/lib64
INCDIRS =-I/usr/local/cuda/samples/common/inc -I/usr/local/cuda/include
INCDIRS+=-I/home/kwoksun2/sundials-3.2.1/instdir/include


%.o: %.cu  
	$(NVCC) $(INCDIRS) $(CUDAFLAGS) -c $< -o $@ $(LIBDIRS) $(LIBS) -DRADAU2A 

%.o: %.C  
	$(CC) $(INCDIRS) $(CXXFLAGS) -c $< -o $@ $(LIBDIRS) $(LIBS) -DRADAU2A 


my_dlink.o:
	$(NVCC) $(INCDIRS) -arch=sm_60 dydt.o jacob.o gpu_memory.o solver_generic.o solver_interface.o radau2a.o radau2a_init.o radau2a_props.o mechanism.o complexInverse.o inverse.o dengo_solver.o check_memory_usuage.o $(LIBDIR) $(LIBS) -dlink -o my_dlink.o

test: dydt.o jacob.o gpu_memory.o solver_generic.o solver_interface.o radau2a.o radau2a_init.o radau2a_props.o mechanism.o check_memory_usuage.o inverse.o complexInverse.o dengo_solver.o my_dlink.o
	$(CC) -fPIC -fopenmp -O3 -Xlinker -rpath /usr/local/cuda/lib64 dydt.o jacob.o gpu_memory.o solver_generic.o solver_interface.o complexInverse.o inverse.o radau2a.o radau2a_init.o radau2a_props.o mechanism.o dengo_solver.o check_memory_usuage.o  my_dlink.o $(LIBDIRS) $(LIBS) -o test 

clean:
	rm -f *o
	rm -f test
