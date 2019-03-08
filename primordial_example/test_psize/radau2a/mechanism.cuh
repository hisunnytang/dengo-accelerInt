#ifndef MECHANISM_cuh
#define MECHANISM_cuh

#ifdef __GNUG__
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "launch_bounds.cuh"
#include "gpu_macros.cuh"
#include "dengo_solver.h"
#endif
//last_spec 8
/* Species Indexes
0  H
1  H2
2  O
3  OH
4  H2O
5  O2
6  HO2
7  H2O2
8  AR
9  HE
10  CO
11  CO2
12  N2
*/

/*
//Number of species
#define NSP 13
//Number of variables. NN = NSP + 1 (temperature)
#define NN 14
//Number of forward reactions
#define FWD_RATES 27
//Number of reversible reactions
#define REV_RATES 27
//Number of reactions with pressure modified rates
#define PRES_MOD_RATES 6
*/



/* Species Indexes
0  H2_1
1  H2_2
2  H_1
3  H_2
4  H_m0
5  He_1
6  He_2
7  He_3
8  de
9  ge
*/

// should also print the reactions here 
// or separate files

//Number of species
#define NSP 10
//Number of variables. NN = NSP + 1 (temperature)
#define NN 11
//We dont distinguish fwd/ rev rates
//We have only reaction and cooling rates
#define REACTION_RATES 23
#define COOLING_RATES 26

struct mechanism_memory {
  double * y;
  double * dy;
  double * reaction_rates;
  double * cooling_rates;
  double * temperature;
  double * density;
  double * var;
  const char * rateData_location;
  cvklu_data * chemistry_data;
  double *scale;
  double *inv_scale;
  double *temp_array;
  double *jac;
  double *drrate_dT;
  double *dcrate_dT;
  double *h2_optical_depth_approx;
  double *dTs_ge;
};


//Must be implemented by user on a per mechanism basis in mechanism.cu
void set_same_initial_conditions(int, double**, double**);

#if defined (RATES_TEST) || defined (PROFILER)
    void write_jacobian_and_rates_output(int NUM);
#endif
//apply masking of ICs for cache optimized mechanisms
void apply_mask(double*);
void apply_reverse_mask(double*);
#endif

