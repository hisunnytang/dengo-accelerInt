#ifndef MECHANISM_h
#define MECHANISM_h

#include <string.h>
//Number of species
#define NSP 10
//Number of variables. NN = NSP + 1 (temperature)
#define NN 11
//We dont distinguish fwd/ rev rates
//We have only reaction and cooling rates
#define REACTION_RATES 23
#define COOLING_RATES 27

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
  int *rhs_call;
  int *jac_call;
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

