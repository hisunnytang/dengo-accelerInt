/*

The generalized rate data type holders.

*/


/* stdlib, hdf5, local includes */

#include "omp.h"

#include "time.h"
#include "sys/time.h"
#include "stdlib.h"
#include "math.h"
#include "hdf5.h"
#include "hdf5_hl.h"
#include "stdio.h"
#include "string.h"

/* header files for CVODES/SUNDIALS */
#include <cvode/cvode.h>               /* prototypes for CVODE fcts., consts.  */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <cvode/cvode_direct.h>        /* access to CVDls interface            */
#include <sundials/sundials_types.h>   /* defs. of realtype, sunindextype      */

#ifdef CVKLU
#include <sunmatrix/sunmatrix_sparse.h>
#include <sunlinsol/sunlinsol_klu.h>
#endif


/* User-defined vector and matrix accessor macros: Ith, IJth */

/* These macros are defined in order to write code which exactly matches
   the mathematical problem description given above.

   Ith(v,i) references the ith component of the vector v, where i is in
   the range [1..NEQ] and NEQ is defined below. The Ith macro is defined
   using the N_VIth macro in nvector.h. N_VIth numbers the components of
   a vector starting from 0.

   IJth(A,i,j) references the (i,j)th element of the dense matrix A, where
   i and j are in the range [1..NEQ]. The IJth macro is defined using the
   DENSE_ELEM macro in dense.h. DENSE_ELEM numbers rows and columns of a
   dense matrix starting from 0. */

#define Ith(v,i)    NV_Ith_S(v,i-1)       /* Ith numbers components 1..NEQ */
#define IJth(A,i,j) SM_ELEMENT_D(A,i-1,j-1) /* IJth numbers rows,cols 1..NEQ */


#ifndef MAX_NCELLS
#define MAX_NCELLS 1024
#endif

#ifndef NTHREADS
#define NTHREADS 8
#endif

#define DMAX(A,B) ((A) > (B) ? (A) : (B))
#define DMIN(A,B) ((A) < (B) ? (A) : (B))

 

int cvklu_main(int argc, char **argv);

struct cvklu_data {
    /* All of the network bins will be the same width */
    double dbin;
    double idbin;
    double bounds[2];
    int nbins;

    /* These will be for bins in redshift space */
    double d_zbin;
    double id_zbin;
    double z_bounds[2];
    int n_zbins;

    /* For storing and passing around
       redshift information */
    double current_z;
    double zdef;
    double dz;


    /* Now we do all of our cooling and chemical tables */
    double r_k01[1024];
    double r_k02[1024];
    double r_k03[1024];
    double r_k04[1024];
    double r_k05[1024];
    double r_k06[1024];
    double r_k07[1024];
    double r_k08[1024];
    double r_k09[1024];
    double r_k10[1024];
    double r_k11[1024];
    double r_k12[1024];
    double r_k13[1024];
    double r_k14[1024];
    double r_k15[1024];
    double r_k16[1024];
    double r_k17[1024];
    double r_k18[1024];
    double r_k19[1024];
    double r_k21[1024];
    double r_k22[1024];
    double c_brem_brem[1024];
    double c_ceHeI_ceHeI[1024];
    double c_ceHeII_ceHeII[1024];
    double c_ceHI_ceHI[1024];
    double c_cie_cooling_cieco[1024];
    double c_ciHeI_ciHeI[1024];
    double c_ciHeII_ciHeII[1024];
    double c_ciHeIS_ciHeIS[1024];
    double c_ciHI_ciHI[1024];
    double c_compton_comp_[1024];
    double c_gammah_gammah[1024];
    double c_gloverabel08_gael[1024];
    double c_gloverabel08_gaH2[1024];
    double c_gloverabel08_gaHe[1024];
    double c_gloverabel08_gaHI[1024];
    double c_gloverabel08_gaHp[1024];
    double c_gloverabel08_gphdl[1024];
    double c_gloverabel08_gpldl[1024];
    double c_gloverabel08_h2lte[1024];
    
    double c_h2formation_h2mcool[1024];
    double c_h2formation_h2mheat[1024];
    double c_h2formation_ncrd1[1024];
    double c_h2formation_ncrd2[1024];
    double c_h2formation_ncrn[1024];
    
    double c_reHeII1_reHeII1[1024];
    double c_reHeII2_reHeII2[1024];
    double c_reHeIII_reHeIII[1024];
    double c_reHII_reHII[1024];
    
    int ncells;

    // gamma as a function of temperature
    double g_gammaH2_1[1024];
    double g_dgammaH2_1_dT[1024];

    double g_gammaH2_2[1024];
    double g_dgammaH2_2_dT[1024];

    int nstrip;
    const char *dengo_data_file;
};


/* Declare ctype RHS and Jacobian */
typedef int(*rhs_f)( realtype, N_Vector , N_Vector , void * );
#ifndef CVSPILS
typedef int(*jac_f)( realtype, N_Vector  , N_Vector , SUNMatrix , void *, N_Vector, N_Vector, N_Vector);
#endif
#ifdef CVSPILS
typedef int(*jac_f)(N_Vector , N_Vector , realtype,
             N_Vector, N_Vector,
             void *user_data, N_Vector);
#endif


void *setup_cvode_solver( rhs_f f, jac_f Jac,  int NEQ, 
        cvklu_data *data, SUNLinearSolver LS, SUNMatrix A, N_Vector y, double reltol, N_Vector abstol);

int cvode_solver( void *cvode_mem, double *output, int NEQ, double *dt, cvklu_data * data, N_Vector y, double reltol, N_Vector abstol );

cvklu_data *cvklu_setup_data(const char *, int *, char***);
void cvklu_read_rate_tables(cvklu_data*);
void cvklu_read_cooling_tables(cvklu_data*);
void cvklu_read_gamma(cvklu_data*);
void cvklu_interpolate_gamma(cvklu_data*, int );

void setting_up_extra_variables( cvklu_data * data, double * input, int nstrip );

int dengo_evolve_cvklu (double dtf, double &dt, double z,
                                     double *input, double *rtol,
                                     double *atol, unsigned long dims,
                                     cvklu_data *data, double *temp);

double evolve_in_batches( void * cvode_mem, N_Vector y_vec, N_Vector abstol,  
                          double reltol,double *input, int v_size, int d, int start_idx, 
                          int MAX_ITERATION, double dtf, cvklu_data *data );


 




#ifndef CVSPILS
#ifdef  CVKLU
int calculate_sparse_jacobian_cvklu( realtype t,
                                        N_Vector y, N_Vector fy,
                                        SUNMatrix J, void *user_data,
                                        N_Vector tmp1, N_Vector tmp2,
                                        N_Vector tmp3);
#else
int calculate_jacobian_cvklu( realtype t,
               N_Vector y, N_Vector fy, SUNMatrix J, void *user_data,
               N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
#endif
#endif

#ifdef CVSPILS
int calculate_JacTimesVec_cvklu
            (N_Vector v, N_Vector Jv, realtype t,
             N_Vector y, N_Vector fy,
             void *user_data, N_Vector tmp);
#endif

int calculate_rhs_cvklu(realtype t, N_Vector y, N_Vector ydot, void *user_data);
void ensure_electron_consistency(double *input, int nstrip, int nchem);
void temperature_from_mass_density(double *input, int nstrip, int nchem, 
                                   double *strip_temperature);

int cvklu_calculate_temperature(cvklu_data *data, double *input, int nstrip, int nchem);




typedef struct dengo_field_data
{

  unsigned long int nstrip;
  unsigned long int ncells; 
  // This should be updated dynamically 
  // with dengo
  double *density;
  double *H2_1_density;
  
  double *H2_2_density;
  
  double *H_1_density;
  
  double *H_2_density;
  
  double *H_m0_density;
  
  double *He_1_density;
  
  double *He_2_density;
  
  double *He_3_density;
  
  double *de_density;
  
  double *ge_density;
  
    
  double *CoolingTime;
  double *MolecularWeight;
  double *temperature;
  double *Gamma;

  const char *dengo_data_file;
} dengo_field_data;

typedef struct code_units
{

  int comoving_coordinates;
  double density_units;
  double length_units;
  double time_units;
  double velocity_units;
  double a_units;
  double a_value;

} code_units;

int cvklu_solve_chemistry_dt( code_units *units, dengo_field_data *field_data, double dt );

int dengo_estimate_cooling_time( code_units* units, dengo_field_data * field_data );

int cvklu_calculate_cooling_timescale( double *cooling_time, double *input, int nstrip, cvklu_data *data);
int reshape_to_dengo_field_data( code_units* units, dengo_field_data *field_data, double* input );
int flatten_dengo_field_data( code_units* units, dengo_field_data *field_data, double *input );

int dengo_calculate_temperature( code_units*, dengo_field_data* );
int dengo_calculate_gamma( double* gamma_eff, cvklu_data*, double*, int );
int dengo_calculate_mean_molecular_weight( code_units*, dengo_field_data * );
