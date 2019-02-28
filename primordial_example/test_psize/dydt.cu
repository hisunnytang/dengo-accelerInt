#include "header.cuh"
#include "gpu_memory.cuh"

__device__ void interpolate_gamma( cvklu_data *rate_data, double T, double *gamma, double *dgamma_dT )
{
  int tid, bin_id, zbin_id;
  double t1, t2;
  double Tdef, log_temp_out;
  int no_photo = 0;
  double lb = log(rate_data->bounds[0]);
 
  log_temp_out = log(T);
  bin_id = (int) ( rate_data->idbin * ( log_temp_out -  lb ) );
  if ( bin_id <= 0) {
    bin_id = 0;
  } else if ( bin_id >= rate_data->nbins) {
    bin_id = rate_data->nbins - 1;
  }
    
  //printf( "bin_id = %d; temp_out = %0.5g \n", bin_id, temp_out[tid]);
  t1 = (lb + (bin_id    ) * rate_data->dbin);
  t2 = (lb + (bin_id + 1) * rate_data->dbin);
  Tdef = (log_temp_out - t1)/(t2 - t1);

  *gamma     = rate_data->g_gammaH2_1[bin_id] + Tdef * (rate_data->g_gammaH2_1[bin_id+1] - rate_data->g_gammaH2_1[bin_id]);
  *dgamma_dT = rate_data->g_dgammaH2_1_dT[bin_id] + Tdef * (rate_data->g_dgammaH2_1_dT[bin_id+1] - rate_data->g_dgammaH2_1_dT[bin_id]);

}



__device__ void evaluate_temperature( double* T, const double __const__ *y, cvklu_data *rate_data )
{
   // iterate temperature to convergence
  double t, tnew, tdiff;
  double dge, dge_dT;
  double gammaH2, dgammaH2_dT, _gammaH2_m1;
  
  int count = 0;
  int MAX_ITERATION = 100; 
  double gamma     = 5./3.;
  double _gamma_m1 = 1.0 / (gamma - 1.0);
  double kb = 1.2806504e-16; // Boltzamann constant [erg/K] 
  int tid = threadIdx.x + blockDim.x * blockIdx.x; 
  double mdensity = 1.0e14 * 1.67e-24; 
  // prepare t, tnew for the newton's iteration;
  t     = *T;
  tnew  = 1.1*t;
  tdiff = tnew - t;
 
  while ( tdiff/ tnew > 0.001 ){
    // We do Newton's Iteration to calculate the temperature
    // Since gammaH2 is dependent on the temperature too!
    interpolate_gamma( rate_data, t, &gammaH2, &dgammaH2_dT );
    _gammaH2_m1 = 1.0 / (gammaH2 - 1.0);

    dge_dT = t*kb*(-y[INDEX(0)]*_gammaH2_m1*_gammaH2_m1*dgammaH2_dT - y[INDEX(1)]*_gammaH2_m1*_gammaH2_m1*dgammaH2_dT)/(mdensity) 
               + kb*(y[INDEX(0)]*_gammaH2_m1 + y[INDEX(1)]*_gammaH2_m1 + y[INDEX(2)]*_gamma_m1 + y[INDEX(3)]*_gamma_m1 + y[INDEX(4)]*_gamma_m1 
               + y[INDEX(5)]*_gamma_m1 + y[INDEX(6)]*_gamma_m1 + y[INDEX(7)]*_gamma_m1 + _gamma_m1*y[INDEX(8)])/(mdensity);
    dge    = t*kb*(y[INDEX(0)]*_gammaH2_m1 + y[INDEX(1)]*_gammaH2_m1 + y[INDEX(2)]*_gamma_m1 + y[INDEX(3)]*_gamma_m1 
           + y[INDEX(4)]*_gamma_m1 + y[INDEX(5)]*_gamma_m1 + y[INDEX(6)]*_gamma_m1 + y[INDEX(7)]*_gamma_m1 + _gamma_m1*y[INDEX(8)])/(mdensity) - y[INDEX(9)]; 

    //This is the change in ge for each iteration
    tnew = t - dge/dge_dT;
    count += 1;

    tdiff = fabs(t - tnew);
    t     = tnew;
    if (count > MAX_ITERATION){
      printf("T[tid = %d] failed to converge (iteration: %d); at T = %0.3g \n", tid, count, tnew );
    }
    if (t != t){
      printf("T[tid = %d] is NaN, count = %d; ge = %0.5g, gamma_H2 = %0.5g \n", tid, count, y[INDEX(9)], gammaH2);
      t = 1000.0;
      for (int i = 0; i < 10; i++){
          printf("y[INDEX(%d)] = %0.5g \n", i, y[INDEX(i)]);
      }
      break;
    }

  }
  // update the temperature;
  *T = t;

}



__device__ void interpolate_reaction_rates( double *reaction_rates_out, double temp_out, cvklu_data *rate_data)
{
    
    int tid, bin_id, zbin_id;
    double t1, t2;
    double Tdef, log_temp_out;
    int no_photo = 0;
    double lb = log(rate_data->bounds[0]);

    tid = threadIdx.x + blockDim.x * blockIdx.x;
 
    log_temp_out = log(temp_out);
    bin_id = (int) ( rate_data->idbin * ( log_temp_out -  lb ) );
    if ( bin_id <= 0) {
        bin_id = 0;
    } else if ( bin_id >= rate_data->nbins) {
        bin_id = rate_data->nbins - 1;
    }
    
    //printf( "bin_id = %d; temp_out = %0.5g \n", bin_id, temp_out[tid]);
    t1 = (lb + (bin_id    ) * rate_data->dbin);
    t2 = (lb + (bin_id + 1) * rate_data->dbin);
    Tdef = (log_temp_out - t1)/(t2 - t1);

    // rate_out is a long 1D array
    // NRATE is the number of rate required by the solver network
    reaction_rates_out[ 0] = rate_data->r_k01[bin_id] + Tdef * (rate_data->r_k01[bin_id+1] - rate_data->r_k01[bin_id]);
    reaction_rates_out[ 1] = rate_data->r_k02[bin_id] + Tdef * (rate_data->r_k02[bin_id+1] - rate_data->r_k02[bin_id]);
    reaction_rates_out[ 2] = rate_data->r_k03[bin_id] + Tdef * (rate_data->r_k03[bin_id+1] - rate_data->r_k03[bin_id]);
    reaction_rates_out[ 3] = rate_data->r_k04[bin_id] + Tdef * (rate_data->r_k04[bin_id+1] - rate_data->r_k04[bin_id]);
    reaction_rates_out[ 4] = rate_data->r_k05[bin_id] + Tdef * (rate_data->r_k05[bin_id+1] - rate_data->r_k05[bin_id]);
    reaction_rates_out[ 5] = rate_data->r_k06[bin_id] + Tdef * (rate_data->r_k06[bin_id+1] - rate_data->r_k06[bin_id]);
    reaction_rates_out[ 6] = rate_data->r_k07[bin_id] + Tdef * (rate_data->r_k07[bin_id+1] - rate_data->r_k07[bin_id]);
    reaction_rates_out[ 7] = rate_data->r_k08[bin_id] + Tdef * (rate_data->r_k08[bin_id+1] - rate_data->r_k08[bin_id]);
    reaction_rates_out[ 8] = rate_data->r_k09[bin_id] + Tdef * (rate_data->r_k09[bin_id+1] - rate_data->r_k09[bin_id]);
    reaction_rates_out[ 9] = rate_data->r_k10[bin_id] + Tdef * (rate_data->r_k10[bin_id+1] - rate_data->r_k10[bin_id]);
    reaction_rates_out[10] = rate_data->r_k11[bin_id] + Tdef * (rate_data->r_k11[bin_id+1] - rate_data->r_k11[bin_id]);
    reaction_rates_out[11] = rate_data->r_k12[bin_id] + Tdef * (rate_data->r_k12[bin_id+1] - rate_data->r_k12[bin_id]);
    reaction_rates_out[12] = rate_data->r_k13[bin_id] + Tdef * (rate_data->r_k13[bin_id+1] - rate_data->r_k13[bin_id]);
    reaction_rates_out[13] = rate_data->r_k14[bin_id] + Tdef * (rate_data->r_k14[bin_id+1] - rate_data->r_k14[bin_id]);
    reaction_rates_out[14] = rate_data->r_k15[bin_id] + Tdef * (rate_data->r_k15[bin_id+1] - rate_data->r_k15[bin_id]);
    reaction_rates_out[15] = rate_data->r_k16[bin_id] + Tdef * (rate_data->r_k16[bin_id+1] - rate_data->r_k16[bin_id]);
    reaction_rates_out[16] = rate_data->r_k17[bin_id] + Tdef * (rate_data->r_k17[bin_id+1] - rate_data->r_k17[bin_id]);
    reaction_rates_out[17] = rate_data->r_k18[bin_id] + Tdef * (rate_data->r_k18[bin_id+1] - rate_data->r_k18[bin_id]);
    reaction_rates_out[18] = rate_data->r_k19[bin_id] + Tdef * (rate_data->r_k19[bin_id+1] - rate_data->r_k19[bin_id]);
    // reaction_rates_out[19] = rate_data->r_k20[bin_id] + Tdef * (rate_data->r_k20[bin_id+1] - rate_data->r_k20[bin_id]);
    reaction_rates_out[20] = rate_data->r_k21[bin_id] + Tdef * (rate_data->r_k21[bin_id+1] - rate_data->r_k21[bin_id]);
    reaction_rates_out[21] = rate_data->r_k22[bin_id] + Tdef * (rate_data->r_k22[bin_id+1] - rate_data->r_k22[bin_id]);
    // reaction_rates_out[22] = rate_data->r_k23[bin_id] + Tdef * (rate_data->r_k23[bin_id+1] - rate_data->r_k23[bin_id]);
    // printf( "reaction rates done from %d thread; k22: %0.5g \n", tid, reaction_rates_out[ tid*NRATE + 21 ] );

}

__device__ void interpolate_cooling_rates( double *cooling_rates_out, double temp_out, cvklu_data *rate_data)
{
    
    int tid, bin_id, zbin_id;
    double t1, t2;
    double Tdef, log_temp_out;
    int no_photo = 0;
    double lb = log(rate_data->bounds[0]);

    tid = threadIdx.x + blockDim.x * blockIdx.x;
 
    log_temp_out = log(temp_out);
    bin_id = (int) ( rate_data->idbin * ( log_temp_out -  lb ) );
    // printf( "bin_id = %d; log_temp_out = %0.5g \n", bin_id, log_temp_out);
    if ( bin_id <= 0) {
        bin_id = 0;
    } else if ( bin_id >= rate_data->nbins) {
        bin_id = rate_data->nbins - 1;
    }
    t1 = (lb + (bin_id    ) * rate_data->dbin);
    t2 = (lb + (bin_id + 1) * rate_data->dbin);
    Tdef = (log_temp_out - t1)/(t2 - t1);

    // rate_out is a long 1D array
    // NRATE is the number of rate required by the solver network
    int NCOOL = 26;

    cooling_rates_out[ 0] = rate_data->c_ceHI_ceHI[bin_id] + Tdef * (rate_data->c_ceHI_ceHI[bin_id+1] - rate_data->c_ceHI_ceHI[bin_id]);
    cooling_rates_out[ 1] = rate_data->c_ceHeI_ceHeI[bin_id] + Tdef * (rate_data->c_ceHeI_ceHeI[bin_id+1] - rate_data->c_ceHeI_ceHeI[bin_id]);
    cooling_rates_out[ 2] = rate_data->c_ceHeII_ceHeII[bin_id] + Tdef * (rate_data->c_ceHeII_ceHeII[bin_id+1] - rate_data->c_ceHeII_ceHeII[bin_id]);
    cooling_rates_out[ 3] = rate_data->c_ciHeIS_ciHeIS[bin_id] + Tdef * (rate_data->c_ciHeIS_ciHeIS[bin_id+1] - rate_data->c_ciHeIS_ciHeIS[bin_id]);
    cooling_rates_out[ 4] = rate_data->c_ciHI_ciHI[bin_id] + Tdef * (rate_data->c_ciHI_ciHI[bin_id+1] - rate_data->c_ciHI_ciHI[bin_id]);
    cooling_rates_out[ 5] = rate_data->c_ciHeI_ciHeI[bin_id] + Tdef * (rate_data->c_ciHeI_ciHeI[bin_id+1] - rate_data->c_ciHeI_ciHeI[bin_id]);
    cooling_rates_out[ 6] = rate_data->c_ciHeII_ciHeII[bin_id] + Tdef * (rate_data->c_ciHeII_ciHeII[bin_id+1] - rate_data->c_ciHeII_ciHeII[bin_id]);
    cooling_rates_out[ 7] = rate_data->c_reHII_reHII[bin_id] + Tdef * (rate_data->c_reHII_reHII[bin_id+1] - rate_data->c_reHII_reHII[bin_id]);
    cooling_rates_out[ 8] = rate_data->c_reHeII1_reHeII1[bin_id] + Tdef * (rate_data->c_reHeII1_reHeII1[bin_id+1] - rate_data->c_reHeII1_reHeII1[bin_id]);
    cooling_rates_out[ 9] = rate_data->c_reHeII2_reHeII2[bin_id] + Tdef * (rate_data->c_reHeII2_reHeII2[bin_id+1] - rate_data->c_reHeII2_reHeII2[bin_id]);
    cooling_rates_out[10] = rate_data->c_reHeIII_reHeIII[bin_id] + Tdef * (rate_data->c_reHeIII_reHeIII[bin_id+1] - rate_data->c_reHeIII_reHeIII[bin_id]);
    cooling_rates_out[11] = rate_data->c_brem_brem[bin_id] + Tdef * (rate_data->c_brem_brem[bin_id+1] - rate_data->c_brem_brem[bin_id]);
    cooling_rates_out[12] = rate_data->c_gloverabel08_gaHI[bin_id] + Tdef * (rate_data->c_gloverabel08_gaHI[bin_id+1] - rate_data->c_gloverabel08_gaHI[bin_id]);
    cooling_rates_out[13] = rate_data->c_gloverabel08_gaH2[bin_id] + Tdef * (rate_data->c_gloverabel08_gaH2[bin_id+1] - rate_data->c_gloverabel08_gaH2[bin_id]);
    cooling_rates_out[14] = rate_data->c_gloverabel08_gaHe[bin_id] + Tdef * (rate_data->c_gloverabel08_gaHe[bin_id+1] - rate_data->c_gloverabel08_gaHe[bin_id]);
    cooling_rates_out[15] = rate_data->c_gloverabel08_gaHp[bin_id] + Tdef * (rate_data->c_gloverabel08_gaHp[bin_id+1] - rate_data->c_gloverabel08_gaHp[bin_id]);
    cooling_rates_out[16] = rate_data->c_gloverabel08_gael[bin_id] + Tdef * (rate_data->c_gloverabel08_gael[bin_id+1] - rate_data->c_gloverabel08_gael[bin_id]);
    cooling_rates_out[17] = rate_data->c_gloverabel08_h2lte[bin_id] + Tdef * (rate_data->c_gloverabel08_h2lte[bin_id+1] - rate_data->c_gloverabel08_h2lte[bin_id]);
    cooling_rates_out[18] = rate_data->c_compton_comp_[bin_id] + Tdef * (rate_data->c_compton_comp_[bin_id+1] - rate_data->c_compton_comp_[bin_id]);
    cooling_rates_out[19] = rate_data->c_gammah_gammah[bin_id] + Tdef * (rate_data->c_gammah_gammah[bin_id+1] - rate_data->c_gammah_gammah[bin_id]);
    cooling_rates_out[20] = rate_data->c_h2formation_h2mheat[bin_id] + Tdef * (rate_data->c_h2formation_h2mheat[bin_id+1] - rate_data->c_h2formation_h2mheat[bin_id]);
    cooling_rates_out[21] = rate_data->c_h2formation_h2mcool[bin_id] + Tdef * (rate_data->c_h2formation_h2mcool[bin_id+1] - rate_data->c_h2formation_h2mcool[bin_id]);
    cooling_rates_out[22] = rate_data->c_h2formation_ncrn[bin_id] + Tdef * (rate_data->c_h2formation_ncrn[bin_id+1] - rate_data->c_h2formation_ncrn[bin_id]);
    cooling_rates_out[23] = rate_data->c_h2formation_ncrd1[bin_id] + Tdef * (rate_data->c_h2formation_ncrd1[bin_id+1] - rate_data->c_h2formation_ncrd1[bin_id]);
    cooling_rates_out[24] = rate_data->c_h2formation_ncrd2[bin_id] + Tdef * (rate_data->c_h2formation_ncrd2[bin_id+1] - rate_data->c_h2formation_ncrd2[bin_id]);
    cooling_rates_out[25] = rate_data->c_cie_cooling_cieco[bin_id] + Tdef * (rate_data->c_cie_cooling_cieco[bin_id+1] - rate_data->c_cie_cooling_cieco[bin_id]);
}


__device__ void dydt (const double t, const double pres, const double * __restrict__ y, double * __restrict__ dy, const mechanism_memory * __restrict__ d_mem) {


  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int NSPECIES = 10;
  int NRATE    = 23;
  int NCOOL    = 26;

  double * __restrict__ local_reaction_rates = &d_mem->reaction_rates[tid*NRATE];
  double * __restrict__ local_cooling_rates  = &d_mem->cooling_rates [tid*NCOOL];
  cvklu_data *rate_data = d_mem->chemistry_data;

  double T_local = 1000.0;
  double mdensity = 1.0e14*1.67e-24; //d_mem->temperature[tid]; 
  const double density_local = d_mem->density[tid]; 
  double h2_optical_depth_approx = fmin( 1.0, pow((density_local / (1.34e-14)), -0.45) ); 
 
  evaluate_temperature ( &T_local, y, rate_data );
  
  //if ( tid == 0 ){
  //printf("T_local: %0.5g\n", T_local);
  //}
  // double * __restrict__ fwd_rates = d_mem->fwd_rates;
  // double * __restrict__ rev_rates = d_mem->rev_rates;
  
  
  interpolate_reaction_rates( local_reaction_rates, T_local, rate_data);
  interpolate_cooling_rates ( local_cooling_rates , T_local, rate_data);

  //# 0: H2_1
  dy[INDEX(0)] = local_reaction_rates[7]*y[INDEX(2)]*y[INDEX(4)] + local_reaction_rates[9]*y[INDEX(1)]*y[INDEX(2)] - local_reaction_rates[10]*y[INDEX(0)]*y[INDEX(3)] - local_reaction_rates[11]*y[INDEX(0)]*y[INDEX(8)] - local_reaction_rates[12]*y[INDEX(0)]*y[INDEX(2)] + local_reaction_rates[18]*y[INDEX(1)]*y[INDEX(4)] + local_reaction_rates[20]*y[INDEX(0)]*y[INDEX(2)]*y[INDEX(2)] + local_reaction_rates[21]*y[INDEX(2)]*y[INDEX(2)]*y[INDEX(2)];
  //# 1: H2_2
  dy[INDEX(1)] = local_reaction_rates[8]*y[INDEX(2)]*y[INDEX(3)] - local_reaction_rates[9]*y[INDEX(1)]*y[INDEX(2)] + local_reaction_rates[10]*y[INDEX(0)]*y[INDEX(3)] + local_reaction_rates[16]*y[INDEX(3)]*y[INDEX(4)] - local_reaction_rates[17]*y[INDEX(1)]*y[INDEX(8)] - local_reaction_rates[18]*y[INDEX(1)]*y[INDEX(4)];
  //# 2: H_1
  dy[INDEX(2)] = -local_reaction_rates[0]*y[INDEX(2)]*y[INDEX(8)] + local_reaction_rates[1]*y[INDEX(3)]*y[INDEX(8)] - local_reaction_rates[6]*y[INDEX(2)]*y[INDEX(8)] - local_reaction_rates[7]*y[INDEX(2)]*y[INDEX(4)] - local_reaction_rates[8]*y[INDEX(2)]*y[INDEX(3)] - local_reaction_rates[9]*y[INDEX(1)]*y[INDEX(2)] + local_reaction_rates[10]*y[INDEX(0)]*y[INDEX(3)] + 2*local_reaction_rates[11]*y[INDEX(0)]*y[INDEX(8)] + 2*local_reaction_rates[12]*y[INDEX(0)]*y[INDEX(2)] + local_reaction_rates[13]*y[INDEX(4)]*y[INDEX(8)] + local_reaction_rates[14]*y[INDEX(2)]*y[INDEX(4)] + 2*local_reaction_rates[15]*y[INDEX(3)]*y[INDEX(4)] + 2*local_reaction_rates[17]*y[INDEX(1)]*y[INDEX(8)] + local_reaction_rates[18]*y[INDEX(1)]*y[INDEX(4)] - 2*local_reaction_rates[20]*y[INDEX(0)]*y[INDEX(2)]*y[INDEX(2)] - 2*local_reaction_rates[21]*y[INDEX(2)]*y[INDEX(2)]*y[INDEX(2)];
  //# 3: H_2
  dy[INDEX(3)] = local_reaction_rates[0]*y[INDEX(2)]*y[INDEX(8)] - local_reaction_rates[1]*y[INDEX(3)]*y[INDEX(8)] - local_reaction_rates[8]*y[INDEX(2)]*y[INDEX(3)] + local_reaction_rates[9]*y[INDEX(1)]*y[INDEX(2)] - local_reaction_rates[10]*y[INDEX(0)]*y[INDEX(3)] - local_reaction_rates[15]*y[INDEX(3)]*y[INDEX(4)] - local_reaction_rates[16]*y[INDEX(3)]*y[INDEX(4)];
  //# 4: H_m0
  dy[INDEX(4)] = local_reaction_rates[6]*y[INDEX(2)]*y[INDEX(8)] - local_reaction_rates[7]*y[INDEX(2)]*y[INDEX(4)] - local_reaction_rates[13]*y[INDEX(4)]*y[INDEX(8)] - local_reaction_rates[14]*y[INDEX(2)]*y[INDEX(4)] - local_reaction_rates[15]*y[INDEX(3)]*y[INDEX(4)] - local_reaction_rates[16]*y[INDEX(3)]*y[INDEX(4)] - local_reaction_rates[18]*y[INDEX(1)]*y[INDEX(4)];
  //# 5: He_1
  dy[INDEX(5)] = -local_reaction_rates[2]*y[INDEX(5)]*y[INDEX(8)] + local_reaction_rates[3]*y[INDEX(6)]*y[INDEX(8)];
  //# 6: He_2
  dy[INDEX(6)] = local_reaction_rates[2]*y[INDEX(5)]*y[INDEX(8)] - local_reaction_rates[3]*y[INDEX(6)]*y[INDEX(8)] - local_reaction_rates[4]*y[INDEX(6)]*y[INDEX(8)] + local_reaction_rates[5]*y[INDEX(7)]*y[INDEX(8)];
  //# 7: He_3
  dy[INDEX(7)] = local_reaction_rates[4]*y[INDEX(6)]*y[INDEX(8)] - local_reaction_rates[5]*y[INDEX(7)]*y[INDEX(8)];
  //# 8: de
  dy[INDEX(8)] = local_reaction_rates[0]*y[INDEX(2)]*y[INDEX(8)] - local_reaction_rates[1]*y[INDEX(3)]*y[INDEX(8)] + local_reaction_rates[2]*y[INDEX(5)]*y[INDEX(8)] - local_reaction_rates[3]*y[INDEX(6)]*y[INDEX(8)] + local_reaction_rates[4]*y[INDEX(6)]*y[INDEX(8)] - local_reaction_rates[5]*y[INDEX(7)]*y[INDEX(8)] - local_reaction_rates[6]*y[INDEX(2)]*y[INDEX(8)] + local_reaction_rates[7]*y[INDEX(2)]*y[INDEX(4)] + local_reaction_rates[13]*y[INDEX(4)]*y[INDEX(8)] + local_reaction_rates[14]*y[INDEX(2)]*y[INDEX(4)] + local_reaction_rates[16]*y[INDEX(3)]*y[INDEX(4)] - local_reaction_rates[17]*y[INDEX(1)]*y[INDEX(8)];
  //# 9: ge
  dy[INDEX(9)] = -2.01588*y[INDEX(0)]*local_cooling_rates[25]*local_cooling_rates[26]*density_local - y[INDEX(0)]*local_cooling_rates[26]*local_cooling_rates[17]*h2_optical_depth_approx/(local_cooling_rates[17]/(y[INDEX(0)]*local_cooling_rates[13] + y[INDEX(2)]*local_cooling_rates[12] + y[INDEX(3)]*local_cooling_rates[15] + y[INDEX(5)]*local_cooling_rates[14] + y[INDEX(8)]*local_cooling_rates[16]) + 1.0) - y[INDEX(2)]*local_cooling_rates[0]*local_cooling_rates[26]*y[INDEX(8)] - y[INDEX(2)]*local_cooling_rates[4]*local_cooling_rates[26]*y[INDEX(8)] - y[INDEX(3)]*local_cooling_rates[26]*y[INDEX(8)]*local_cooling_rates[7] - y[INDEX(5)]*local_cooling_rates[5]*local_cooling_rates[26]*y[INDEX(8)] - y[INDEX(6)]*local_cooling_rates[2]*local_cooling_rates[26]*y[INDEX(8)] - y[INDEX(6)]*local_cooling_rates[1]*local_cooling_rates[26]*pow(y[INDEX(8)], 2) - y[INDEX(6)]*local_cooling_rates[6]*local_cooling_rates[26]*y[INDEX(8)] - y[INDEX(6)]*local_cooling_rates[3]*local_cooling_rates[26]*pow(y[INDEX(8)], 2) - y[INDEX(6)]*local_cooling_rates[26]*y[INDEX(8)]*local_cooling_rates[8] - y[INDEX(6)]*local_cooling_rates[26]*y[INDEX(8)]*local_cooling_rates[9] - y[INDEX(7)]*local_cooling_rates[26]*y[INDEX(8)]*local_cooling_rates[10] - local_cooling_rates[11]*local_cooling_rates[26]*y[INDEX(8)]*(y[INDEX(3)] + y[INDEX(6)] + 4.0*y[INDEX(7)]) - local_cooling_rates[26]*local_cooling_rates[18]*y[INDEX(8)]*pow(1.0, 4)*( T_local - 2.73) + 0.5*1.0/(local_cooling_rates[22]/(y[INDEX(0)]*local_cooling_rates[24] + y[INDEX(2)]*local_cooling_rates[23]) + 1.0)*(-y[INDEX(0)]*y[INDEX(2)]*local_cooling_rates[21] + pow(y[INDEX(2)], 3)*local_cooling_rates[20]);
 dy[INDEX(9)] /= mdensity;
/*
  for (int i = 0; i< 10; i++){
    printf(" y[INDEX(%d)] = %0.5g\n", i, y[INDEX(i)]);
    printf("dy[INDEX(%d)] = %0.5g\n", i,dy[INDEX(i)]);
  }
*/
}

