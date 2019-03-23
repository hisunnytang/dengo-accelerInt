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



__device__ void evaluate_temperature( double* T, double* dTs_ge, const double *y, const double mdensity, cvklu_data *rate_data )
{
   // iterate temperature to convergence
  double t, tnew, tdiff;
  double dge, dge_dT;
  double gammaH2, dgammaH2_dT, _gammaH2_m1;

  int count = 0;
  int MAX_ITERATION = 100; 
  double gamma     = 5./3.;
  double _gamma_m1 = 1.0 / (gamma - 1.0);
  double kb = 1.3806504e-16; // Boltzamann constant [erg/K] 
  // prepare t, tnew for the newton's iteration;

  t     = *T;
  if (t != t) t = 1000.0;
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
      printf("T[tid = %d] failed to converge (iteration: %d); at T = %0.3g \n", T_ID, count, tnew );
    }
    if ( t!= t && T_ID == 0){
      printf("T[tid = %d] is %0.5g, count = %d; ge = %0.5g, gamma_H2 = %0.5g \n", T_ID, t, count, y[INDEX(9)], gammaH2);
      t = 1000.0;
      for (int i = 0; i < 10; i++){
          printf("y[INDEX(%d)] = %0.5g \n", i, y[INDEX(i)]);
      }
      break;
    }

  }
  // update the temperature;
  *T = t;
  *dTs_ge = 1.0 / dge_dT;

  // printf("T[tid = %d] is %0.5g, count = %d; ge = %0.5g, gamma_H2 = %0.5g \n", tid, t, count, y[INDEX(9)], gammaH2);
  
}



__device__ void interpolate_reaction_rates( double *reaction_rates_out, double temp_out, cvklu_data *rate_data)
{
    
    int tid, bin_id, zbin_id;
    double t1, t2;
    double Tdef, dT, invTs, log_temp_out;
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
    dT    = t2 - t1;
    invTs = 1.0 / temp_out;
 
    // rate_out is a long 1D array
    // NRATE is the number of rate required by the solver network

    reaction_rates_out[INDEX( 0)] = rate_data->r_k01[bin_id] + Tdef * (rate_data->r_k01[bin_id+1] - rate_data->r_k01[bin_id]);
    reaction_rates_out[INDEX( 1)] = rate_data->r_k02[bin_id] + Tdef * (rate_data->r_k02[bin_id+1] - rate_data->r_k02[bin_id]);
    reaction_rates_out[INDEX( 2)] = rate_data->r_k03[bin_id] + Tdef * (rate_data->r_k03[bin_id+1] - rate_data->r_k03[bin_id]);
    reaction_rates_out[INDEX( 3)] = rate_data->r_k04[bin_id] + Tdef * (rate_data->r_k04[bin_id+1] - rate_data->r_k04[bin_id]);
    reaction_rates_out[INDEX( 4)] = rate_data->r_k05[bin_id] + Tdef * (rate_data->r_k05[bin_id+1] - rate_data->r_k05[bin_id]);
    reaction_rates_out[INDEX( 5)] = rate_data->r_k06[bin_id] + Tdef * (rate_data->r_k06[bin_id+1] - rate_data->r_k06[bin_id]);
    reaction_rates_out[INDEX( 6)] = rate_data->r_k07[bin_id] + Tdef * (rate_data->r_k07[bin_id+1] - rate_data->r_k07[bin_id]);
    reaction_rates_out[INDEX( 7)] = rate_data->r_k08[bin_id] + Tdef * (rate_data->r_k08[bin_id+1] - rate_data->r_k08[bin_id]);
    reaction_rates_out[INDEX( 8)] = rate_data->r_k09[bin_id] + Tdef * (rate_data->r_k09[bin_id+1] - rate_data->r_k09[bin_id]);
    reaction_rates_out[INDEX( 9)] = rate_data->r_k10[bin_id] + Tdef * (rate_data->r_k10[bin_id+1] - rate_data->r_k10[bin_id]);
    reaction_rates_out[INDEX(10)] = rate_data->r_k11[bin_id] + Tdef * (rate_data->r_k11[bin_id+1] - rate_data->r_k11[bin_id]);
    reaction_rates_out[INDEX(11)] = rate_data->r_k12[bin_id] + Tdef * (rate_data->r_k12[bin_id+1] - rate_data->r_k12[bin_id]);
    reaction_rates_out[INDEX(12)] = rate_data->r_k13[bin_id] + Tdef * (rate_data->r_k13[bin_id+1] - rate_data->r_k13[bin_id]);
    reaction_rates_out[INDEX(13)] = rate_data->r_k14[bin_id] + Tdef * (rate_data->r_k14[bin_id+1] - rate_data->r_k14[bin_id]);
    reaction_rates_out[INDEX(14)] = rate_data->r_k15[bin_id] + Tdef * (rate_data->r_k15[bin_id+1] - rate_data->r_k15[bin_id]);
    reaction_rates_out[INDEX(15)] = rate_data->r_k16[bin_id] + Tdef * (rate_data->r_k16[bin_id+1] - rate_data->r_k16[bin_id]);
    reaction_rates_out[INDEX(16)] = rate_data->r_k17[bin_id] + Tdef * (rate_data->r_k17[bin_id+1] - rate_data->r_k17[bin_id]);
    reaction_rates_out[INDEX(17)] = rate_data->r_k18[bin_id] + Tdef * (rate_data->r_k18[bin_id+1] - rate_data->r_k18[bin_id]);
    reaction_rates_out[INDEX(18)] = rate_data->r_k19[bin_id] + Tdef * (rate_data->r_k19[bin_id+1] - rate_data->r_k19[bin_id]);
    //reaction_rates_out[INDEX(19)] = rate_data->r_k20[bin_id] + Tdef * (rate_data->r_k20[bin_id+1] - rate_data->r_k20[bin_id]);
    reaction_rates_out[INDEX(20)] = rate_data->r_k21[bin_id] + Tdef * (rate_data->r_k21[bin_id+1] - rate_data->r_k21[bin_id]);
    reaction_rates_out[INDEX(21)] = rate_data->r_k22[bin_id] + Tdef * (rate_data->r_k22[bin_id+1] - rate_data->r_k22[bin_id]);
    //reaction_rates_out[INDEX(22)] = rate_data->r_k23[bin_id] + Tdef * (rate_data->r_k23[bin_id+1] - rate_data->r_k23[bin_id]);

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

/*
    if (T_ID == 0){
    printf( "bin_id = %d; temp_out = %0.5g \n", bin_id, temp_out);
    }
*/

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

    cooling_rates_out[INDEX( 0)] = rate_data->c_ceHI_ceHI[bin_id] + Tdef * (rate_data->c_ceHI_ceHI[bin_id+1] - rate_data->c_ceHI_ceHI[bin_id]);
    cooling_rates_out[INDEX( 1)] = rate_data->c_ceHeI_ceHeI[bin_id] + Tdef * (rate_data->c_ceHeI_ceHeI[bin_id+1] - rate_data->c_ceHeI_ceHeI[bin_id]);
    cooling_rates_out[INDEX( 2)] = rate_data->c_ceHeII_ceHeII[bin_id] + Tdef * (rate_data->c_ceHeII_ceHeII[bin_id+1] - rate_data->c_ceHeII_ceHeII[bin_id]);
    cooling_rates_out[INDEX( 3)] = rate_data->c_ciHeIS_ciHeIS[bin_id] + Tdef * (rate_data->c_ciHeIS_ciHeIS[bin_id+1] - rate_data->c_ciHeIS_ciHeIS[bin_id]);
    cooling_rates_out[INDEX( 4)] = rate_data->c_ciHI_ciHI[bin_id] + Tdef * (rate_data->c_ciHI_ciHI[bin_id+1] - rate_data->c_ciHI_ciHI[bin_id]);
    cooling_rates_out[INDEX( 5)] = rate_data->c_ciHeI_ciHeI[bin_id] + Tdef * (rate_data->c_ciHeI_ciHeI[bin_id+1] - rate_data->c_ciHeI_ciHeI[bin_id]);
    cooling_rates_out[INDEX( 6)] = rate_data->c_ciHeII_ciHeII[bin_id] + Tdef * (rate_data->c_ciHeII_ciHeII[bin_id+1] - rate_data->c_ciHeII_ciHeII[bin_id]);
    cooling_rates_out[INDEX( 7)] = rate_data->c_reHII_reHII[bin_id] + Tdef * (rate_data->c_reHII_reHII[bin_id+1] - rate_data->c_reHII_reHII[bin_id]);
    cooling_rates_out[INDEX( 8)] = rate_data->c_reHeII1_reHeII1[bin_id] + Tdef * (rate_data->c_reHeII1_reHeII1[bin_id+1] - rate_data->c_reHeII1_reHeII1[bin_id]);
    cooling_rates_out[INDEX( 9)] = rate_data->c_reHeII2_reHeII2[bin_id] + Tdef * (rate_data->c_reHeII2_reHeII2[bin_id+1] - rate_data->c_reHeII2_reHeII2[bin_id]);
    cooling_rates_out[INDEX(10)] = rate_data->c_reHeIII_reHeIII[bin_id] + Tdef * (rate_data->c_reHeIII_reHeIII[bin_id+1] - rate_data->c_reHeIII_reHeIII[bin_id]);
    cooling_rates_out[INDEX(11)] = rate_data->c_brem_brem[bin_id] + Tdef * (rate_data->c_brem_brem[bin_id+1] - rate_data->c_brem_brem[bin_id]);
    cooling_rates_out[INDEX(12)] = rate_data->c_gloverabel08_gaHI[bin_id] + Tdef * (rate_data->c_gloverabel08_gaHI[bin_id+1] - rate_data->c_gloverabel08_gaHI[bin_id]);
    cooling_rates_out[INDEX(13)] = rate_data->c_gloverabel08_gaH2[bin_id] + Tdef * (rate_data->c_gloverabel08_gaH2[bin_id+1] - rate_data->c_gloverabel08_gaH2[bin_id]);
    cooling_rates_out[INDEX(14)] = rate_data->c_gloverabel08_gaHe[bin_id] + Tdef * (rate_data->c_gloverabel08_gaHe[bin_id+1] - rate_data->c_gloverabel08_gaHe[bin_id]);
    cooling_rates_out[INDEX(15)] = rate_data->c_gloverabel08_gaHp[bin_id] + Tdef * (rate_data->c_gloverabel08_gaHp[bin_id+1] - rate_data->c_gloverabel08_gaHp[bin_id]);
    cooling_rates_out[INDEX(16)] = rate_data->c_gloverabel08_gael[bin_id] + Tdef * (rate_data->c_gloverabel08_gael[bin_id+1] - rate_data->c_gloverabel08_gael[bin_id]);
    cooling_rates_out[INDEX(17)] = rate_data->c_gloverabel08_h2lte[bin_id] + Tdef * (rate_data->c_gloverabel08_h2lte[bin_id+1] - rate_data->c_gloverabel08_h2lte[bin_id]);
    cooling_rates_out[INDEX(18)] = rate_data->c_compton_comp_[bin_id] + Tdef * (rate_data->c_compton_comp_[bin_id+1] - rate_data->c_compton_comp_[bin_id]);
    cooling_rates_out[INDEX(19)] = rate_data->c_gammah_gammah[bin_id] + Tdef * (rate_data->c_gammah_gammah[bin_id+1] - rate_data->c_gammah_gammah[bin_id]);
    cooling_rates_out[INDEX(20)] = rate_data->c_h2formation_h2mheat[bin_id] + Tdef * (rate_data->c_h2formation_h2mheat[bin_id+1] - rate_data->c_h2formation_h2mheat[bin_id]);
    cooling_rates_out[INDEX(21)] = rate_data->c_h2formation_h2mcool[bin_id] + Tdef * (rate_data->c_h2formation_h2mcool[bin_id+1] - rate_data->c_h2formation_h2mcool[bin_id]);
    cooling_rates_out[INDEX(22)] = rate_data->c_h2formation_ncrn[bin_id] + Tdef * (rate_data->c_h2formation_ncrn[bin_id+1] - rate_data->c_h2formation_ncrn[bin_id]);
    cooling_rates_out[INDEX(23)] = rate_data->c_h2formation_ncrd1[bin_id] + Tdef * (rate_data->c_h2formation_ncrd1[bin_id+1] - rate_data->c_h2formation_ncrd1[bin_id]);
    cooling_rates_out[INDEX(24)] = rate_data->c_h2formation_ncrd2[bin_id] + Tdef * (rate_data->c_h2formation_ncrd2[bin_id+1] - rate_data->c_h2formation_ncrd2[bin_id]);
    cooling_rates_out[INDEX(25)] = rate_data->c_cie_cooling_cieco[bin_id] + Tdef * (rate_data->c_cie_cooling_cieco[bin_id+1] - rate_data->c_cie_cooling_cieco[bin_id]);
    cooling_rates_out[INDEX(26)] = 1.0; //rate_data->c_cie_cooling_cieco[bin_id] + Tdef * (rate_data->c_cie_cooling_cieco[bin_id+1] - rate_data->c_cie_cooling_cieco[bin_id]);

}

__device__ void interpolate_dcrate_dT(double *dcr_dT, const double temp_out, cvklu_data *rate_data ){
    int tid, bin_id, zbin_id;
    double t1, t2;
    double Tdef, dT, inv_Ts, log_temp_out;
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
    dT    = t2 - t1;
    inv_Ts = temp_out;
 
    //ceHI_ceHI: 0
    dcr_dT[INDEX( 0)] = rate_data->c_ceHI_ceHI[bin_id+1] - rate_data->c_ceHI_ceHI[bin_id];
    dcr_dT[INDEX( 0)] /= dT ;
    dcr_dT[INDEX( 0)] /= inv_Ts ;
    
    //ceHeI_ceHeI: 1
    dcr_dT[INDEX( 1)] = rate_data->c_ceHeI_ceHeI[bin_id+1] - rate_data->c_ceHeI_ceHeI[bin_id];
    dcr_dT[INDEX( 1)] /= dT ;
    dcr_dT[INDEX( 1)] /= inv_Ts ;
    
    //ceHeII_ceHeII: 2
    dcr_dT[INDEX( 2)] = rate_data->c_ceHeII_ceHeII[bin_id+1] - rate_data->c_ceHeII_ceHeII[bin_id];
    dcr_dT[INDEX( 2)] /= dT ;
    dcr_dT[INDEX( 2)] /= inv_Ts ;
    
    //ciHeIS_ciHeIS: 3
    dcr_dT[INDEX( 3)] = rate_data->c_ciHeIS_ciHeIS[bin_id+1] - rate_data->c_ciHeIS_ciHeIS[bin_id];
    dcr_dT[INDEX( 3)] /= dT ;
    dcr_dT[INDEX( 3)] /= inv_Ts ;
    
    //ciHI_ciHI: 4
    dcr_dT[INDEX( 4)] = rate_data->c_ciHI_ciHI[bin_id+1] - rate_data->c_ciHI_ciHI[bin_id];
    dcr_dT[INDEX( 4)] /= dT ;
    dcr_dT[INDEX( 4)] /= inv_Ts ;
    
    //ciHeI_ciHeI: 5
    dcr_dT[INDEX( 5)] = rate_data->c_ciHeI_ciHeI[bin_id+1] - rate_data->c_ciHeI_ciHeI[bin_id];
    dcr_dT[INDEX( 5)] /= dT ;
    dcr_dT[INDEX( 5)] /= inv_Ts ;
    
    //ciHeII_ciHeII: 6
    dcr_dT[INDEX( 6)] = rate_data->c_ciHeII_ciHeII[bin_id+1] - rate_data->c_ciHeII_ciHeII[bin_id];
    dcr_dT[INDEX( 6)] /= dT ;
    dcr_dT[INDEX( 6)] /= inv_Ts ;
    
    //reHII_reHII: 7
    dcr_dT[INDEX( 7)] = rate_data->c_reHII_reHII[bin_id+1] - rate_data->c_reHII_reHII[bin_id];
    dcr_dT[INDEX( 7)] /= dT ;
    dcr_dT[INDEX( 7)] /= inv_Ts ;
    
    //reHeII1_reHeII1: 8
    dcr_dT[INDEX( 8)] = rate_data->c_reHeII1_reHeII1[bin_id+1] - rate_data->c_reHeII1_reHeII1[bin_id];
    dcr_dT[INDEX( 8)] /= dT ;
    dcr_dT[INDEX( 8)] /= inv_Ts ;
    
    //reHeII2_reHeII2: 9
    dcr_dT[INDEX( 9)] = rate_data->c_reHeII2_reHeII2[bin_id+1] - rate_data->c_reHeII2_reHeII2[bin_id];
    dcr_dT[INDEX( 9)] /= dT ;
    dcr_dT[INDEX( 9)] /= inv_Ts ;
    
    //reHeIII_reHeIII: 10
    dcr_dT[INDEX(10)] = rate_data->c_reHeIII_reHeIII[bin_id+1] - rate_data->c_reHeIII_reHeIII[bin_id];
    dcr_dT[INDEX(10)] /= dT ;
    dcr_dT[INDEX(10)] /= inv_Ts ;
    
    //brem_brem: 11
    dcr_dT[INDEX(11)] = rate_data->c_brem_brem[bin_id+1] - rate_data->c_brem_brem[bin_id];
    dcr_dT[INDEX(11)] /= dT ;
    dcr_dT[INDEX(11)] /= inv_Ts ;
    
    //gloverabel08_gaHI: 12
    dcr_dT[INDEX(12)] = rate_data->c_gloverabel08_gaHI[bin_id+1] - rate_data->c_gloverabel08_gaHI[bin_id];
    dcr_dT[INDEX(12)] /= dT ;
    dcr_dT[INDEX(12)] /= inv_Ts ;
    
    //gloverabel08_gaH2: 13
    dcr_dT[INDEX(13)] = rate_data->c_gloverabel08_gaH2[bin_id+1] - rate_data->c_gloverabel08_gaH2[bin_id];
    dcr_dT[INDEX(13)] /= dT ;
    dcr_dT[INDEX(13)] /= inv_Ts ;
    
    //gloverabel08_gaHe: 14
    dcr_dT[INDEX(14)] = rate_data->c_gloverabel08_gaHe[bin_id+1] - rate_data->c_gloverabel08_gaHe[bin_id];
    dcr_dT[INDEX(14)] /= dT ;
    dcr_dT[INDEX(14)] /= inv_Ts ;
    
    //gloverabel08_gaHp: 15
    dcr_dT[INDEX(15)] = rate_data->c_gloverabel08_gaHp[bin_id+1] - rate_data->c_gloverabel08_gaHp[bin_id];
    dcr_dT[INDEX(15)] /= dT ;
    dcr_dT[INDEX(15)] /= inv_Ts ;
    
    //gloverabel08_gael: 16
    dcr_dT[INDEX(16)] = rate_data->c_gloverabel08_gael[bin_id+1] - rate_data->c_gloverabel08_gael[bin_id];
    dcr_dT[INDEX(16)] /= dT ;
    dcr_dT[INDEX(16)] /= inv_Ts ;
    
    //gloverabel08_h2lte: 17
    dcr_dT[INDEX(17)] = rate_data->c_gloverabel08_h2lte[bin_id+1] - rate_data->c_gloverabel08_h2lte[bin_id];
    dcr_dT[INDEX(17)] /= dT ;
    dcr_dT[INDEX(17)] /= inv_Ts ;
    
    //compton_comp_: 18
    dcr_dT[INDEX(18)] = rate_data->c_compton_comp_[bin_id+1] - rate_data->c_compton_comp_[bin_id];
    dcr_dT[INDEX(18)] /= dT ;
    dcr_dT[INDEX(18)] /= inv_Ts ;
    
    //gammah_gammah: 19
    dcr_dT[INDEX(19)] = rate_data->c_gammah_gammah[bin_id+1] - rate_data->c_gammah_gammah[bin_id];
    dcr_dT[INDEX(19)] /= dT ;
    dcr_dT[INDEX(19)] /= inv_Ts ;
    
    //h2formation_h2mheat: 20
    dcr_dT[INDEX(20)] = rate_data->c_h2formation_h2mheat[bin_id+1] - rate_data->c_h2formation_h2mheat[bin_id];
    dcr_dT[INDEX(20)] /= dT ;
    dcr_dT[INDEX(20)] /= inv_Ts ;
    
    //h2formation_h2mcool: 21
    dcr_dT[INDEX(21)] = rate_data->c_h2formation_h2mcool[bin_id+1] - rate_data->c_h2formation_h2mcool[bin_id];
    dcr_dT[INDEX(21)] /= dT ;
    dcr_dT[INDEX(21)] /= inv_Ts ;
    
    //h2formation_ncrn: 22
    dcr_dT[INDEX(22)] = rate_data->c_h2formation_ncrn[bin_id+1] - rate_data->c_h2formation_ncrn[bin_id];
    dcr_dT[INDEX(22)] /= dT ;
    dcr_dT[INDEX(22)] /= inv_Ts ;
    
    //h2formation_ncrd1: 23
    dcr_dT[INDEX(23)] = rate_data->c_h2formation_ncrd1[bin_id+1] - rate_data->c_h2formation_ncrd1[bin_id];
    dcr_dT[INDEX(23)] /= dT ;
    dcr_dT[INDEX(23)] /= inv_Ts ;
    
    //h2formation_ncrd2: 24
    dcr_dT[INDEX(24)] = rate_data->c_h2formation_ncrd2[bin_id+1] - rate_data->c_h2formation_ncrd2[bin_id];
    dcr_dT[INDEX(24)] /= dT ;
    dcr_dT[INDEX(24)] /= inv_Ts ;
    
    //cie_cooling_cieco: 25
    dcr_dT[INDEX(25)] = rate_data->c_cie_cooling_cieco[bin_id+1] - rate_data->c_cie_cooling_cieco[bin_id];
    dcr_dT[INDEX(25)] /= dT ;
    dcr_dT[INDEX(25)] /= inv_Ts ;
    
    //cie_optical_depth_approx: 26
    //dcr_dT[INDEX(26)] = rate_data->c_cie_optical_depth_approx[bin_id+1] - rate_data->c_cie_optical_depth_approx[bin_id];
    //dcr_dT[INDEX(26)] /= dT ;
    //dcr_dT[INDEX(26)] /= inv_Ts ;
    dcr_dT[INDEX(26)] = 0.0;
}

__device__ void interpolate_drrate_dT(double *drr_dT, const double temp_out, cvklu_data *rate_data ){
    int tid, bin_id, zbin_id;
    double t1, t2;
    double Tdef, dT, inv_Ts, log_temp_out;
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
    dT    = t2 - t1;
    inv_Ts = temp_out;
 
    //k01: 0
    drr_dT[INDEX( 0)] = rate_data->r_k01[bin_id+1] - rate_data->r_k01[bin_id];
    drr_dT[INDEX( 0)] /= dT ;
    drr_dT[INDEX( 0)] /= inv_Ts ;
    
    //k02: 1
    drr_dT[INDEX( 1)] = rate_data->r_k02[bin_id+1] - rate_data->r_k02[bin_id];
    drr_dT[INDEX( 1)] /= dT ;
    drr_dT[INDEX( 1)] /= inv_Ts ;
    
    //k03: 2
    drr_dT[INDEX( 2)] = rate_data->r_k03[bin_id+1] - rate_data->r_k03[bin_id];
    drr_dT[INDEX( 2)] /= dT ;
    drr_dT[INDEX( 2)] /= inv_Ts ;
    
    //k04: 3
    drr_dT[INDEX( 3)] = rate_data->r_k04[bin_id+1] - rate_data->r_k04[bin_id];
    drr_dT[INDEX( 3)] /= dT ;
    drr_dT[INDEX( 3)] /= inv_Ts ;
    
    //k05: 4
    drr_dT[INDEX( 4)] = rate_data->r_k05[bin_id+1] - rate_data->r_k05[bin_id];
    drr_dT[INDEX( 4)] /= dT ;
    drr_dT[INDEX( 4)] /= inv_Ts ;
    
    //k06: 5
    drr_dT[INDEX( 5)] = rate_data->r_k06[bin_id+1] - rate_data->r_k06[bin_id];
    drr_dT[INDEX( 5)] /= dT ;
    drr_dT[INDEX( 5)] /= inv_Ts ;
    
    //k07: 6
    drr_dT[INDEX( 6)] = rate_data->r_k07[bin_id+1] - rate_data->r_k07[bin_id];
    drr_dT[INDEX( 6)] /= dT ;
    drr_dT[INDEX( 6)] /= inv_Ts ;
    
    //k08: 7
    drr_dT[INDEX( 7)] = rate_data->r_k08[bin_id+1] - rate_data->r_k08[bin_id];
    drr_dT[INDEX( 7)] /= dT ;
    drr_dT[INDEX( 7)] /= inv_Ts ;
    
    //k09: 8
    drr_dT[INDEX( 8)] = rate_data->r_k09[bin_id+1] - rate_data->r_k09[bin_id];
    drr_dT[INDEX( 8)] /= dT ;
    drr_dT[INDEX( 8)] /= inv_Ts ;
    
    //k10: 9
    drr_dT[INDEX( 9)] = rate_data->r_k10[bin_id+1] - rate_data->r_k10[bin_id];
    drr_dT[INDEX( 9)] /= dT ;
    drr_dT[INDEX( 9)] /= inv_Ts ;
    
    //k11: 10
    drr_dT[INDEX(10)] = rate_data->r_k11[bin_id+1] - rate_data->r_k11[bin_id];
    drr_dT[INDEX(10)] /= dT ;
    drr_dT[INDEX(10)] /= inv_Ts ;
    
    //k12: 11
    drr_dT[INDEX(11)] = rate_data->r_k12[bin_id+1] - rate_data->r_k12[bin_id];
    drr_dT[INDEX(11)] /= dT ;
    drr_dT[INDEX(11)] /= inv_Ts ;
    
    //k13: 12
    drr_dT[INDEX(12)] = rate_data->r_k13[bin_id+1] - rate_data->r_k13[bin_id];
    drr_dT[INDEX(12)] /= dT ;
    drr_dT[INDEX(12)] /= inv_Ts ;
    
    //k14: 13
    drr_dT[INDEX(13)] = rate_data->r_k14[bin_id+1] - rate_data->r_k14[bin_id];
    drr_dT[INDEX(13)] /= dT ;
    drr_dT[INDEX(13)] /= inv_Ts ;
    
    //k15: 14
    drr_dT[INDEX(14)] = rate_data->r_k15[bin_id+1] - rate_data->r_k15[bin_id];
    drr_dT[INDEX(14)] /= dT ;
    drr_dT[INDEX(14)] /= inv_Ts ;
    
    //k16: 15
    drr_dT[INDEX(15)] = rate_data->r_k16[bin_id+1] - rate_data->r_k16[bin_id];
    drr_dT[INDEX(15)] /= dT ;
    drr_dT[INDEX(15)] /= inv_Ts ;
    
    //k17: 16
    drr_dT[INDEX(16)] = rate_data->r_k17[bin_id+1] - rate_data->r_k17[bin_id];
    drr_dT[INDEX(16)] /= dT ;
    drr_dT[INDEX(16)] /= inv_Ts ;
    
    //k18: 17
    drr_dT[INDEX(17)] = rate_data->r_k18[bin_id+1] - rate_data->r_k18[bin_id];
    drr_dT[INDEX(17)] /= dT ;
    drr_dT[INDEX(17)] /= inv_Ts ;
    
    //k19: 18
    drr_dT[INDEX(18)] = rate_data->r_k19[bin_id+1] - rate_data->r_k19[bin_id];
    drr_dT[INDEX(18)] /= dT ;
    drr_dT[INDEX(18)] /= inv_Ts ;
    
    //k20: 19
    //drr_dT[INDEX(19)] = rate_data->r_k20[bin_id+1] - rate_data->r_k20[bin_id];
    //drr_dT[INDEX(19)] /= dT ;
    //drr_dT[INDEX(19)] /= inv_Ts ;
    
    //k21: 20
    drr_dT[INDEX(20)] = rate_data->r_k21[bin_id+1] - rate_data->r_k21[bin_id];
    drr_dT[INDEX(20)] /= dT ;
    drr_dT[INDEX(20)] /= inv_Ts ;
    
    //k22: 21
    drr_dT[INDEX(21)] = rate_data->r_k22[bin_id+1] - rate_data->r_k22[bin_id];
    drr_dT[INDEX(21)] /= dT ;
    drr_dT[INDEX(21)] /= inv_Ts ;
    
    //k23: 22
    //drr_dT[INDEX(22)] = rate_data->r_k23[bin_id+1] - rate_data->r_k23[bin_id];
    //drr_dT[INDEX(22)] /= dT ;
    //drr_dT[INDEX(22)] /= inv_Ts ;
}


__device__ void dydt (const double t, const double pres, const double * __restrict__ y_in, double * __restrict__ dy, const mechanism_memory * d_mem) {


  int tid = threadIdx.x + blockDim.x * blockIdx.x;
//  int NSPECIES = 10;
  int NRATE    = 23;
  int NCOOL    = 26;

  double * local_reaction_rates = d_mem->reaction_rates;
  double * local_cooling_rates  = d_mem->cooling_rates ;

  // scale related piece
  double * y = d_mem->temp_array; // working space for scaling the variable back;
  cvklu_data *rate_data = d_mem->chemistry_data;

  // these should be retreieved from d_mem object
  double T_local  = d_mem->temperature[T_ID];
  double Tge      = d_mem->dTs_ge[T_ID];

  double mdensity = d_mem->density[T_ID];
  double inv_mdensity = 1.0 / mdensity;
  double h2_optical_depth_approx = d_mem->h2_optical_depth_approx[T_ID];


  // scaling the input vector back to cgs units
  #ifdef SCALE_INPUT
  double * __restrict__ scale = d_mem->scale;
  double * __restrict__ inv_scale = d_mem->inv_scale;
  for (int i = 0; i < 10; i++){
    y[INDEX(i)] = y_in[INDEX(i)]*scale[INDEX(i)];
    // printf( "y_in[%d] = %0.5g; scale[%d] = %0.5g\n", i, y_in[INDEX(i)], i, scale[INDEX(i)] );
  }
  #else
  for (int i = 0; i < 10; i++){
    y[INDEX(i)] = y_in[INDEX(i)];
  }
  #endif
  
  evaluate_temperature ( &T_local, &Tge, y, mdensity, rate_data );
  interpolate_reaction_rates( local_reaction_rates, T_local, rate_data);
  interpolate_cooling_rates ( local_cooling_rates , T_local, rate_data);


  //# 0: H2_1
  dy[INDEX(0)] = local_reaction_rates[INDEX(7)]*y[INDEX(2)]*y[INDEX(4)] + local_reaction_rates[INDEX(9)]*y[INDEX(1)]*y[INDEX(2)] - local_reaction_rates[INDEX(10)]*y[INDEX(0)]*y[INDEX(3)] - local_reaction_rates[INDEX(11)]*y[INDEX(0)]*y[INDEX(8)] - local_reaction_rates[INDEX(12)]*y[INDEX(0)]*y[INDEX(2)] + local_reaction_rates[INDEX(18)]*y[INDEX(1)]*y[INDEX(4)] + local_reaction_rates[INDEX(20)]*y[INDEX(0)]*y[INDEX(2)]*y[INDEX(2)] + local_reaction_rates[INDEX(21)]*y[INDEX(2)]*y[INDEX(2)]*y[INDEX(2)];
  //# 1: H2_2
  dy[INDEX(1)] = local_reaction_rates[INDEX(8)]*y[INDEX(2)]*y[INDEX(3)] - local_reaction_rates[INDEX(9)]*y[INDEX(1)]*y[INDEX(2)] + local_reaction_rates[INDEX(10)]*y[INDEX(0)]*y[INDEX(3)] + local_reaction_rates[INDEX(16)]*y[INDEX(3)]*y[INDEX(4)] - local_reaction_rates[INDEX(17)]*y[INDEX(1)]*y[INDEX(8)] - local_reaction_rates[INDEX(18)]*y[INDEX(1)]*y[INDEX(4)];
  //# 2: H_1
  dy[INDEX(2)] = -local_reaction_rates[INDEX(0)]*y[INDEX(2)]*y[INDEX(8)] + local_reaction_rates[INDEX(1)]*y[INDEX(3)]*y[INDEX(8)] - local_reaction_rates[INDEX(6)]*y[INDEX(2)]*y[INDEX(8)] - local_reaction_rates[INDEX(7)]*y[INDEX(2)]*y[INDEX(4)] - local_reaction_rates[INDEX(8)]*y[INDEX(2)]*y[INDEX(3)] - local_reaction_rates[INDEX(9)]*y[INDEX(1)]*y[INDEX(2)] + local_reaction_rates[INDEX(10)]*y[INDEX(0)]*y[INDEX(3)] + 2*local_reaction_rates[INDEX(11)]*y[INDEX(0)]*y[INDEX(8)] + 2*local_reaction_rates[INDEX(12)]*y[INDEX(0)]*y[INDEX(2)] + local_reaction_rates[INDEX(13)]*y[INDEX(4)]*y[INDEX(8)] + local_reaction_rates[INDEX(14)]*y[INDEX(2)]*y[INDEX(4)] + 2*local_reaction_rates[INDEX(15)]*y[INDEX(3)]*y[INDEX(4)] + 2*local_reaction_rates[INDEX(17)]*y[INDEX(1)]*y[INDEX(8)] + local_reaction_rates[INDEX(18)]*y[INDEX(1)]*y[INDEX(4)] - 2*local_reaction_rates[INDEX(20)]*y[INDEX(0)]*y[INDEX(2)]*y[INDEX(2)] - 2*local_reaction_rates[INDEX(21)]*y[INDEX(2)]*y[INDEX(2)]*y[INDEX(2)];
  //# 3: H_2
  dy[INDEX(3)] = local_reaction_rates[INDEX(0)]*y[INDEX(2)]*y[INDEX(8)] - local_reaction_rates[INDEX(1)]*y[INDEX(3)]*y[INDEX(8)] - local_reaction_rates[INDEX(8)]*y[INDEX(2)]*y[INDEX(3)] + local_reaction_rates[INDEX(9)]*y[INDEX(1)]*y[INDEX(2)] - local_reaction_rates[INDEX(10)]*y[INDEX(0)]*y[INDEX(3)] - local_reaction_rates[INDEX(15)]*y[INDEX(3)]*y[INDEX(4)] - local_reaction_rates[INDEX(16)]*y[INDEX(3)]*y[INDEX(4)];
  //# 4: H_m0
  dy[INDEX(4)] = local_reaction_rates[INDEX(6)]*y[INDEX(2)]*y[INDEX(8)] - local_reaction_rates[INDEX(7)]*y[INDEX(2)]*y[INDEX(4)] - local_reaction_rates[INDEX(13)]*y[INDEX(4)]*y[INDEX(8)] - local_reaction_rates[INDEX(14)]*y[INDEX(2)]*y[INDEX(4)] - local_reaction_rates[INDEX(15)]*y[INDEX(3)]*y[INDEX(4)] - local_reaction_rates[INDEX(16)]*y[INDEX(3)]*y[INDEX(4)] - local_reaction_rates[INDEX(18)]*y[INDEX(1)]*y[INDEX(4)];
  //# 5: He_1
  dy[INDEX(5)] = -local_reaction_rates[INDEX(2)]*y[INDEX(5)]*y[INDEX(8)] + local_reaction_rates[INDEX(3)]*y[INDEX(6)]*y[INDEX(8)];
  //# 6: He_2
  dy[INDEX(6)] = local_reaction_rates[INDEX(2)]*y[INDEX(5)]*y[INDEX(8)] - local_reaction_rates[INDEX(3)]*y[INDEX(6)]*y[INDEX(8)] - local_reaction_rates[INDEX(4)]*y[INDEX(6)]*y[INDEX(8)] + local_reaction_rates[INDEX(5)]*y[INDEX(7)]*y[INDEX(8)];
  //# 7: He_3
  dy[INDEX(7)] = local_reaction_rates[INDEX(4)]*y[INDEX(6)]*y[INDEX(8)] - local_reaction_rates[INDEX(5)]*y[INDEX(7)]*y[INDEX(8)];
  //# 8: de
  dy[INDEX(8)] = local_reaction_rates[INDEX(0)]*y[INDEX(2)]*y[INDEX(8)] - local_reaction_rates[INDEX(1)]*y[INDEX(3)]*y[INDEX(8)] + local_reaction_rates[INDEX(2)]*y[INDEX(5)]*y[INDEX(8)] - local_reaction_rates[INDEX(3)]*y[INDEX(6)]*y[INDEX(8)] + local_reaction_rates[INDEX(4)]*y[INDEX(6)]*y[INDEX(8)] - local_reaction_rates[INDEX(5)]*y[INDEX(7)]*y[INDEX(8)] - local_reaction_rates[INDEX(6)]*y[INDEX(2)]*y[INDEX(8)] + local_reaction_rates[INDEX(7)]*y[INDEX(2)]*y[INDEX(4)] + local_reaction_rates[INDEX(13)]*y[INDEX(4)]*y[INDEX(8)] + local_reaction_rates[INDEX(14)]*y[INDEX(2)]*y[INDEX(4)] + local_reaction_rates[INDEX(16)]*y[INDEX(3)]*y[INDEX(4)] - local_reaction_rates[INDEX(17)]*y[INDEX(1)]*y[INDEX(8)];
  //# 9: ge
  dy[INDEX(9)] = -2.01588*y[INDEX(0)]*local_cooling_rates[INDEX(25)]*local_cooling_rates[INDEX(26)]*mdensity - y[INDEX(0)]*local_cooling_rates[INDEX(26)]*local_cooling_rates[INDEX(17)]*h2_optical_depth_approx/(local_cooling_rates[INDEX(17)]/(y[INDEX(0)]*local_cooling_rates[INDEX(13)] + y[INDEX(2)]*local_cooling_rates[INDEX(12)] + y[INDEX(3)]*local_cooling_rates[INDEX(15)] + y[INDEX(5)]*local_cooling_rates[INDEX(14)] + y[INDEX(8)]*local_cooling_rates[INDEX(16)]) + 1.0) - y[INDEX(2)]*local_cooling_rates[INDEX(0)]*local_cooling_rates[INDEX(26)]*y[INDEX(8)] - y[INDEX(2)]*local_cooling_rates[INDEX(4)]*local_cooling_rates[INDEX(26)]*y[INDEX(8)] - y[INDEX(3)]*local_cooling_rates[INDEX(26)]*y[INDEX(8)]*local_cooling_rates[INDEX(7)] - y[INDEX(5)]*local_cooling_rates[INDEX(5)]*local_cooling_rates[INDEX(26)]*y[INDEX(8)] - y[INDEX(6)]*local_cooling_rates[INDEX(2)]*local_cooling_rates[INDEX(26)]*y[INDEX(8)] - y[INDEX(6)]*local_cooling_rates[INDEX(1)]*local_cooling_rates[INDEX(26)]*pow(y[INDEX(8)], 2) - y[INDEX(6)]*local_cooling_rates[INDEX(6)]*local_cooling_rates[INDEX(26)]*y[INDEX(8)] - y[INDEX(6)]*local_cooling_rates[INDEX(3)]*local_cooling_rates[INDEX(26)]*pow(y[INDEX(8)], 2) - y[INDEX(6)]*local_cooling_rates[INDEX(26)]*y[INDEX(8)]*local_cooling_rates[INDEX(8)] - y[INDEX(6)]*local_cooling_rates[INDEX(26)]*y[INDEX(8)]*local_cooling_rates[INDEX(9)] - y[INDEX(7)]*local_cooling_rates[INDEX(26)]*y[INDEX(8)]*local_cooling_rates[INDEX(10)] - local_cooling_rates[INDEX(11)]*local_cooling_rates[INDEX(26)]*y[INDEX(8)]*(y[INDEX(3)] + y[INDEX(6)] + 4.0*y[INDEX(7)]) - local_cooling_rates[INDEX(26)]*local_cooling_rates[INDEX(18)]*y[INDEX(8)]*( T_local - 2.73) + 0.5*1.0/(local_cooling_rates[INDEX(22)]/(y[INDEX(0)]*local_cooling_rates[INDEX(24)] + y[INDEX(2)]*local_cooling_rates[INDEX(23)]) + 1.0)*(-y[INDEX(0)]*y[INDEX(2)]*local_cooling_rates[INDEX(21)] + pow(y[INDEX(2)], 3)*local_cooling_rates[INDEX(20)]);

/*
  dy[INDEX(9)] = -2.01588*y[INDEX(0)]*local_cooling_rates[INDEX(25)]*local_cooling_rates[INDEX(26)]*mdensity - y[INDEX(0)]*local_cooling_rates[INDEX(26)]*local_cooling_rates[INDEX(17)]*h2_optical_depth_approx/(local_cooling_rates[INDEX(17)]/(y[INDEX(0)]*local_cooling_rates[INDEX(13)] + y[INDEX(2)]*local_cooling_rates[INDEX(12)] + y[INDEX(3)]*local_cooling_rates[INDEX(15)] + y[INDEX(5)]*local_cooling_rates[INDEX(14)] + y[INDEX(8)]*local_cooling_rates[INDEX(16)]) + 1.0) - y[INDEX(2)]*local_cooling_rates[INDEX(0)]*local_cooling_rates[INDEX(26)]*y[INDEX(8)] - y[INDEX(2)]*local_cooling_rates[INDEX(4)]*local_cooling_rates[INDEX(26)]*y[INDEX(8)] - y[INDEX(3)]*local_cooling_rates[INDEX(26)]*y[INDEX(8)]*local_cooling_rates[INDEX(7)] - y[INDEX(5)]*local_cooling_rates[INDEX(5)]*local_cooling_rates[INDEX(26)]*y[INDEX(8)] - y[INDEX(6)]*local_cooling_rates[INDEX(2)]*local_cooling_rates[INDEX(26)]*y[INDEX(8)] - y[INDEX(6)]*local_cooling_rates[INDEX(1)]*local_cooling_rates[INDEX(26)]*pow(y[INDEX(8)], 2) - y[INDEX(6)]*local_cooling_rates[INDEX(6)]*local_cooling_rates[INDEX(26)]*y[INDEX(8)] - y[INDEX(6)]*local_cooling_rates[INDEX(3)]*local_cooling_rates[INDEX(26)]*pow(y[INDEX(8)], 2) - y[INDEX(6)]*local_cooling_rates[INDEX(26)]*y[INDEX(8)]*local_cooling_rates[INDEX(8)] - y[INDEX(6)]*local_cooling_rates[INDEX(26)]*y[INDEX(8)]*local_cooling_rates[INDEX(9)] - y[INDEX(7)]*local_cooling_rates[INDEX(26)]*y[INDEX(8)]*local_cooling_rates[INDEX(10)] - local_cooling_rates[INDEX(11)]*local_cooling_rates[INDEX(26)]*y[INDEX(8)]*(y[INDEX(3)] + y[INDEX(6)] + 4.0*y[INDEX(7)]) - local_cooling_rates[INDEX(26)]*local_cooling_rates[INDEX(18)]*y[INDEX(8)]*(T_local - 2.73) + 0.5*1.0/(local_cooling_rates[INDEX(22)]/(y[INDEX(0)]*local_cooling_rates[INDEX(24)] + y[INDEX(2)]*local_cooling_rates[INDEX(23)]) + 1.0)*(-y[INDEX(0)]*y[INDEX(2)]*local_cooling_rates[INDEX(21)] + pow(y[INDEX(2)], 3)*local_cooling_rates[INDEX(20)]);
*/
  dy[INDEX(9)] *= inv_mdensity;



  #ifdef SCALE_INPUT
  // scaling the dydt vector back to code untis
  for (int i = 0; i< 10; i++){
    dy[INDEX(i)] *= inv_scale[INDEX(i)];
  }
  #endif

/*
  if ( T_ID == 0 ){
    *d_mem->rhs_call += 1;
    printf("t = %0.5g; rhs_call = %d\n", t, *d_mem->rhs_call );
  }
*/

/*
  if ( T_ID == 0 ){
    printf("time = %0.5g, at temp = %0.5g\n", t, T_local);
    for (int i = 0; i< 10; i++){
      printf("from tid[%d]: dy[%d] = %0.5g, y = %0.5g at t = %0.5g \n", T_ID, i, dy[INDEX(i)], y_in[INDEX(i)], t);
    }
  }
*/

//  printf(" \n");
//  }
}

