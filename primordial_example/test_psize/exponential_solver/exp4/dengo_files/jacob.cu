#include "jacob.cuh"

__device__ void eval_jacob (const double t, const double pres, const double * __restrict__ y_in, double * __restrict__ jac, const mechanism_memory * d_mem) {

  double * local_reaction_rates = d_mem->reaction_rates;
  double * local_cooling_rates  = d_mem->cooling_rates ;
  double * rlocal_reaction_rates = d_mem->drrate_dT;
  double * rlocal_cooling_rates  = d_mem->dcrate_dT;

  // scale related piece
  double * y = d_mem->temp_array; // working space for scaling the variable back;
  cvklu_data *rate_data = d_mem->chemistry_data;

  // these should be retreieved from d_mem object
  double T_local  = d_mem->temperature[T_ID];
  double Tge = d_mem->dTs_ge[T_ID];

  double mdensity = d_mem->density[T_ID];
  double inv_mdensity = 1.0 / mdensity;
  double h2_optical_depth_approx = d_mem->h2_optical_depth_approx[T_ID];

  // scaling the input vector back to cgs units
  #ifdef SCALE_INPUT
  double * __restrict__ scale     = d_mem->scale;
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
//  interpolate_reaction_rates ( local_reaction_rates, T_local, rate_data);
//  interpolate_cooling_rates  ( local_cooling_rates , T_local, rate_data); 
  interpolate_drrate_dT( rlocal_reaction_rates, T_local, rate_data );
  interpolate_dcrate_dT( rlocal_cooling_rates, T_local, rate_data );

  if (T_ID == 0 ){
    printf("FROM JAC[%ld]: at time = %0.5g, t_local = %0.5g, h2_od: %0.5g\n", T_ID,t, T_local, h2_optical_depth_approx);
  }

/*
    for (int i = 0; i < 23; i++){
      printf("reaction rate[%d] = %0.5g\n", i, local_reaction_rates[INDEX(i)]);
    }
    for (int i = 0; i < 23; i++){
      printf("drrate_dT    [%d] = %0.5g\n", i, rlocal_reaction_rates[INDEX(i)]);
    }
    printf("\n");
*/
//    *d_mem->jac_call += 1;
//    printf("jac_call = %d\n", *d_mem->jac_call );
//  }

  // df_H2_1 / H2_1:
  jac[INDEX(0)] = -local_reaction_rates[INDEX(10)]*y[INDEX(3)] - local_reaction_rates[INDEX(11)]*y[INDEX(8)] - local_reaction_rates[INDEX(12)]*y[INDEX(2)] + local_reaction_rates[INDEX(20)]*pow(y[INDEX(2)], 2);
 
  // df_H2_2 / H2_1:
  jac[INDEX(1)] = local_reaction_rates[INDEX(10)]*y[INDEX(3)];
 
  // df_H_1 / H2_1:
  jac[INDEX(2)] = local_reaction_rates[INDEX(10)]*y[INDEX(3)] + 2*local_reaction_rates[INDEX(11)]*y[INDEX(8)] + 2*local_reaction_rates[INDEX(12)]*y[INDEX(2)] - 2*local_reaction_rates[INDEX(20)]*pow(y[INDEX(2)], 2);
 
  // df_H_2 / H2_1:
  jac[INDEX(3)] = -local_reaction_rates[INDEX(10)]*y[INDEX(3)];
 
  // df_ge / H2_1:
  jac[INDEX(9)] = -y[INDEX(0)]*local_cooling_rates[INDEX(13)]*pow(local_cooling_rates[INDEX(17)], 2)*h2_optical_depth_approx/(pow(local_cooling_rates[INDEX(17)]/(y[INDEX(0)]*local_cooling_rates[INDEX(13)] + y[INDEX(2)]*local_cooling_rates[INDEX(12)] + y[INDEX(3)]*local_cooling_rates[INDEX(15)] + y[INDEX(5)]*local_cooling_rates[INDEX(14)] + y[INDEX(8)]*local_cooling_rates[INDEX(16)]) + 1.0, 2)*pow(y[INDEX(0)]*local_cooling_rates[INDEX(13)] + y[INDEX(2)]*local_cooling_rates[INDEX(12)] + y[INDEX(3)]*local_cooling_rates[INDEX(15)] + y[INDEX(5)]*local_cooling_rates[INDEX(14)] + y[INDEX(8)]*local_cooling_rates[INDEX(16)], 2)) - 0.5*y[INDEX(2)]*local_cooling_rates[INDEX(21)]*1.0/(local_cooling_rates[INDEX(22)]/(y[INDEX(0)]*local_cooling_rates[INDEX(24)] + y[INDEX(2)]*local_cooling_rates[INDEX(23)]) + 1.0) - 2.01588*local_cooling_rates[INDEX(25)]*mdensity - local_cooling_rates[INDEX(17)]*h2_optical_depth_approx/(local_cooling_rates[INDEX(17)]/(y[INDEX(0)]*local_cooling_rates[INDEX(13)] + y[INDEX(2)]*local_cooling_rates[INDEX(12)] + y[INDEX(3)]*local_cooling_rates[INDEX(15)] + y[INDEX(5)]*local_cooling_rates[INDEX(14)] + y[INDEX(8)]*local_cooling_rates[INDEX(16)]) + 1.0) + 0.5*local_cooling_rates[INDEX(24)]*local_cooling_rates[INDEX(22)]*pow(local_cooling_rates[INDEX(22)]/(y[INDEX(0)]*local_cooling_rates[INDEX(24)] + y[INDEX(2)]*local_cooling_rates[INDEX(23)]) + 1.0, -2.0)*(-y[INDEX(0)]*y[INDEX(2)]*local_cooling_rates[INDEX(21)] + pow(y[INDEX(2)], 3)*local_cooling_rates[INDEX(20)])/pow(y[INDEX(0)]*local_cooling_rates[INDEX(24)] + y[INDEX(2)]*local_cooling_rates[INDEX(23)], 2);
  jac[INDEX(9)] *= inv_mdensity;
 
  // df_H2_1 / H2_2:
  jac[INDEX(10)] = local_reaction_rates[INDEX(9)]*y[INDEX(2)] + local_reaction_rates[INDEX(18)]*y[INDEX(4)];
 
  // df_H2_2 / H2_2:
  jac[INDEX(11)] = -local_reaction_rates[INDEX(9)]*y[INDEX(2)] - local_reaction_rates[INDEX(17)]*y[INDEX(8)] - local_reaction_rates[INDEX(18)]*y[INDEX(4)];
 
  // df_H_1 / H2_2:
  jac[INDEX(12)] = -local_reaction_rates[INDEX(9)]*y[INDEX(2)] + 2*local_reaction_rates[INDEX(17)]*y[INDEX(8)] + local_reaction_rates[INDEX(18)]*y[INDEX(4)];
 
  // df_H_2 / H2_2:
  jac[INDEX(13)] = local_reaction_rates[INDEX(9)]*y[INDEX(2)];
 
  // df_H_m0 / H2_2:
  jac[INDEX(14)] = -local_reaction_rates[INDEX(18)]*y[INDEX(4)];
 
  // df_de / H2_2:
  jac[INDEX(18)] = -local_reaction_rates[INDEX(17)]*y[INDEX(8)];
 
  // df_H2_1 / H_1:
  jac[INDEX(20)] = local_reaction_rates[INDEX(7)]*y[INDEX(4)] + local_reaction_rates[INDEX(9)]*y[INDEX(1)] - local_reaction_rates[INDEX(12)]*y[INDEX(0)] + 2*local_reaction_rates[INDEX(20)]*y[INDEX(0)]*y[INDEX(2)] + 3*local_reaction_rates[INDEX(21)]*pow(y[INDEX(2)], 2);
 
  // df_H2_2 / H_1:
  jac[INDEX(21)] = local_reaction_rates[INDEX(8)]*y[INDEX(3)] - local_reaction_rates[INDEX(9)]*y[INDEX(1)];
 
  // df_H_1 / H_1:
  jac[INDEX(22)] = -local_reaction_rates[INDEX(0)]*y[INDEX(8)] - local_reaction_rates[INDEX(6)]*y[INDEX(8)] - local_reaction_rates[INDEX(7)]*y[INDEX(4)] - local_reaction_rates[INDEX(8)]*y[INDEX(3)] - local_reaction_rates[INDEX(9)]*y[INDEX(1)] + 2*local_reaction_rates[INDEX(12)]*y[INDEX(0)] + local_reaction_rates[INDEX(14)]*y[INDEX(4)] - 4*local_reaction_rates[INDEX(20)]*y[INDEX(0)]*y[INDEX(2)] - 6*local_reaction_rates[INDEX(21)]*pow(y[INDEX(2)], 2);
 
  // df_H_2 / H_1:
  jac[INDEX(23)] = local_reaction_rates[INDEX(0)]*y[INDEX(8)] - local_reaction_rates[INDEX(8)]*y[INDEX(3)] + local_reaction_rates[INDEX(9)]*y[INDEX(1)];
 
  // df_H_m0 / H_1:
  jac[INDEX(24)] = local_reaction_rates[INDEX(6)]*y[INDEX(8)] - local_reaction_rates[INDEX(7)]*y[INDEX(4)] - local_reaction_rates[INDEX(14)]*y[INDEX(4)];
 
  // df_de / H_1:
  jac[INDEX(28)] = local_reaction_rates[INDEX(0)]*y[INDEX(8)] - local_reaction_rates[INDEX(6)]*y[INDEX(8)] + local_reaction_rates[INDEX(7)]*y[INDEX(4)] + local_reaction_rates[INDEX(14)]*y[INDEX(4)];
 
  // df_ge / H_1:
  jac[INDEX(29)] = -y[INDEX(0)]*local_cooling_rates[INDEX(12)]*pow(local_cooling_rates[INDEX(17)], 2)*h2_optical_depth_approx/(pow(local_cooling_rates[INDEX(17)]/(y[INDEX(0)]*local_cooling_rates[INDEX(13)] + y[INDEX(2)]*local_cooling_rates[INDEX(12)] + y[INDEX(3)]*local_cooling_rates[INDEX(15)] + y[INDEX(5)]*local_cooling_rates[INDEX(14)] + y[INDEX(8)]*local_cooling_rates[INDEX(16)]) + 1.0, 2)*pow(y[INDEX(0)]*local_cooling_rates[INDEX(13)] + y[INDEX(2)]*local_cooling_rates[INDEX(12)] + y[INDEX(3)]*local_cooling_rates[INDEX(15)] + y[INDEX(5)]*local_cooling_rates[INDEX(14)] + y[INDEX(8)]*local_cooling_rates[INDEX(16)], 2)) - local_cooling_rates[INDEX(0)]*y[INDEX(8)] - local_cooling_rates[INDEX(4)]*y[INDEX(8)] + 0.5*local_cooling_rates[INDEX(23)]*local_cooling_rates[INDEX(22)]*pow(local_cooling_rates[INDEX(22)]/(y[INDEX(0)]*local_cooling_rates[INDEX(24)] + y[INDEX(2)]*local_cooling_rates[INDEX(23)]) + 1.0, -2.0)*(-y[INDEX(0)]*y[INDEX(2)]*local_cooling_rates[INDEX(21)] + pow(y[INDEX(2)], 3)*local_cooling_rates[INDEX(20)])/pow(y[INDEX(0)]*local_cooling_rates[INDEX(24)] + y[INDEX(2)]*local_cooling_rates[INDEX(23)], 2) + 0.5*(-y[INDEX(0)]*local_cooling_rates[INDEX(21)] + 3*pow(y[INDEX(2)], 2)*local_cooling_rates[INDEX(20)])*1.0/(local_cooling_rates[INDEX(22)]/(y[INDEX(0)]*local_cooling_rates[INDEX(24)] + y[INDEX(2)]*local_cooling_rates[INDEX(23)]) + 1.0);
  jac[INDEX(29)] *= inv_mdensity;
 
  // df_H2_1 / H_2:
  jac[INDEX(30)] = -local_reaction_rates[INDEX(10)]*y[INDEX(0)];
 
  // df_H2_2 / H_2:
  jac[INDEX(31)] = local_reaction_rates[INDEX(8)]*y[INDEX(2)] + local_reaction_rates[INDEX(10)]*y[INDEX(0)] + local_reaction_rates[INDEX(16)]*y[INDEX(4)];
 
  // df_H_1 / H_2:
  jac[INDEX(32)] = local_reaction_rates[INDEX(1)]*y[INDEX(8)] - local_reaction_rates[INDEX(8)]*y[INDEX(2)] + local_reaction_rates[INDEX(10)]*y[INDEX(0)] + 2*local_reaction_rates[INDEX(15)]*y[INDEX(4)];
 
  // df_H_2 / H_2:
  jac[INDEX(33)] = -local_reaction_rates[INDEX(1)]*y[INDEX(8)] - local_reaction_rates[INDEX(8)]*y[INDEX(2)] - local_reaction_rates[INDEX(10)]*y[INDEX(0)] - local_reaction_rates[INDEX(15)]*y[INDEX(4)] - local_reaction_rates[INDEX(16)]*y[INDEX(4)];
 
  // df_H_m0 / H_2:
  jac[INDEX(34)] = -local_reaction_rates[INDEX(15)]*y[INDEX(4)] - local_reaction_rates[INDEX(16)]*y[INDEX(4)];
 
  // df_de / H_2:
  jac[INDEX(38)] = -local_reaction_rates[INDEX(1)]*y[INDEX(8)] + local_reaction_rates[INDEX(16)]*y[INDEX(4)];
 
  // df_ge / H_2:
  jac[INDEX(39)] = -y[INDEX(0)]*local_cooling_rates[INDEX(15)]*pow(local_cooling_rates[INDEX(17)], 2)*h2_optical_depth_approx/(pow(local_cooling_rates[INDEX(17)]/(y[INDEX(0)]*local_cooling_rates[INDEX(13)] + y[INDEX(2)]*local_cooling_rates[INDEX(12)] + y[INDEX(3)]*local_cooling_rates[INDEX(15)] + y[INDEX(5)]*local_cooling_rates[INDEX(14)] + y[INDEX(8)]*local_cooling_rates[INDEX(16)]) + 1.0, 2)*pow(y[INDEX(0)]*local_cooling_rates[INDEX(13)] + y[INDEX(2)]*local_cooling_rates[INDEX(12)] + y[INDEX(3)]*local_cooling_rates[INDEX(15)] + y[INDEX(5)]*local_cooling_rates[INDEX(14)] + y[INDEX(8)]*local_cooling_rates[INDEX(16)], 2)) - local_cooling_rates[INDEX(11)]*y[INDEX(8)] - y[INDEX(8)]*local_cooling_rates[INDEX(7)];
  jac[INDEX(39)] *= inv_mdensity;
 
  // df_H2_1 / H_m0:
  jac[INDEX(40)] = local_reaction_rates[INDEX(7)]*y[INDEX(2)] + local_reaction_rates[INDEX(18)]*y[INDEX(1)];
 
  // df_H2_2 / H_m0:
  jac[INDEX(41)] = local_reaction_rates[INDEX(16)]*y[INDEX(3)] - local_reaction_rates[INDEX(18)]*y[INDEX(1)];
 
  // df_H_1 / H_m0:
  jac[INDEX(42)] = -local_reaction_rates[INDEX(7)]*y[INDEX(2)] + local_reaction_rates[INDEX(13)]*y[INDEX(8)] + local_reaction_rates[INDEX(14)]*y[INDEX(2)] + 2*local_reaction_rates[INDEX(15)]*y[INDEX(3)] + local_reaction_rates[INDEX(18)]*y[INDEX(1)];
 
  // df_H_2 / H_m0:
  jac[INDEX(43)] = -local_reaction_rates[INDEX(15)]*y[INDEX(3)] - local_reaction_rates[INDEX(16)]*y[INDEX(3)];
 
  // df_H_m0 / H_m0:
  jac[INDEX(44)] = -local_reaction_rates[INDEX(7)]*y[INDEX(2)] - local_reaction_rates[INDEX(13)]*y[INDEX(8)] - local_reaction_rates[INDEX(14)]*y[INDEX(2)] - local_reaction_rates[INDEX(15)]*y[INDEX(3)] - local_reaction_rates[INDEX(16)]*y[INDEX(3)] - local_reaction_rates[INDEX(18)]*y[INDEX(1)];
 
  // df_de / H_m0:
  jac[INDEX(48)] = local_reaction_rates[INDEX(7)]*y[INDEX(2)] + local_reaction_rates[INDEX(13)]*y[INDEX(8)] + local_reaction_rates[INDEX(14)]*y[INDEX(2)] + local_reaction_rates[INDEX(16)]*y[INDEX(3)];
 
  // df_He_1 / He_1:
  jac[INDEX(55)] = -local_reaction_rates[INDEX(2)]*y[INDEX(8)];
 
  // df_He_2 / He_1:
  jac[INDEX(56)] = local_reaction_rates[INDEX(2)]*y[INDEX(8)];
 
  // df_de / He_1:
  jac[INDEX(58)] = local_reaction_rates[INDEX(2)]*y[INDEX(8)];
 
  // df_ge / He_1:
  jac[INDEX(59)] = -y[INDEX(0)]*local_cooling_rates[INDEX(14)]*pow(local_cooling_rates[INDEX(17)], 2)*h2_optical_depth_approx/(pow(local_cooling_rates[INDEX(17)]/(y[INDEX(0)]*local_cooling_rates[INDEX(13)] + y[INDEX(2)]*local_cooling_rates[INDEX(12)] + y[INDEX(3)]*local_cooling_rates[INDEX(15)] + y[INDEX(5)]*local_cooling_rates[INDEX(14)] + y[INDEX(8)]*local_cooling_rates[INDEX(16)]) + 1.0, 2)*pow(y[INDEX(0)]*local_cooling_rates[INDEX(13)] + y[INDEX(2)]*local_cooling_rates[INDEX(12)] + y[INDEX(3)]*local_cooling_rates[INDEX(15)] + y[INDEX(5)]*local_cooling_rates[INDEX(14)] + y[INDEX(8)]*local_cooling_rates[INDEX(16)], 2)) - local_cooling_rates[INDEX(5)]*y[INDEX(8)];
  jac[INDEX(59)] *= inv_mdensity;
 
  // df_He_1 / He_2:
  jac[INDEX(65)] = local_reaction_rates[INDEX(3)]*y[INDEX(8)];
 
  // df_He_2 / He_2:
  jac[INDEX(66)] = -local_reaction_rates[INDEX(3)]*y[INDEX(8)] - local_reaction_rates[INDEX(4)]*y[INDEX(8)];
 
  // df_He_3 / He_2:
  jac[INDEX(67)] = local_reaction_rates[INDEX(4)]*y[INDEX(8)];
 
  // df_de / He_2:
  jac[INDEX(68)] = -local_reaction_rates[INDEX(3)]*y[INDEX(8)] + local_reaction_rates[INDEX(4)]*y[INDEX(8)];
 
  // df_ge / He_2:
  jac[INDEX(69)] = -local_cooling_rates[INDEX(11)]*y[INDEX(8)] - local_cooling_rates[INDEX(2)]*y[INDEX(8)] - local_cooling_rates[INDEX(1)]*pow(y[INDEX(8)], 2) - local_cooling_rates[INDEX(6)]*y[INDEX(8)] - local_cooling_rates[INDEX(3)]*pow(y[INDEX(8)], 2) - y[INDEX(8)]*local_cooling_rates[INDEX(8)] - y[INDEX(8)]*local_cooling_rates[INDEX(9)];
  jac[INDEX(69)] *= inv_mdensity;
 
  // df_He_2 / He_3:
  jac[INDEX(76)] = local_reaction_rates[INDEX(5)]*y[INDEX(8)];
 
  // df_He_3 / He_3:
  jac[INDEX(77)] = -local_reaction_rates[INDEX(5)]*y[INDEX(8)];
 
  // df_de / He_3:
  jac[INDEX(78)] = -local_reaction_rates[INDEX(5)]*y[INDEX(8)];
 
  // df_ge / He_3:
  jac[INDEX(79)] = -4.0*local_cooling_rates[INDEX(11)]*y[INDEX(8)] - y[INDEX(8)]*local_cooling_rates[INDEX(10)];
  jac[INDEX(79)] *= inv_mdensity;
 
  // df_H2_1 / de:
  jac[INDEX(80)] = -local_reaction_rates[INDEX(11)]*y[INDEX(0)];
 
  // df_H2_2 / de:
  jac[INDEX(81)] = -local_reaction_rates[INDEX(17)]*y[INDEX(1)];
 
  // df_H_1 / de:
  jac[INDEX(82)] = -local_reaction_rates[INDEX(0)]*y[INDEX(2)] + local_reaction_rates[INDEX(1)]*y[INDEX(3)] - local_reaction_rates[INDEX(6)]*y[INDEX(2)] + 2*local_reaction_rates[INDEX(11)]*y[INDEX(0)] + local_reaction_rates[INDEX(13)]*y[INDEX(4)] + 2*local_reaction_rates[INDEX(17)]*y[INDEX(1)];
 
  // df_H_2 / de:
  jac[INDEX(83)] = local_reaction_rates[INDEX(0)]*y[INDEX(2)] - local_reaction_rates[INDEX(1)]*y[INDEX(3)];
 
  // df_H_m0 / de:
  jac[INDEX(84)] = local_reaction_rates[INDEX(6)]*y[INDEX(2)] - local_reaction_rates[INDEX(13)]*y[INDEX(4)];
 
  // df_He_1 / de:
  jac[INDEX(85)] = -local_reaction_rates[INDEX(2)]*y[INDEX(5)] + local_reaction_rates[INDEX(3)]*y[INDEX(6)];
 
  // df_He_2 / de:
  jac[INDEX(86)] = local_reaction_rates[INDEX(2)]*y[INDEX(5)] - local_reaction_rates[INDEX(3)]*y[INDEX(6)] - local_reaction_rates[INDEX(4)]*y[INDEX(6)] + local_reaction_rates[INDEX(5)]*y[INDEX(7)];
 
  // df_He_3 / de:
  jac[INDEX(87)] = local_reaction_rates[INDEX(4)]*y[INDEX(6)] - local_reaction_rates[INDEX(5)]*y[INDEX(7)];
 
  // df_de / de:
  jac[INDEX(88)] = local_reaction_rates[INDEX(0)]*y[INDEX(2)] - local_reaction_rates[INDEX(1)]*y[INDEX(3)] + local_reaction_rates[INDEX(2)]*y[INDEX(5)] - local_reaction_rates[INDEX(3)]*y[INDEX(6)] + local_reaction_rates[INDEX(4)]*y[INDEX(6)] - local_reaction_rates[INDEX(5)]*y[INDEX(7)] - local_reaction_rates[INDEX(6)]*y[INDEX(2)] + local_reaction_rates[INDEX(13)]*y[INDEX(4)] - local_reaction_rates[INDEX(17)]*y[INDEX(1)];
 
  // df_ge / de:
  jac[INDEX(89)] = -y[INDEX(0)]*local_cooling_rates[INDEX(16)]*pow(local_cooling_rates[INDEX(17)], 2)*h2_optical_depth_approx/(pow(local_cooling_rates[INDEX(17)]/(y[INDEX(0)]*local_cooling_rates[INDEX(13)] + y[INDEX(2)]*local_cooling_rates[INDEX(12)] + y[INDEX(3)]*local_cooling_rates[INDEX(15)] + y[INDEX(5)]*local_cooling_rates[INDEX(14)] + y[INDEX(8)]*local_cooling_rates[INDEX(16)]) + 1.0, 2)*pow(y[INDEX(0)]*local_cooling_rates[INDEX(13)] + y[INDEX(2)]*local_cooling_rates[INDEX(12)] + y[INDEX(3)]*local_cooling_rates[INDEX(15)] + y[INDEX(5)]*local_cooling_rates[INDEX(14)] + y[INDEX(8)]*local_cooling_rates[INDEX(16)], 2)) - y[INDEX(2)]*local_cooling_rates[INDEX(0)] - y[INDEX(2)]*local_cooling_rates[INDEX(4)] - y[INDEX(3)]*local_cooling_rates[INDEX(7)] - y[INDEX(5)]*local_cooling_rates[INDEX(5)] - y[INDEX(6)]*local_cooling_rates[INDEX(2)] - 2*y[INDEX(6)]*local_cooling_rates[INDEX(1)]*y[INDEX(8)] - y[INDEX(6)]*local_cooling_rates[INDEX(6)] - 2*y[INDEX(6)]*local_cooling_rates[INDEX(3)]*y[INDEX(8)] - y[INDEX(6)]*local_cooling_rates[INDEX(8)] - y[INDEX(6)]*local_cooling_rates[INDEX(9)] - y[INDEX(7)]*local_cooling_rates[INDEX(10)] - local_cooling_rates[INDEX(11)]*(y[INDEX(3)] + y[INDEX(6)] + 4.0*y[INDEX(7)]) - local_cooling_rates[INDEX(18)]*(T_local - 2.73);
  jac[INDEX(89)] *= inv_mdensity;
 
  // df_H2_1 / ge:
  jac[INDEX(90)] = rlocal_reaction_rates[INDEX(7)]*y[INDEX(2)]*y[INDEX(4)] + rlocal_reaction_rates[INDEX(9)]*y[INDEX(1)]*y[INDEX(2)] - rlocal_reaction_rates[INDEX(10)]*y[INDEX(0)]*y[INDEX(3)] - rlocal_reaction_rates[INDEX(11)]*y[INDEX(0)]*y[INDEX(8)] - rlocal_reaction_rates[INDEX(12)]*y[INDEX(0)]*y[INDEX(2)] + rlocal_reaction_rates[INDEX(18)]*y[INDEX(1)]*y[INDEX(4)] + rlocal_reaction_rates[INDEX(20)]*y[INDEX(0)]*y[INDEX(2)]*y[INDEX(2)] + rlocal_reaction_rates[INDEX(21)]*y[INDEX(2)]*y[INDEX(2)]*y[INDEX(2)];
  jac[INDEX(90)] *= Tge;
 
  // df_H2_2 / ge:
  jac[INDEX(91)] = rlocal_reaction_rates[INDEX(8)]*y[INDEX(2)]*y[INDEX(3)] - rlocal_reaction_rates[INDEX(9)]*y[INDEX(1)]*y[INDEX(2)] + rlocal_reaction_rates[INDEX(10)]*y[INDEX(0)]*y[INDEX(3)] + rlocal_reaction_rates[INDEX(16)]*y[INDEX(3)]*y[INDEX(4)] - rlocal_reaction_rates[INDEX(17)]*y[INDEX(1)]*y[INDEX(8)] - rlocal_reaction_rates[INDEX(18)]*y[INDEX(1)]*y[INDEX(4)];
  jac[INDEX(91)] *= Tge;
 
  // df_H_1 / ge:
  jac[INDEX(92)] = -rlocal_reaction_rates[INDEX(0)]*y[INDEX(2)]*y[INDEX(8)] + rlocal_reaction_rates[INDEX(1)]*y[INDEX(3)]*y[INDEX(8)] - rlocal_reaction_rates[INDEX(6)]*y[INDEX(2)]*y[INDEX(8)] - rlocal_reaction_rates[INDEX(7)]*y[INDEX(2)]*y[INDEX(4)] - rlocal_reaction_rates[INDEX(8)]*y[INDEX(2)]*y[INDEX(3)] - rlocal_reaction_rates[INDEX(9)]*y[INDEX(1)]*y[INDEX(2)] + rlocal_reaction_rates[INDEX(10)]*y[INDEX(0)]*y[INDEX(3)] + 2*rlocal_reaction_rates[INDEX(11)]*y[INDEX(0)]*y[INDEX(8)] + 2*rlocal_reaction_rates[INDEX(12)]*y[INDEX(0)]*y[INDEX(2)] + rlocal_reaction_rates[INDEX(13)]*y[INDEX(4)]*y[INDEX(8)] + rlocal_reaction_rates[INDEX(14)]*y[INDEX(2)]*y[INDEX(4)] + 2*rlocal_reaction_rates[INDEX(15)]*y[INDEX(3)]*y[INDEX(4)] + 2*rlocal_reaction_rates[INDEX(17)]*y[INDEX(1)]*y[INDEX(8)] + rlocal_reaction_rates[INDEX(18)]*y[INDEX(1)]*y[INDEX(4)] - 2*rlocal_reaction_rates[INDEX(20)]*y[INDEX(0)]*y[INDEX(2)]*y[INDEX(2)] - 2*rlocal_reaction_rates[INDEX(21)]*y[INDEX(2)]*y[INDEX(2)]*y[INDEX(2)];
  jac[INDEX(92)] *= Tge;
 
  // df_H_2 / ge:
  jac[INDEX(93)] = rlocal_reaction_rates[INDEX(0)]*y[INDEX(2)]*y[INDEX(8)] - rlocal_reaction_rates[INDEX(1)]*y[INDEX(3)]*y[INDEX(8)] - rlocal_reaction_rates[INDEX(8)]*y[INDEX(2)]*y[INDEX(3)] + rlocal_reaction_rates[INDEX(9)]*y[INDEX(1)]*y[INDEX(2)] - rlocal_reaction_rates[INDEX(10)]*y[INDEX(0)]*y[INDEX(3)] - rlocal_reaction_rates[INDEX(15)]*y[INDEX(3)]*y[INDEX(4)] - rlocal_reaction_rates[INDEX(16)]*y[INDEX(3)]*y[INDEX(4)];
  jac[INDEX(93)] *= Tge;
 
  // df_H_m0 / ge:
  jac[INDEX(94)] = rlocal_reaction_rates[INDEX(6)]*y[INDEX(2)]*y[INDEX(8)] - rlocal_reaction_rates[INDEX(7)]*y[INDEX(2)]*y[INDEX(4)] - rlocal_reaction_rates[INDEX(13)]*y[INDEX(4)]*y[INDEX(8)] - rlocal_reaction_rates[INDEX(14)]*y[INDEX(2)]*y[INDEX(4)] - rlocal_reaction_rates[INDEX(15)]*y[INDEX(3)]*y[INDEX(4)] - rlocal_reaction_rates[INDEX(16)]*y[INDEX(3)]*y[INDEX(4)] - rlocal_reaction_rates[INDEX(18)]*y[INDEX(1)]*y[INDEX(4)];
  jac[INDEX(94)] *= Tge;
 
  // df_He_1 / ge:
  jac[INDEX(95)] = -rlocal_reaction_rates[INDEX(2)]*y[INDEX(5)]*y[INDEX(8)] + rlocal_reaction_rates[INDEX(3)]*y[INDEX(6)]*y[INDEX(8)];
  jac[INDEX(95)] *= Tge;
 
  // df_He_2 / ge:
  jac[INDEX(96)] = rlocal_reaction_rates[INDEX(2)]*y[INDEX(5)]*y[INDEX(8)] - rlocal_reaction_rates[INDEX(3)]*y[INDEX(6)]*y[INDEX(8)] - rlocal_reaction_rates[INDEX(4)]*y[INDEX(6)]*y[INDEX(8)] + rlocal_reaction_rates[INDEX(5)]*y[INDEX(7)]*y[INDEX(8)];
  jac[INDEX(96)] *= Tge;
 
  // df_He_3 / ge:
  jac[INDEX(97)] = rlocal_reaction_rates[INDEX(4)]*y[INDEX(6)]*y[INDEX(8)] - rlocal_reaction_rates[INDEX(5)]*y[INDEX(7)]*y[INDEX(8)];
  jac[INDEX(97)] *= Tge;
 
  // df_de / ge:
  jac[INDEX(98)] = rlocal_reaction_rates[INDEX(0)]*y[INDEX(2)]*y[INDEX(8)] - rlocal_reaction_rates[INDEX(1)]*y[INDEX(3)]*y[INDEX(8)] + rlocal_reaction_rates[INDEX(2)]*y[INDEX(5)]*y[INDEX(8)] - rlocal_reaction_rates[INDEX(3)]*y[INDEX(6)]*y[INDEX(8)] + rlocal_reaction_rates[INDEX(4)]*y[INDEX(6)]*y[INDEX(8)] - rlocal_reaction_rates[INDEX(5)]*y[INDEX(7)]*y[INDEX(8)] - rlocal_reaction_rates[INDEX(6)]*y[INDEX(2)]*y[INDEX(8)] + rlocal_reaction_rates[INDEX(7)]*y[INDEX(2)]*y[INDEX(4)] + rlocal_reaction_rates[INDEX(13)]*y[INDEX(4)]*y[INDEX(8)] + rlocal_reaction_rates[INDEX(14)]*y[INDEX(2)]*y[INDEX(4)] + rlocal_reaction_rates[INDEX(16)]*y[INDEX(3)]*y[INDEX(4)] - rlocal_reaction_rates[INDEX(17)]*y[INDEX(1)]*y[INDEX(8)];
  jac[INDEX(98)] *= Tge;
 
  // df_ge / ge:
  jac[INDEX(99)] =- y[INDEX(0)]*local_cooling_rates[INDEX(17)]*h2_optical_depth_approx*(-local_cooling_rates[INDEX(17)]*(-y[INDEX(0)]*rlocal_cooling_rates[INDEX(13)] - y[INDEX(2)]*rlocal_cooling_rates[INDEX(12)] - y[INDEX(3)]*rlocal_cooling_rates[INDEX(15)] - y[INDEX(5)]*rlocal_cooling_rates[INDEX(14)] - y[INDEX(8)]*rlocal_cooling_rates[INDEX(16)])/pow(y[INDEX(0)]*local_cooling_rates[INDEX(13)] + y[INDEX(2)]*local_cooling_rates[INDEX(12)] + y[INDEX(3)]*local_cooling_rates[INDEX(15)] + y[INDEX(5)]*local_cooling_rates[INDEX(14)] + y[INDEX(8)]*local_cooling_rates[INDEX(16)], 2) - rlocal_cooling_rates[INDEX(17)]/(y[INDEX(0)]*local_cooling_rates[INDEX(13)] + y[INDEX(2)]*local_cooling_rates[INDEX(12)] + y[INDEX(3)]*local_cooling_rates[INDEX(15)] + y[INDEX(5)]*local_cooling_rates[INDEX(14)] + y[INDEX(8)]*local_cooling_rates[INDEX(16)]))/pow(local_cooling_rates[INDEX(17)]/(y[INDEX(0)]*local_cooling_rates[INDEX(13)] + y[INDEX(2)]*local_cooling_rates[INDEX(12)] + y[INDEX(3)]*local_cooling_rates[INDEX(15)] + y[INDEX(5)]*local_cooling_rates[INDEX(14)] + y[INDEX(8)]*local_cooling_rates[INDEX(16)]) + 1.0, 2) - y[INDEX(0)]*h2_optical_depth_approx*rlocal_cooling_rates[INDEX(17)]/(local_cooling_rates[INDEX(17)]/(y[INDEX(0)]*local_cooling_rates[INDEX(13)] + y[INDEX(2)]*local_cooling_rates[INDEX(12)] + y[INDEX(3)]*local_cooling_rates[INDEX(15)] + y[INDEX(5)]*local_cooling_rates[INDEX(14)] + y[INDEX(8)]*local_cooling_rates[INDEX(16)]) + 1.0) - 2.01588*y[INDEX(0)]*mdensity*rlocal_cooling_rates[INDEX(25)] - y[INDEX(2)]*y[INDEX(8)]*rlocal_cooling_rates[INDEX(0)] - y[INDEX(2)]*y[INDEX(8)]*rlocal_cooling_rates[INDEX(4)] - y[INDEX(3)]*y[INDEX(8)]*rlocal_cooling_rates[INDEX(7)] - y[INDEX(5)]*y[INDEX(8)]*rlocal_cooling_rates[INDEX(5)] - y[INDEX(6)]*pow(y[INDEX(8)], 2)*rlocal_cooling_rates[INDEX(1)] - y[INDEX(6)]*pow(y[INDEX(8)], 2)*rlocal_cooling_rates[INDEX(3)] - y[INDEX(6)]*y[INDEX(8)]*rlocal_cooling_rates[INDEX(2)] - y[INDEX(6)]*y[INDEX(8)]*rlocal_cooling_rates[INDEX(6)] - y[INDEX(6)]*y[INDEX(8)]*rlocal_cooling_rates[INDEX(8)] - y[INDEX(6)]*y[INDEX(8)]*rlocal_cooling_rates[INDEX(9)] - y[INDEX(7)]*y[INDEX(8)]*rlocal_cooling_rates[INDEX(10)] - local_cooling_rates[INDEX(18)]*y[INDEX(8)] - y[INDEX(8)]*rlocal_cooling_rates[INDEX(11)]*(y[INDEX(3)] + y[INDEX(6)] + 4.0*y[INDEX(7)]) - y[INDEX(8)]*rlocal_cooling_rates[INDEX(18)]*(T_local - 2.73) + 0.5*pow(local_cooling_rates[INDEX(22)]/(y[INDEX(0)]*local_cooling_rates[INDEX(24)] + y[INDEX(2)]*local_cooling_rates[INDEX(23)]) + 1.0, -2.0)*(-y[INDEX(0)]*y[INDEX(2)]*local_cooling_rates[INDEX(21)] + pow(y[INDEX(2)], 3)*local_cooling_rates[INDEX(20)])*(-1.0*local_cooling_rates[INDEX(22)]*(-y[INDEX(0)]*rlocal_cooling_rates[INDEX(24)] - y[INDEX(2)]*rlocal_cooling_rates[INDEX(23)])/pow(y[INDEX(0)]*local_cooling_rates[INDEX(24)] + y[INDEX(2)]*local_cooling_rates[INDEX(23)], 2) - 1.0*rlocal_cooling_rates[INDEX(22)]/(y[INDEX(0)]*local_cooling_rates[INDEX(24)] + y[INDEX(2)]*local_cooling_rates[INDEX(23)])) + 0.5*1.0/(local_cooling_rates[INDEX(22)]/(y[INDEX(0)]*local_cooling_rates[INDEX(24)] + y[INDEX(2)]*local_cooling_rates[INDEX(23)]) + 1.0)*(-y[INDEX(0)]*y[INDEX(2)]*rlocal_cooling_rates[INDEX(21)] + pow(y[INDEX(2)], 3)*rlocal_cooling_rates[INDEX(20)]);

  jac[INDEX(99)] *= inv_mdensity;
  jac[INDEX(99)] *= Tge;


#ifdef SCALE_INPUT
  jac[INDEX(0)] *= inv_scale[INDEX(0)] * scale[INDEX(0)];
  jac[INDEX(1)] *= inv_scale[INDEX(1)] * scale[INDEX(0)];
  jac[INDEX(2)] *= inv_scale[INDEX(2)] * scale[INDEX(0)];
  jac[INDEX(3)] *= inv_scale[INDEX(3)] * scale[INDEX(0)];
  jac[INDEX(9)] *= inv_scale[INDEX(9)] * scale[INDEX(0)];
  jac[INDEX(10)] *= inv_scale[INDEX(0)] * scale[INDEX(1)];
  jac[INDEX(11)] *= inv_scale[INDEX(1)] * scale[INDEX(1)];
  jac[INDEX(12)] *= inv_scale[INDEX(2)] * scale[INDEX(1)];
  jac[INDEX(13)] *= inv_scale[INDEX(3)] * scale[INDEX(1)];
  jac[INDEX(14)] *= inv_scale[INDEX(4)] * scale[INDEX(1)];
  jac[INDEX(18)] *= inv_scale[INDEX(8)] * scale[INDEX(1)];
  jac[INDEX(20)] *= inv_scale[INDEX(0)] * scale[INDEX(2)];
  jac[INDEX(21)] *= inv_scale[INDEX(1)] * scale[INDEX(2)];
  jac[INDEX(22)] *= inv_scale[INDEX(2)] * scale[INDEX(2)];
  jac[INDEX(23)] *= inv_scale[INDEX(3)] * scale[INDEX(2)];
  jac[INDEX(24)] *= inv_scale[INDEX(4)] * scale[INDEX(2)];
  jac[INDEX(28)] *= inv_scale[INDEX(8)] * scale[INDEX(2)];
  jac[INDEX(29)] *= inv_scale[INDEX(9)] * scale[INDEX(2)];
  jac[INDEX(30)] *= inv_scale[INDEX(0)] * scale[INDEX(3)];
  jac[INDEX(31)] *= inv_scale[INDEX(1)] * scale[INDEX(3)];
  jac[INDEX(32)] *= inv_scale[INDEX(2)] * scale[INDEX(3)];
  jac[INDEX(33)] *= inv_scale[INDEX(3)] * scale[INDEX(3)];
  jac[INDEX(34)] *= inv_scale[INDEX(4)] * scale[INDEX(3)];
  jac[INDEX(38)] *= inv_scale[INDEX(8)] * scale[INDEX(3)];
  jac[INDEX(39)] *= inv_scale[INDEX(9)] * scale[INDEX(3)];
  jac[INDEX(40)] *= inv_scale[INDEX(0)] * scale[INDEX(4)];
  jac[INDEX(41)] *= inv_scale[INDEX(1)] * scale[INDEX(4)];
  jac[INDEX(42)] *= inv_scale[INDEX(2)] * scale[INDEX(4)];
  jac[INDEX(43)] *= inv_scale[INDEX(3)] * scale[INDEX(4)];
  jac[INDEX(44)] *= inv_scale[INDEX(4)] * scale[INDEX(4)];
  jac[INDEX(48)] *= inv_scale[INDEX(8)] * scale[INDEX(4)];
  jac[INDEX(55)] *= inv_scale[INDEX(5)] * scale[INDEX(5)];
  jac[INDEX(56)] *= inv_scale[INDEX(6)] * scale[INDEX(5)];
  jac[INDEX(58)] *= inv_scale[INDEX(8)] * scale[INDEX(5)];
  jac[INDEX(59)] *= inv_scale[INDEX(9)] * scale[INDEX(5)];
  jac[INDEX(65)] *= inv_scale[INDEX(5)] * scale[INDEX(6)];
  jac[INDEX(66)] *= inv_scale[INDEX(6)] * scale[INDEX(6)];
  jac[INDEX(67)] *= inv_scale[INDEX(7)] * scale[INDEX(6)];
  jac[INDEX(68)] *= inv_scale[INDEX(8)] * scale[INDEX(6)];
  jac[INDEX(69)] *= inv_scale[INDEX(9)] * scale[INDEX(6)];
  jac[INDEX(76)] *= inv_scale[INDEX(6)] * scale[INDEX(7)];
  jac[INDEX(77)] *= inv_scale[INDEX(7)] * scale[INDEX(7)];
  jac[INDEX(78)] *= inv_scale[INDEX(8)] * scale[INDEX(7)];
  jac[INDEX(79)] *= inv_scale[INDEX(9)] * scale[INDEX(7)];
  jac[INDEX(80)] *= inv_scale[INDEX(0)] * scale[INDEX(8)];
  jac[INDEX(81)] *= inv_scale[INDEX(1)] * scale[INDEX(8)];
  jac[INDEX(82)] *= inv_scale[INDEX(2)] * scale[INDEX(8)];
  jac[INDEX(83)] *= inv_scale[INDEX(3)] * scale[INDEX(8)];
  jac[INDEX(84)] *= inv_scale[INDEX(4)] * scale[INDEX(8)];
  jac[INDEX(85)] *= inv_scale[INDEX(5)] * scale[INDEX(8)];
  jac[INDEX(86)] *= inv_scale[INDEX(6)] * scale[INDEX(8)];
  jac[INDEX(87)] *= inv_scale[INDEX(7)] * scale[INDEX(8)];
  jac[INDEX(88)] *= inv_scale[INDEX(8)] * scale[INDEX(8)];
  jac[INDEX(89)] *= inv_scale[INDEX(9)] * scale[INDEX(8)];
  jac[INDEX(90)] *= inv_scale[INDEX(0)] * scale[INDEX(9)];
  jac[INDEX(91)] *= inv_scale[INDEX(1)] * scale[INDEX(9)];
  jac[INDEX(92)] *= inv_scale[INDEX(2)] * scale[INDEX(9)];
  jac[INDEX(93)] *= inv_scale[INDEX(3)] * scale[INDEX(9)];
  jac[INDEX(94)] *= inv_scale[INDEX(4)] * scale[INDEX(9)];
  jac[INDEX(95)] *= inv_scale[INDEX(5)] * scale[INDEX(9)];
  jac[INDEX(96)] *= inv_scale[INDEX(6)] * scale[INDEX(9)];
  jac[INDEX(97)] *= inv_scale[INDEX(7)] * scale[INDEX(9)];
  jac[INDEX(98)] *= inv_scale[INDEX(8)] * scale[INDEX(9)];
  jac[INDEX(99)] *= inv_scale[INDEX(9)] * scale[INDEX(9)];
#endif




/*
  if (T_ID == 0){
//  printf("at time = %0.5g, temp = %0.5g\n",t, *T_local);

  for (int i = 0; i<10; i++){
    printf("y[INDEX(%d)] = %0.5g\n", i, y[INDEX(i)]);
  }
  }
*/

/*
  if (T_ID == 0){
  printf("density: %0.5g\n", mdensity);
  printf("T_local: %0.5g\n", T_local);
  printf("Tge    : %0.5g\n", Tge);


  for (int i = 0; i<100; i++){
    printf("jac[INDEX(%d)] = %0.5g\n", i, jac[INDEX(i)]);
  }
  }
*/
//  }

} // end eval_jacob

