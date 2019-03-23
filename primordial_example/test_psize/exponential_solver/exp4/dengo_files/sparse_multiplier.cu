#include "sparse_multiplier.cuh"

__device__
void sparse_multiplier(const double * A, const double * Vm, double* w) {

  // df_H2_1 / H2_1:
  // w[H2_1] =  df_H2_1 / H2_1: * H2_1;
  w[INDEX(0)] = A[INDEX(0)] * Vm[INDEX(0)];
  // df_H2_2 / H2_1:
  // w[H2_2] =  df_H2_2 / H2_1: * H2_1;
  w[INDEX(1)] = A[INDEX(1)] * Vm[INDEX(0)];
  // df_H_1 / H2_1:
  // w[H_1] =  df_H_1 / H2_1: * H2_1;
  w[INDEX(2)] = A[INDEX(2)] * Vm[INDEX(0)];
  // df_H_2 / H2_1:
  // w[H_2] =  df_H_2 / H2_1: * H2_1;
  w[INDEX(3)] = A[INDEX(3)] * Vm[INDEX(0)];
  // df_ge / H2_1:
  // w[ge] =  df_ge / H2_1: * H2_1;
  w[INDEX(9)] = A[INDEX(9)] * Vm[INDEX(0)];
  // df_H2_1 / H2_2:
  // w[H2_1] +=  df_H2_1 / H2_2: * H2_2;
  w[INDEX(0)] += A[INDEX(10)] * Vm[INDEX(1)];
  // df_H2_2 / H2_2:
  // w[H2_2] +=  df_H2_2 / H2_2: * H2_2;
  w[INDEX(1)] += A[INDEX(11)] * Vm[INDEX(1)];
  // df_H_1 / H2_2:
  // w[H_1] +=  df_H_1 / H2_2: * H2_2;
  w[INDEX(2)] += A[INDEX(12)] * Vm[INDEX(1)];
  // df_H_2 / H2_2:
  // w[H_2] +=  df_H_2 / H2_2: * H2_2;
  w[INDEX(3)] += A[INDEX(13)] * Vm[INDEX(1)];
  // df_H_m0 / H2_2:
  // w[H_m0] +=  df_H_m0 / H2_2: * H2_2;
  w[INDEX(4)] += A[INDEX(14)] * Vm[INDEX(1)];
  // df_de / H2_2:
  // w[de] +=  df_de / H2_2: * H2_2;
  w[INDEX(8)] += A[INDEX(18)] * Vm[INDEX(1)];
  // df_H2_1 / H_1:
  // w[H2_1] +=  df_H2_1 / H_1: * H_1;
  w[INDEX(0)] += A[INDEX(20)] * Vm[INDEX(2)];
  // df_H2_2 / H_1:
  // w[H2_2] +=  df_H2_2 / H_1: * H_1;
  w[INDEX(1)] += A[INDEX(21)] * Vm[INDEX(2)];
  // df_H_1 / H_1:
  // w[H_1] +=  df_H_1 / H_1: * H_1;
  w[INDEX(2)] += A[INDEX(22)] * Vm[INDEX(2)];
  // df_H_2 / H_1:
  // w[H_2] +=  df_H_2 / H_1: * H_1;
  w[INDEX(3)] += A[INDEX(23)] * Vm[INDEX(2)];
  // df_H_m0 / H_1:
  // w[H_m0] +=  df_H_m0 / H_1: * H_1;
  w[INDEX(4)] += A[INDEX(24)] * Vm[INDEX(2)];
  // df_de / H_1:
  // w[de] +=  df_de / H_1: * H_1;
  w[INDEX(8)] += A[INDEX(28)] * Vm[INDEX(2)];
  // df_ge / H_1:
  // w[ge] +=  df_ge / H_1: * H_1;
  w[INDEX(9)] += A[INDEX(29)] * Vm[INDEX(2)];
  // df_H2_1 / H_2:
  // w[H2_1] +=  df_H2_1 / H_2: * H_2;
  w[INDEX(0)] += A[INDEX(30)] * Vm[INDEX(3)];
  // df_H2_2 / H_2:
  // w[H2_2] +=  df_H2_2 / H_2: * H_2;
  w[INDEX(1)] += A[INDEX(31)] * Vm[INDEX(3)];
  // df_H_1 / H_2:
  // w[H_1] +=  df_H_1 / H_2: * H_2;
  w[INDEX(2)] += A[INDEX(32)] * Vm[INDEX(3)];
  // df_H_2 / H_2:
  // w[H_2] +=  df_H_2 / H_2: * H_2;
  w[INDEX(3)] += A[INDEX(33)] * Vm[INDEX(3)];
  // df_H_m0 / H_2:
  // w[H_m0] +=  df_H_m0 / H_2: * H_2;
  w[INDEX(4)] += A[INDEX(34)] * Vm[INDEX(3)];
  // df_de / H_2:
  // w[de] +=  df_de / H_2: * H_2;
  w[INDEX(8)] += A[INDEX(38)] * Vm[INDEX(3)];
  // df_ge / H_2:
  // w[ge] +=  df_ge / H_2: * H_2;
  w[INDEX(9)] += A[INDEX(39)] * Vm[INDEX(3)];
  // df_H2_1 / H_m0:
  // w[H2_1] +=  df_H2_1 / H_m0: * H_m0;
  w[INDEX(0)] += A[INDEX(40)] * Vm[INDEX(4)];
  // df_H2_2 / H_m0:
  // w[H2_2] +=  df_H2_2 / H_m0: * H_m0;
  w[INDEX(1)] += A[INDEX(41)] * Vm[INDEX(4)];
  // df_H_1 / H_m0:
  // w[H_1] +=  df_H_1 / H_m0: * H_m0;
  w[INDEX(2)] += A[INDEX(42)] * Vm[INDEX(4)];
  // df_H_2 / H_m0:
  // w[H_2] +=  df_H_2 / H_m0: * H_m0;
  w[INDEX(3)] += A[INDEX(43)] * Vm[INDEX(4)];
  // df_H_m0 / H_m0:
  // w[H_m0] +=  df_H_m0 / H_m0: * H_m0;
  w[INDEX(4)] += A[INDEX(44)] * Vm[INDEX(4)];
  // df_de / H_m0:
  // w[de] +=  df_de / H_m0: * H_m0;
  w[INDEX(8)] += A[INDEX(48)] * Vm[INDEX(4)];
  // df_He_1 / He_1:
  // w[He_1] +=  df_He_1 / He_1: * He_1;
  w[INDEX(5)] += A[INDEX(55)] * Vm[INDEX(5)];
  // df_He_2 / He_1:
  // w[He_2] +=  df_He_2 / He_1: * He_1;
  w[INDEX(6)] += A[INDEX(56)] * Vm[INDEX(5)];
  // df_de / He_1:
  // w[de] +=  df_de / He_1: * He_1;
  w[INDEX(8)] += A[INDEX(58)] * Vm[INDEX(5)];
  // df_ge / He_1:
  // w[ge] +=  df_ge / He_1: * He_1;
  w[INDEX(9)] += A[INDEX(59)] * Vm[INDEX(5)];
  // df_He_1 / He_2:
  // w[He_1] +=  df_He_1 / He_2: * He_2;
  w[INDEX(5)] += A[INDEX(65)] * Vm[INDEX(6)];
  // df_He_2 / He_2:
  // w[He_2] +=  df_He_2 / He_2: * He_2;
  w[INDEX(6)] += A[INDEX(66)] * Vm[INDEX(6)];
  // df_He_3 / He_2:
  // w[He_3] +=  df_He_3 / He_2: * He_2;
  w[INDEX(7)] += A[INDEX(67)] * Vm[INDEX(6)];
  // df_de / He_2:
  // w[de] +=  df_de / He_2: * He_2;
  w[INDEX(8)] += A[INDEX(68)] * Vm[INDEX(6)];
  // df_ge / He_2:
  // w[ge] +=  df_ge / He_2: * He_2;
  w[INDEX(9)] += A[INDEX(69)] * Vm[INDEX(6)];
  // df_He_2 / He_3:
  // w[He_2] +=  df_He_2 / He_3: * He_3;
  w[INDEX(6)] += A[INDEX(76)] * Vm[INDEX(7)];
  // df_He_3 / He_3:
  // w[He_3] +=  df_He_3 / He_3: * He_3;
  w[INDEX(7)] += A[INDEX(77)] * Vm[INDEX(7)];
  // df_de / He_3:
  // w[de] +=  df_de / He_3: * He_3;
  w[INDEX(8)] += A[INDEX(78)] * Vm[INDEX(7)];
  // df_ge / He_3:
  // w[ge] +=  df_ge / He_3: * He_3;
  w[INDEX(9)] += A[INDEX(79)] * Vm[INDEX(7)];
  // df_H2_1 / de:
  // w[H2_1] +=  df_H2_1 / de: * de;
  w[INDEX(0)] += A[INDEX(80)] * Vm[INDEX(8)];
  // df_H2_2 / de:
  // w[H2_2] +=  df_H2_2 / de: * de;
  w[INDEX(1)] += A[INDEX(81)] * Vm[INDEX(8)];
  // df_H_1 / de:
  // w[H_1] +=  df_H_1 / de: * de;
  w[INDEX(2)] += A[INDEX(82)] * Vm[INDEX(8)];
  // df_H_2 / de:
  // w[H_2] +=  df_H_2 / de: * de;
  w[INDEX(3)] += A[INDEX(83)] * Vm[INDEX(8)];
  // df_H_m0 / de:
  // w[H_m0] +=  df_H_m0 / de: * de;
  w[INDEX(4)] += A[INDEX(84)] * Vm[INDEX(8)];
  // df_He_1 / de:
  // w[He_1] +=  df_He_1 / de: * de;
  w[INDEX(5)] += A[INDEX(85)] * Vm[INDEX(8)];
  // df_He_2 / de:
  // w[He_2] +=  df_He_2 / de: * de;
  w[INDEX(6)] += A[INDEX(86)] * Vm[INDEX(8)];
  // df_He_3 / de:
  // w[He_3] +=  df_He_3 / de: * de;
  w[INDEX(7)] += A[INDEX(87)] * Vm[INDEX(8)];
  // df_de / de:
  // w[de] +=  df_de / de: * de;
  w[INDEX(8)] += A[INDEX(88)] * Vm[INDEX(8)];
  // df_ge / de:
  // w[ge] +=  df_ge / de: * de;
  w[INDEX(9)] += A[INDEX(89)] * Vm[INDEX(8)];
  // df_H2_1 / ge:
  // w[H2_1] +=  df_H2_1 / ge: * ge;
  w[INDEX(0)] += A[INDEX(90)] * Vm[INDEX(9)];
  // df_H2_2 / ge:
  // w[H2_2] +=  df_H2_2 / ge: * ge;
  w[INDEX(1)] += A[INDEX(91)] * Vm[INDEX(9)];
  // df_H_1 / ge:
  // w[H_1] +=  df_H_1 / ge: * ge;
  w[INDEX(2)] += A[INDEX(92)] * Vm[INDEX(9)];
  // df_H_2 / ge:
  // w[H_2] +=  df_H_2 / ge: * ge;
  w[INDEX(3)] += A[INDEX(93)] * Vm[INDEX(9)];
  // df_H_m0 / ge:
  // w[H_m0] +=  df_H_m0 / ge: * ge;
  w[INDEX(4)] += A[INDEX(94)] * Vm[INDEX(9)];
  // df_He_1 / ge:
  // w[He_1] +=  df_He_1 / ge: * ge;
  w[INDEX(5)] += A[INDEX(95)] * Vm[INDEX(9)];
  // df_He_2 / ge:
  // w[He_2] +=  df_He_2 / ge: * ge;
  w[INDEX(6)] += A[INDEX(96)] * Vm[INDEX(9)];
  // df_He_3 / ge:
  // w[He_3] +=  df_He_3 / ge: * ge;
  w[INDEX(7)] += A[INDEX(97)] * Vm[INDEX(9)];
  // df_de / ge:
  // w[de] +=  df_de / ge: * ge;
  w[INDEX(8)] += A[INDEX(98)] * Vm[INDEX(9)];
  // df_ge / ge:
  // w[ge] +=  df_ge / ge: * ge;
  w[INDEX(9)] += A[INDEX(99)] * Vm[INDEX(9)];

}
