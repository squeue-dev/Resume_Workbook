/* Code example for the evaluation of quantum Chemical Electron Repulsion 
   Integrals solved analytically through Obara and Saika's Recurrence eqs.
   
   Martin Head‐Gordon and John A. Pople, A method for two‐electron Gaussian
   integral and integral derivative evaluation using recurrence relations, 
   J. Chem. Phys. 89, 5777-5786 (1988) https://doi.org/10.1063/1.455553

Author: Alfonso Esqueda García, esqueda.alfonso.94@gmail.com
Year: 2018
*/  




#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <ctime>

//Linear algebra libraries
#include "cublas_v2.h"
#include "magma_v2.h"
#include "magma_lapack.h"

#define BOYS_MIN 1.0e-13
#define GALLETA_MAX_L_I 8
#define GALLETA_MAX_TAB_GAM 120
#define GALLETA_ABS(a) ((a)>=0.0?(a):-(a))
#define GALLETA_PI M_PI

using namespace std;
using std::cout;
using std::endl;


__device__ double F0(double t){

  if (t < 1.0E-6){
    //ASYMPTOTIC VALUE FOR SMALL ARGUMENTS
    return 1.0 - t / 3.0;
  }
  else{
    //F0 IN TERMS OF THE ERROR FUNCTION
    return 0.5*pow(3.1416/t,1.0/2.0)*erf(pow(t,1.0/2.0));
  }
}

// AEG: Function to contract the primitive integrals from within the GPU for -#
//      any contracted integral. ---------------------------------------------#
// Note: In the middle of creating a function to contract each type of eri ---#
//       I noticed there was a way to create a general function to do that ---#
//       regardless of the type of eri to cotract ----------------------------#
__global__ void primi2contr(int *shellidxs_ll_a, int * shellidxs_ul_a,
                            int *shellidxs_ll_b, int * shellidxs_ul_b,
                            double *primeri, double *contreri,
                            double *coef_a, double *coef_b, int nshells_a,
                            int nshells_b, int nprimis, int auxnco,
                            int ori1, int ori2, int ori3, double *ncsto_a, double *ncsto_b, double *ncsto_aux)
{
  for (int i=blockIdx.x * blockDim.x + threadIdx.x; i<nshells_a; i+=blockDim.x * gridDim.x) {
    for (int j=blockIdx.y * blockDim.y + threadIdx.y; j<nshells_b; j+=blockDim.y * gridDim.y) {
      for (int r=blockIdx.z * blockDim.z + threadIdx.z; r<auxnco; r+=blockDim.z * gridDim.z) {

        int nori = ori1*ori2*ori3;

        int lli = shellidxs_ll_a[i];
        int uli = shellidxs_ul_a[i];

        int llj = shellidxs_ll_b[j];
        int ulj = shellidxs_ul_b[j];

        for (int ctr_aux=0; ctr_aux<ori3; ctr_aux++) {
          for (int ctr_a=0; ctr_a<ori1; ctr_a++) {
            for (int ctr_b=0; ctr_b<ori2; ctr_b++) {
              double sum = 0.0;
              for (int k=lli; k<=uli; k+=ori1) {
                for (int l=llj; l<=ulj; l+=ori2) {
                  sum += primeri[(k+ctr_a)*nprimis*ori2*auxnco*ori3+(l+ctr_b)*auxnco*ori3+r*ori3+ctr_aux] *
                         coef_a[k/ori1] * coef_b[l/ori2];
                }
              }
              contreri[(i*ori1+ctr_a)*nshells_b*ori2*auxnco*ori3+(j*ori2+ctr_b)*auxnco*ori3+(r*ori3+ctr_aux)] = sum*ncsto_a[i*ori1+ctr_a]*ncsto_b[j*ori2+ctr_b]*ncsto_aux[ori3*r+ctr_aux];
            }
          }
        }

      }
    }
  }
}
// ---------------------------------------------------------------------------#
// ---------------------------------------------------------------------------#


// AEG: Function to add the contributions of the contracted eris to the Fock -#
//      matrix. --------------------------------------------------------------#
__global__ void contr2FockM(double *contreri, double *FockM, int nshells,
                            int n_shells_a, int n_shells_b, int n_aux, double *w, double sf, int *ll_ao_shell_a, int *ll_ao_shell_b, int nbas, int ori1, int ori2, int ori3)
{
  double sum;
  double factor;
  int ii, jj;
  //Falta considerar las orientaciones, o quiza si estan, hay que ver como se calcula nshells
  for (int i=blockIdx.x * blockDim.x + threadIdx.x; i<n_shells_a; i+=blockDim.x * gridDim.x) {
    ii = ll_ao_shell_a[i];
    for (int j=blockIdx.y * blockDim.y + threadIdx.y; j<n_shells_b; j+=blockDim.y * gridDim.y) {
      jj = ll_ao_shell_b[j];
        for (int iori1=0; iori1<ori1; iori1++) {
          int iii = ii + iori1;
          for (int jori2=0; jori2<ori2; jori2++) {
            int jjj = jj + jori2;
            sum = 0.0;
            for (int k=0; k<n_aux; k++) {
              for (int kori3=0; kori3<ori3; kori3++) {
                int nori = ori1*ori2*ori3;
                factor = w[k*ori3+kori3] * sf;
                sum += factor * contreri[(i*ori1+iori1)*n_shells_b*ori2*n_aux*ori3+(j*ori2+jori2)*n_aux*ori3+(k*ori3+kori3)];
            }
          }
          FockM[iii*nbas+jjj] += sum;
        }
      }
    }
  }
}
// ---------------------------------------------------------------------------#
// ---------------------------------------------------------------------------#



// AEG: Computes the integral (s|s) needed for the normalization of auxiliary-#
// basis.---------------------------------------------------------------------#
__device__ void ss(double *a,double *b, double exp_a, double exp_b,
                   double coef_a, double coef_b, int la, int lb,
                   double &eri) {
  //From Szabo Apendix A p 410 y 416

  double Rab2, coef, f0;

  // Compute ERI
  coef = 2*pow(3.1416,5.0/2.0)/(exp_a*exp_b*pow(exp_a+exp_b,1.0/2.0));

  Rab2 = pow(a[0]-b[0],2) + pow(a[1]-b[1],2) + pow(a[2]-b[2],2);

  f0 = F0(exp_a*exp_b/(exp_a+exp_b)*Rab2);

  eri = coef*f0;
}
// ---------------------------------------------------------------------------#
// ---------------------------------------------------------------------------#



// AEG: Factorial function needed for the Evaluation of the Boys function ----#
//      or "Gamma Function" routien ----------------------------------- ------#
__device__ int factorial(int n) {
    if(n > 1)
        return n * factorial(n - 1);
    else
        return 1;
}
// ---------------------------------------------------------------------------#
// ---------------------------------------------------------------------------#



// AEG: Initialization of the tabulated values needed for the evaluation of --#
//      the Boys function ----------------------------------------------------#
//
// *This function can be device, global or called by host depending of needs
//  with just a little modification
//
// *ftab needs to be of dimensions:
//  ftab[2*GALLETA_MAX_L_I+6+1][GALLETA_MAX_TAB_GAM+1] -----------------------#
__device__ void init_boysfunc(double ftab[][GALLETA_MAX_TAB_GAM+1]) {
  int i, j, k, l;
  const int nitermax = 30; // needs to be a constant for the nvcc compiler to
                           // accept it as the dimension for r[]
  int nmax = 2 * GALLETA_MAX_L_I + 6;
  double eps = 1.0e-15;
  double bessel, expterm, prefak, preterm, produkt, serie, sumterm, term, ttab;
  double r[nitermax+11];

  for (i=0; i<=nmax; i++)
    ftab[i][0] = 1.0/(2*i+1);
  for (i=1; i<=GALLETA_MAX_TAB_GAM; i++) {
    ttab = double(i)/10.0;
    r[nitermax+10] = 0.0;
    for (j=1; j<=nitermax+9; j++)
      r[nitermax+10-j] = -ttab/(4*(nitermax+10-j) + 2.0 - ttab *  r[nitermax+11-j]);
    bessel = (2 * sinh(ttab / 2)) / ttab;
    prefak = exp(-ttab / 2) * bessel;
    term = 1.0;
    serie = prefak * (1.0 / (2.0 * nmax + 1.0));
    for (k=1; k<=nitermax; k++) {
      preterm = (2.0 * k + 1.0) / (2.0 * nmax + 1.0);
      term = term * (2.0 * nmax - 2.0 * k + 1.0) / (2.0 * nmax + 2.0 * k + 1.0);
      produkt = 1.0;
      for (l=1; l<=k; l++)
        produkt = produkt * r[l];
      sumterm = prefak * preterm * term * produkt;
      if (GALLETA_ABS(sumterm)<=eps) goto TABOK;
      else serie = serie + sumterm;
    }
TABOK:
    ftab[nmax][i] = serie;
    expterm = exp(-ttab);
    for (j=1; j<=nmax; j++)
      ftab[nmax-j][i] = 1.0 / (2 * (nmax - j) + 1) * (2 * ttab * ftab[nmax+1-j][i] + expterm);
  }
}
// ---------------------------------------------------------------------------#
// ---------------------------------------------------------------------------#



// AEG: Calculation of the Boys function F(t) for bielectronic integrals over-#
// Gaussian functions. -------------------------------------------------------#
//
// Taken from Green.128.
// Originally taken from:
// L. E. McMurchie and E. R. Davidson,
// J. Comp. Phys. 26, 218 (1978).
// ---------------------------------------------------------------------------#
// ---------------------------------------------------------------------------#
__device__ void boysfunc(int m, double& t, double* fis,// vector with F(i)T
                          double ftab[][GALLETA_MAX_TAB_GAM+1]) {
  int i, k, ttab;
  double a, b, c, d, expterm;

  if (t < 0.0) t = BOYS_MIN;
  if (t <= BOYS_MIN) {
    fis[m] = 1.0/ (2.0 * m + 1.0);
    for (i=1; i<=m; i++)
      fis[m-i] = 1.0/(2.0 * (m-i) + 1.0);
    return;
  }
  else if (t <= 12.0) {
    ttab = int(10 * t + 0.5);
    fis[m] = ftab[m][ttab];
    for (k=1; k<=6; k++)
      fis[m] += ftab[m+k][ttab] * (pow(double(ttab) / 10.0 - t, k)) / factorial(k);
    if (m > 0) expterm = exp(-t);
    for (i=1; i<=m; i++)
      fis[m-i] = 1.0 / (2 * (m - i) + 1) * (2 * t * fis[m+1-i] + expterm);
    return;
  }
  else if (t <= 15.0) {
    a = 0.4999489092;
    b = 0.2473631686;
    c = 0.3211809090;
    d = 0.3811559346;
    fis[0] = 0.5 * sqrt(GALLETA_PI / t) - (exp(-t) / t) * (a - b / t + c / (t * t) - d / (t * t * t));
  }
  else if (t <= 18.0) {
    a = 0.4998436875;
    b = 0.2424943800;
    c = 0.2464284500;
    fis[0] = 0.5 * sqrt(GALLETA_PI / t) - (exp(-t) / t) * (a - b / t + c / (t * t));
  }
  else if (t <= 24.0) {
    a = 0.4990931620;
    b = 0.2152832000;
    fis[0] = 0.5 * sqrt(GALLETA_PI / t) - (exp(-t) / t) * (a - b / t);
  }
  else if (t <= 30.0) {
    a = 0.49000000;
    fis[0] = 0.5 * sqrt(GALLETA_PI / t) - (exp(-t) / t) * a;
  }
  else {
    fis[0] = 0.5 * sqrt(GALLETA_PI / t);
  }
  if (t > (2.0 * m + 36)) {
    for (i=1; i<=m; i++)
      fis[i] = (2 * i - 1) / (2 * t) * fis[i-1];
  }
  else {
    expterm = exp(-t);
    for (i=1; i<=m; i++)
      fis[i] = 1 / (2 * t) * ((2 * i -1) * fis[i-1] - expterm);
  }
}
// ---------------------------------------------------------------------------#
// ---------------------------------------------------------------------------#



// AEG: Function to compute a three center integral of the type (ss|p) -------#
//      using the formula obtained from aplying the OS RR method. ------------#
__global__ void ssp(double *a, double *b, double *caux, double *exp_a,
                    double *exp_b, double *exp_caux, double *coef_a,
                    double *coef_b, double *coef_caux, int la, int lb,
                    int lcaux, int nprimis, int auxnco, double *primeri,
                    int *atomidx, int *atomidxaux) {

  double Rab2, Rpcaux2, Kab, p, coef, t, eri;
  double Rp[3], Rw[3], fis[2], eri_v[3];
  double caux4norm[3];
  int atidxa, atidxb, atidxaux;
  double normceri = 0.0;

  __shared__ double ftab[2*GALLETA_MAX_L_I+6+1][GALLETA_MAX_TAB_GAM+1];

  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    // Initialize ftab
    init_boysfunc(ftab);
  }
  __syncthreads();

  // AEG loop for offsetting
  for (int x=blockIdx.x * blockDim.x + threadIdx.x; x<nprimis; x+=blockDim.x * gridDim.x) {
    for (int y=blockIdx.y * blockDim.y + threadIdx.y; y<nprimis; y+=blockDim.y * gridDim.y) {
      for (int z=blockIdx.z * blockDim.z + threadIdx.z; z<auxnco; z+=blockDim.z * gridDim.z) {
        atidxa = atomidx[x];
        atidxb = atomidx[y];
        atidxaux = atomidxaux[z];

        fis[0] = 0.0;
        fis[1] = 0.0;

        // Calculation of [ss||s]^(1)

          // Bra contraction using gaussian product theorem
        Rab2 = pow(a[atidxa*3]-b[atidxb*3],2) + pow(a[atidxa*3+1]-b[atidxb*3+1],2) + pow(a[atidxa*3+2]-b[atidxb*3+2],2);
        Kab = exp(-((exp_a[x]*exp_b[y]/(exp_a[x]+exp_b[y]))*(Rab2)));
        p = exp_a[x] + exp_b[y];
        Rp[0] = (exp_a[x]*a[atidxa*3]+exp_b[y]*b[atidxb*3])/(exp_a[x]+exp_b[y]);
        Rp[1] = (exp_a[x]*a[atidxa*3+1]+exp_b[y]*b[atidxb*3+1])/(exp_a[x]+exp_b[y]);
        Rp[2] = (exp_a[x]*a[atidxa*3+2]+exp_b[y]*b[atidxb*3+2])/(exp_a[x]+exp_b[y]);

          // Compute ERI
        coef = 2*pow(GALLETA_PI,5.0/2.0)/(p*exp_caux[z]*pow(p+exp_caux[z],1.0/2.0));
        Rpcaux2 = pow(Rp[0]-caux[atidxaux*3],2) + pow(Rp[1]-caux[atidxaux*3+1],2) + pow(Rp[2]-caux[atidxaux*3+2],2);
          // Boys function with m=1
        t = ((p * exp_caux[z]) / (p + exp_caux[z])) * Rpcaux2;
        boysfunc(1, t, fis, ftab);

        eri = coef * Kab * fis[1];

        caux4norm[0] = caux[atidxaux*3];
        caux4norm[1] = caux[atidxaux*3+1];
        caux4norm[2] = caux[atidxaux*3+2];

        // Multiply [ss||s]^(1) by (Wi-Ci)
        Rw[0] = (p * Rp[0] + exp_caux[z] * caux[atidxaux*3]) / (p + exp_caux[z]);
        Rw[1] = (p * Rp[1] + exp_caux[z] * caux[atidxaux*3+1]) / (p + exp_caux[z]);
        Rw[2] = (p * Rp[2] + exp_caux[z] * caux[atidxaux*3+2]) / (p + exp_caux[z]);

        eri_v[0] = (Rw[0] - caux[atidxaux*3]) * eri;
        eri_v[1] = (Rw[1] - caux[atidxaux*3+1]) * eri;
        eri_v[2] = (Rw[2] - caux[atidxaux*3+2]) * eri;

        primeri[x*nprimis*3*auxnco+y*3*auxnco+3*z+0] = eri_v[0];
        primeri[x*nprimis*3*auxnco+y*3*auxnco+3*z+1] = eri_v[1];
        primeri[x*nprimis*3*auxnco+y*3*auxnco+3*z+2] = eri_v[2];

      }
    }
  }
}
// ---------------------------------------------------------------------------#
// ---------------------------------------------------------------------------#
