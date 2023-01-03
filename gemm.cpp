#include <cstdio>
#include <cstdlib>
#include <immintrin.h>

// gemm -- general double precision dense matrix-matrix multiplication.
//
// implement: C = alpha * A x B + beta * C, for matrices A, B, C
// Matrix C is M x N  (M rows, N columns)
// Matrix A is M x K
// Matrix B is K x N

void gemm(int m, int n, int k, double *A, double *B, double *C, double alpha, double beta){
    int i, j;
    double *B_transpose = (double*) malloc(k*n*sizeof(double));// B_transpose is N x K
    //#pragma omp parallel for
    for (int nn=0; nn<n; nn++){
        for (int kk=0; kk<k; kk++){
            B_transpose[nn*k+kk] = B[kk*n+nn];
        }
    }
    #pragma omp parallel for
    for (i=0; i<m; i++){
        for (j=0; j<n; j++){
            double inner_prod = 0;   
            __m256d sum_vec = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
            for(int kk = 0; kk < k/4; ++kk) {
                __m256d a = _mm256_load_pd((A+i*k)+4*kk);
                __m256d b = _mm256_load_pd((B+j*k)+4*kk);
                __m256d ab = _mm256_mul_pd(a, b);
                sum_vec = _mm256_add_pd(sum_vec, ab);
                }
            __m256d temp = _mm256_hadd_pd(sum_vec, sum_vec);
            inner_prod = ((double*)&temp)[0] + ((double*)&temp)[2];
            C[i*n+j] = alpha * inner_prod + beta * C[i*n+j];
        }
    }
    free(B_transpose);
    
    // BEGINNING of NAIVE IMPLEMENTATION
    
    //int i, j, kk;
    //for (i=0; i<m; i++){
    //    for (j=0; j<n; j++){
	//    double inner_prod = 0;
	//    for (kk=0; kk<k; kk++){
	//        inner_prod += A[i*k+kk] * B[kk*n+j];
	//    }
	//    C[i*n+j] = alpha * inner_prod + beta * C[i*n+j];
	//}
    //}
    
    // END OF NAIVE IMPLEMENTATION
}

