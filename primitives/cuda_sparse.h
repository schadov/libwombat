#pragma once

#define THREAD_BLOCK_SIZE 16

template<class RealT> struct GPU_matrix{

};

template<class RealT> struct CRS_matrix_cuda:public GPU_matrix<RealT>{
	RealT* a;
	unsigned int* colind;
	uint2* rowptr;
	unsigned int nrows;
	unsigned int nelements;
};

/*
template<class RealT> struct Dense_matrix_cuda:public GPU_matrix<RealT>{
	RealT* a;
	unsigned int nrows;
	unsigned int nelements;
};*/


void spmv_csr_float(CRS_matrix_cuda<float> A,
					float* x, float* b, 
					unsigned int thread_block_sz = THREAD_BLOCK_SIZE);

void cuda_memberwise_mul_float(unsigned int N, float* x, float* y,float* z,unsigned int thread_block_sz=THREAD_BLOCK_SIZE);



void chebyshev_iterations(CRS_matrix_cuda<float> A,float* x, float* b,
						  float* r, float*z,float* p,
						  float* diag_inv,int max_iter,
						  int thread_block_sz = THREAD_BLOCK_SIZE
						  );

void chebyshev_iteration(CRS_matrix_cuda<float> A,float* x, float* b,
						  float* r, float*z,float* p,
						  float* diag_inv,int curr_iter,
						  int thread_block_sz = THREAD_BLOCK_SIZE
						  );

void chebyshev_iteration_s(CRS_matrix_cuda<float> A,float* x, float* b,
						 float* r, float*z,float* p,
						 float* diag_inv,int curr_iter, int its,
						 int thread_block_sz = THREAD_BLOCK_SIZE
						 );


void axmb_csr_float(CRS_matrix_cuda<float> A,
					float* x, float* b, float* r,
					unsigned int thread_block_sz= THREAD_BLOCK_SIZE );


//------ dense
void spmv_dense_float(float *A, float * x, float* y, int size,unsigned int thread_block_sz);