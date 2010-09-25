#include "BlasCommon.h"
#include "cuda_sparse.h"

//global data
texture<float, 1> texXf;

//Helper functions
__device__  unsigned int compute_thread_index () {
	return ( blockIdx.x*blockDim.x*blockDim.y+
		blockIdx.y*blockDim.x*blockDim.y*gridDim.x+
		threadIdx.x+threadIdx.y*blockDim.x) ;
}

void bind_x_texf(float *x,unsigned int N)
{
	cudaBindTexture(0,texXf,x,N*sizeof(float));
}

void  unbind_x_tex()
{
	cudaUnbindTexture(texXf);
}

//kernel functions
__device__ void spmv_csr_scalar_dev (
									  float * matrix, unsigned int size_matrix,
									  uint2 * rowptr, unsigned int size_rowptr,
									  unsigned int * colind, unsigned int size_colind,
									  float * x, float * b, unsigned int size_vec )
{
	const unsigned int index = compute_thread_index () ;
	if ( index < size_vec ) {

		const uint2 rowptr_bounds = rowptr[index] ;
		float res = 0.0f ;

		// for each block of the block_row, mult
		for ( unsigned int i=rowptr_bounds.x; i<rowptr_bounds.y; i++ ) { 
			const float xv = tex1Dfetch(texXf,colind[i]);
			res += matrix[i]*xv ;
		}
		b[index] = res ;
	}
}

__global__ void axmb_csr_krnl (
									  float * matrix, unsigned int size_matrix,
									  uint2 * rowptr, unsigned int size_rowptr,
									  unsigned int * colind, unsigned int size_colind,
									  float * x, float * b, float* r, unsigned int size_vec )
{
	const unsigned int index = compute_thread_index () ;
	if ( index < size_vec ) {

		const uint2 rowptr_bounds = rowptr[index] ;
		float res = 0.0f ;

		// for each block of the block_row, mult
		for ( unsigned int i=rowptr_bounds.x; i<rowptr_bounds.y; i++ ) { 
			const float xv = tex1Dfetch(texXf,colind[i]);
			res += matrix[i]*xv ;
		}
		r[index] = b[index]  -res ;
	}
}


__device__ void axmb_csr_kernel_vector_dev (
									  float * matrix, unsigned int size_matrix,
									  uint2 * rowptr, unsigned int size_rowptr,
									  unsigned int * colind, unsigned int size_colind,
									  float * x, float * b, float* r, unsigned int size_vec )
{
#define VEC_SZ 8
	__shared__ float vals [THREAD_BLOCK_SIZE*THREAD_BLOCK_SIZE];
	const int thread_id = compute_thread_index() ; // global thread index
	const int warp_id = thread_id / VEC_SZ; // global warp index
	const int lane = thread_id & (VEC_SZ - 1); // thread index within the warp
	// one warp per row
	int row = warp_id;
	const int thread_block_id = threadIdx.x + THREAD_BLOCK_SIZE*threadIdx.y;
	const int thread_max = gridDim.x*gridDim.y*blockDim.x*blockDim.y;
	const int row_delta = thread_max/VEC_SZ;

	for(int J = 0;J<VEC_SZ;J++,row+=row_delta){
		if ( row < size_vec)
		{
			const uint2 rowptr_bounds = rowptr[row] ;
			const int row_start	=	rowptr_bounds.x;
			const int row_end	=	rowptr_bounds.y;
			// compute running sum per thread
			vals [ thread_block_id ] = 0;
			//float res = 0;
			for ( int jj = row_start + lane ; jj < row_end ; jj += VEC_SZ){
				const float xv = tex1Dfetch(texXf,colind[jj]);
				vals [ thread_block_id ] += matrix [jj] * xv;
			}

			// parallel reduction in shared memory
			//if ( lane < 16) vals [ thread_block_id ] += vals [ thread_block_id + 16];
			//if ( lane < 8) vals [ thread_block_id ] += vals [ thread_block_id + 8];
			if ( lane < 4) vals [ thread_block_id ] += vals [ thread_block_id + 4];
			if ( lane < 2) vals [ thread_block_id ] += vals [ thread_block_id + 2];
			if ( lane < 1) vals [ thread_block_id ] += vals [ thread_block_id + 1];
			// first thread writes the result
			if ( lane == 0){
				r[ row ] = b[row]- vals [ thread_block_id];
			}
		}
	}
#undef VEC_SZ
}

__global__ void axmb_csr_kernel_vector (
									  float * matrix, unsigned int size_matrix,
									  uint2 * rowptr, unsigned int size_rowptr,
									  unsigned int * colind, unsigned int size_colind,
									  float * x, float * b, float* r, unsigned int size_vec )
{
	axmb_csr_kernel_vector_dev(matrix,size_matrix,rowptr,size_rowptr,colind,size_colind,x,b,r,size_vec);
}

__device__ void axmb_csr_krnl_dev (
									  float * matrix, unsigned int size_matrix,
									  uint2 * rowptr, unsigned int size_rowptr,
									  unsigned int * colind, unsigned int size_colind,
									  float * x, float * b, float* r, unsigned int size_vec )
{
	const unsigned int index = compute_thread_index () ;
	if ( index < size_vec ) {

		const uint2 rowptr_bounds = rowptr[index] ;
		float res = 0.0f ;

		// for each block of the block_row, mult
		for ( unsigned int i=rowptr_bounds.x; i<rowptr_bounds.y; i++ ) { 
			const float xv = x[colind[i]];
			res += matrix[i]*xv ;
		}
		r[index] = b[index]  -res ;
	}
}

__global__ void spmv_csr_scalar (
									  float * matrix, unsigned int size_matrix,
									  uint2 * rowptr, unsigned int size_rowptr,
									  unsigned int * colind, unsigned int size_colind,
									  float * x, float * b, unsigned int size_vec )
{
	spmv_csr_scalar_dev(matrix,size_matrix,rowptr,size_rowptr,colind,size_colind,x,b,size_vec);
}

__device__ void spmv_csr_kernel_vector_dev (
									  float * matrix, unsigned int size_matrix,
									  uint2 * rowptr, unsigned int size_rowptr,
									  unsigned int * colind, unsigned int size_colind,
									  float * x, float * b, unsigned int size_vec )
{
#define VEC_SZ 8
	__shared__ float vals [THREAD_BLOCK_SIZE*THREAD_BLOCK_SIZE];
	const int thread_id = compute_thread_index() ; // global thread index
	const int warp_id = thread_id / VEC_SZ; // global warp index
	const int lane = thread_id & (VEC_SZ - 1); // thread index within the warp
	// one warp per row
	int row = warp_id;
	const int thread_block_id = threadIdx.x + THREAD_BLOCK_SIZE*threadIdx.y;
	const int thread_max = gridDim.x*gridDim.y*blockDim.x*blockDim.y;
	const int row_delta = thread_max/VEC_SZ;

	for(int J = 0;J<VEC_SZ;J++,row+=row_delta){
		if ( row < size_vec)
		{
			const uint2 rowptr_bounds = rowptr[row] ;
			const int row_start	=	rowptr_bounds.x;
			const int row_end	=	rowptr_bounds.y;
			// compute running sum per thread
			vals [ thread_block_id ] = 0;
			//float res = 0;
			for ( int jj = row_start + lane ; jj < row_end ; jj += VEC_SZ){
				const float xv = tex1Dfetch(texXf,colind[jj]);
				vals [ thread_block_id ] += matrix [jj] * xv;
			}

			// parallel reduction in shared memory
			//if ( lane < 16) vals [ thread_block_id ] += vals [ thread_block_id + 16];
			//if ( lane < 8) vals [ thread_block_id ] += vals [ thread_block_id + 8];
			if ( lane < 4) vals [ thread_block_id ] += vals [ thread_block_id + 4];
			if ( lane < 2) vals [ thread_block_id ] += vals [ thread_block_id + 2];
			if ( lane < 1) vals [ thread_block_id ] += vals [ thread_block_id + 1];
			// first thread writes the result
			if ( lane == 0){
				b[ row ] = vals [ thread_block_id];
			}
		}
	}
#undef VEC_SZ
}

__global__ void spmv_csr_kernel_vector (
									  float * matrix, unsigned int size_matrix,
									  uint2 * rowptr, unsigned int size_rowptr,
									  unsigned int * colind, unsigned int size_colind,
									  float * x, float * b, unsigned int size_vec )
{
	spmv_csr_kernel_vector_dev(matrix,size_matrix,rowptr,size_rowptr,colind,size_colind,x,b,size_vec);

}

__global__ void memberwize_mul_kernel_float ( unsigned int size,
								   float * x,
								   float * y,
								   float * r ) {

	// Thread index
	const unsigned int index = compute_thread_index () ;
		
	if ( index < size )
		r[index] = x[index]*y[index] ;
}


/********************************************/

__device__ void memberwize_mul_device ( unsigned int size,
								   float * x,
								   float * y,
								   float * r ) {

	// Thread index
	const unsigned int index = compute_thread_index () ;
		
	if ( index < size )
		r[index] = x[index]*y[index] ;
}
//
__device__ void scopy_device ( unsigned int size,
								   float * x,
								   float * r )
{

	// Thread index
	const unsigned int index = compute_thread_index () ;
		
	if ( index < size )
		r[index] = x[index] ;
}

__device__ void saxpy_device ( unsigned int size,
								   float  a,
								   float * x,
								   float * y )
{

	// Thread index
	const unsigned int index = compute_thread_index () ;
		
	if ( index < size )
		y[index] += a*x[index] ;
}

__device__ void sscal_device ( unsigned int size,
								   float  a,
								   float * x )
{

	// Thread index
	const unsigned int index = compute_thread_index () ;
		
	if ( index < size )
		x[index] *= a ;
}
//
__device__ void swap_device(float*& x, float*& y){
	float * t = x;
	x = y;
	y = t;
}




__global__ void chebyshev_iterations_krnl(
	CRS_matrix_cuda<float> A,
	float* x, float* b,
	float* r,float *z,float*p,
	float *diag_inv,int max_iter
	)
{
	unsigned int its=0;
	float lmax = 1.f,lmin=1.f;

	float c = (lmax-lmin)/2;
	float d = (lmax+lmin)/2;
	float alpha = 0,beta = 0;
	int N = A.nrows;

	while ((int)its < max_iter)
	{
		//solve M*phat = p
		//z = linsolve(preCond,r);
		memberwize_mul_device( N, diag_inv, r, z );
		
		if(its==0){
			//Blas::copy(N,p,z);
			scopy_device(N,z,p);
			alpha = 2/d;
		}
		else{
			beta = (c*alpha/2)*(c*alpha/2);
			alpha = 1/(d-beta);
			saxpy_device(N,beta,p,z);
			//Blas::axpy(N,beta,p,z);	//z = z + beta*p;
			swap_device(p,z);	//z invalid
		}
		//x=x+alpha*p;
		saxpy_device(N,alpha,p,x);

		//Blas::spmv(N,A,x,r);
		spmv_csr_scalar_dev(
			A.a, A.nelements,
			A.rowptr,A.nrows,
			A.colind,A.nelements,
			x, r , A.nrows
		);

		// r = b - A*x
		saxpy_device(N,-1.0,b,r);
		sscal_device(N,-1.0,r);

		its++;

		//if(Blas::nrm2(N,r)<epsilon){
		//	break;
		//}	
	}
}


__global__ void chebyshev_iteration_krnl(
	CRS_matrix_cuda<float> A,
	float* x, float* b,
	float* r,float *z,float*p,
	float *diag_inv,int its
	)
{
	float lmax = 1.01f;
	float lmin=  1.f;

	float c = (lmax-lmin)/2;
	float d = (lmax+lmin)/2;
	float alpha = 0,beta = 0;
	int N = A.nrows;

		//solve M*phat = p
		memberwize_mul_device( N, diag_inv, r, z );
		
		if(its==0){
			//Blas::copy(N,p,z);
			scopy_device(N,z,p);
			alpha = 2/d;
		}
		else{
			beta = (c*alpha/2)*(c*alpha/2);
			alpha = 1/(d-beta);
			saxpy_device(N,beta,p,z);
			//Blas::axpy(N,beta,p,z);	//z = z + beta*p;
			//swap_device(p,z);	//z invalid
			scopy_device(N,z,p);
		}
		//x=x+alpha*p;
		saxpy_device(N,alpha,p,x);

}


__global__ void chebyshev_iteration_krnl_s(
	CRS_matrix_cuda<float> A,
	float* x, float* b,
	float* r,float *z,float*p,
	float *diag_inv,int cur_it,int its
	)
{
	float lmax = 1.01f;
	float lmin=  1.0f;

	float c = (lmax-lmin)/2;
	float d = (lmax+lmin)/2;
	float alpha = 0,beta = 0;
	int N = A.nrows;
	int nit = 0;
	while(nit<its){
		//solve M*phat = p
		memberwize_mul_device( N, diag_inv, r, z );
		
		if(cur_it+nit==0){
			//Blas::copy(N,p,z);
			scopy_device(N,z,p);
			alpha = 2/d;
		}
		else{
			beta = (c*alpha/2)*(c*alpha/2);
			alpha = 1/(d-beta);
			saxpy_device(N,beta,p,z);
			//Blas::axpy(N,beta,p,z);	//z = z + beta*p;
			//swap_device(p,z);	//z invalid
			scopy_device(N,z,p);
		}
		//x=x+alpha*p;
		saxpy_device(N,alpha,p,x);

		__syncthreads();


		axmb_csr_krnl_dev(
			A.a, A.nelements,
			A.rowptr,A.nrows,
			A.colind,A.nelements,
			x,b, r , A.nrows);

		//saxpy_device(N,-1.0,b,r);
		//sscal_device(N,-1.0,r);
		nit++;
	}
}

void chebyshev_iterations(CRS_matrix_cuda<float> A,float* x, float* b,
						  float* r, float*z,float* p,
						  float* diag_inv,int max_iter,
						  int thread_block_sz
						  )
{
	bind_x_texf(x,A.nrows);
	const unsigned int grid_sz= (unsigned int)(sqrt((float)A.nrows)/thread_block_sz+1); 
	dim3 dim_grid(grid_sz,grid_sz);
	dim3 dim_block(thread_block_sz,thread_block_sz);
	chebyshev_iterations_krnl<<<dim_grid,dim_block >>>(
		A,x,b,r,z,p,diag_inv,max_iter
	);
	unbind_x_tex();
}

void chebyshev_iteration(CRS_matrix_cuda<float> A,float* x, float* b,
						  float* r, float*z,float* p,
						  float* diag_inv,int its,
						  int thread_block_sz
						  )
{
	const unsigned int grid_sz= (unsigned int)(sqrt((float)A.nrows)/thread_block_sz+1); 
	dim3 dim_grid(grid_sz,grid_sz);
	dim3 dim_block(thread_block_sz,thread_block_sz);
	chebyshev_iteration_krnl<<<dim_grid,dim_block >>>(
		A,x,b,r,z,p,diag_inv,its
	);
}

void chebyshev_iteration_s(CRS_matrix_cuda<float> A,float* x, float* b,
						  float* r, float*z,float* p,
						  float* diag_inv,int cur_it, int its,
						  int thread_block_sz
						  )
{
	bind_x_texf(x,A.nrows);
	const unsigned int grid_sz= (unsigned int)(sqrt((float)A.nrows)/thread_block_sz+1); 
	dim3 dim_grid(grid_sz,grid_sz);
	dim3 dim_block(thread_block_sz,thread_block_sz);
	chebyshev_iteration_krnl_s<<<dim_grid,dim_block >>>(
		A,x,b,r,z,p,diag_inv,cur_it,its
	);
	unbind_x_tex();

}

void axmb_csr_float(CRS_matrix_cuda<float> A,
					float* x, float* b, float * r ,
					unsigned int thread_block_sz )
{
	bind_x_texf(x,A.nrows);
	const unsigned int grid_sz= (unsigned int)(sqrt((float)A.nrows)/thread_block_sz+1); 
	dim3 dim_grid(grid_sz,grid_sz);
	dim3 dim_block(thread_block_sz,thread_block_sz);
	axmb_csr_kernel_vector<<<dim_grid,dim_block >>>(
		A.a, A.nelements,
		A.rowptr,A.nrows,
		A.colind,A.nelements,
		x, b ,r, A.nrows
	);
	unbind_x_tex();
}
/***********************************************/

//Interface functions

//find b = Ax
void spmv_csr_float(CRS_matrix_cuda<float> A,
					float* x, float* b, 
					unsigned int thread_block_sz )
{
	bind_x_texf(x,A.nrows);
	const unsigned int grid_sz= (unsigned int)(sqrt((float)A.nrows)/thread_block_sz+1); 
	dim3 dim_grid(grid_sz,grid_sz);
	dim3 dim_block(thread_block_sz,thread_block_sz);
	spmv_csr_kernel_vector<<<dim_grid,dim_block >>>(
		A.a, A.nelements,
		A.rowptr,A.nrows,
		A.colind,A.nelements,
		x, b , A.nrows
	);
	unbind_x_tex();
}

void cuda_memberwise_mul_float(unsigned int N, float* x, float* y,float* z,unsigned int thread_block_sz){

	const unsigned int grid_sz= (unsigned int)(sqrt((float)N)/thread_block_sz+1); 
	dim3 dim_grid(grid_sz,grid_sz);
	dim3 dim_block(thread_block_sz,thread_block_sz);
	memberwize_mul_kernel_float<<< dim_grid,dim_block>>>(N,x,y,z);
}