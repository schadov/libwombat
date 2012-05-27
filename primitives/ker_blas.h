#pragma once



__device__ inline  unsigned int compute_thread_index () {
	return ( blockIdx.x*blockDim.x*blockDim.y+
		blockIdx.y*blockDim.x*blockDim.y*gridDim.x+
		threadIdx.x+threadIdx.y*blockDim.x) ;
}

__device__ inline  unsigned int get_total_num_threads () {
	return (gridDim.x * gridDim.y * gridDim.z)*(blockDim.x * blockDim.y * blockDim.z);
}

namespace kernel_blas{

__device__ void copyf(float* x, float*y){
	const int i = compute_thread_index();
	y[i] = x[i];
}


__device__ void  scalf(float a, float* x, float*y){
	const int i = compute_thread_index();
	y[i] = a*x[i];
}

__device__ void axpyf(float a, float* x, float*y){
	const int i = compute_thread_index();
	y[i] = a*x[i] + y[i];
}

}