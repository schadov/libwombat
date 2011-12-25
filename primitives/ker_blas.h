#pragma once

__device__ inline  unsigned int compute_thread_index () {
	return ( blockIdx.x*blockDim.x*blockDim.y+
		blockIdx.y*blockDim.x*blockDim.y*gridDim.x+
		threadIdx.x+threadIdx.y*blockDim.x) ;
}


__device__ copyf(float* x, float*y){
	const int idx = compute_thread_index();
	y[i] = x[i];
}


__device__ scalf(float a, float* x, float*y){
	const int idx = compute_thread_index();
	y[i] = a*x[i];
}

__device__ axpyf(float a, float* x, float*y){
	const int idx = compute_thread_index();
	y[i] = a*x[i] + y[i];
}