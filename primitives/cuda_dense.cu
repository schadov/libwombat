//global data
texture<float, 1> texXf;

//Helper functions
__device__  unsigned int compute_thread_index () {
	return ( blockIdx.x*blockDim.x*blockDim.y+
		blockIdx.y*blockDim.x*blockDim.y*gridDim.x+
		threadIdx.x+threadIdx.y*blockDim.x) ;
}

void  inline bind_x_texf(float *x,unsigned int N)
{
	cudaBindTexture(0,texXf,x,N*sizeof(float));
}

void  inline unbind_x_tex()
{
	cudaUnbindTexture(texXf);
}


