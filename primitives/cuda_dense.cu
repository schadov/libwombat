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
/*

	void spmv(const Real* x,Real* y){
		
		for (unsigned int j=0;j<dim();++j)
		{
			const Row r = operator[](j);
			Real sum = Real(0);
			for (unsigned int i=0;i<dim();++i){
				sum += r[i] * x[i];
			}
			y[j] = sum;
		}

	}*/

__global__ void spmv_dense_float(float *A, float * x, float* y, int size){
		const int i = compute_thread_index();
			if(i<size){
		
				float s = 0;
				const int abase = size*i;
				for(int j=0;j<size;++j){
					s+=A[abase + j] * x[j];
				}
				y[i] = s;
			}
}

__global__ void spmv_dense_float_tex(float *A, float * x, float* y, int size){
		const int i = compute_thread_index();
			if(i<size){
		
				float s = 0;
				const int abase = size*i;
				for(int j=0;j<size;++j){
					s+=A[abase + j] * tex1Dfetch(texXf,j);
				}
				y[i] = s;
			}
}

void spmv_dense_float(float *A, float * x, float* y, int size,unsigned int thread_block_sz)
{
	const unsigned int grid_sz= (unsigned int)(size/thread_block_sz+1); 
	dim3 dim_grid(grid_sz,grid_sz);
	dim3 dim_block(thread_block_sz,1);
	spmv_dense_float<<< dim_grid,dim_block >>>(A,x,y,size);

/*
	bind_x_texf(x,size);
	const unsigned int grid_sz= (unsigned int)(size/thread_block_sz+1); 
	dim3 dim_grid(grid_sz,grid_sz);
	dim3 dim_block(thread_block_sz,1);
	spmv_dense_float_tex<<< dim_grid,dim_block >>>(A,x,y,size);
	unbind_x_tex();*/


}

