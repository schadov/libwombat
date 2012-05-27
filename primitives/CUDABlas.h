#pragma once

#include <cuda.h>
#include <cublas.h>
//#include "SparseMatrix.h"
#include "BlasCommon.h"

#define BLASFUN(name) cublasS##name

struct CUDABlas
{
	typedef BlasCommon::FloatType Real;
	typedef Real FloatType;
	static const int one = 1;
public:
	static Real dot(const unsigned int N,const Real* x,const Real* y)
	{
		return BLASFUN(dot)(N,x,1,y,1);
	}

	static void scal(const unsigned int  N, const Real alpha, Real* x)
	{
		BLASFUN(scal)(N,alpha,x,1);
	}

	static void axpy(	const unsigned int N,
		const Real alpha,
		const Real* x,
		Real *y
		){
			BLASFUN(axpy)(N,alpha,x,1,y,1);
	}

	static void copy(const unsigned int N,const Real *x, Real* y){
		BLASFUN(copy)(N,x,1,y,1);
	}

	template<class T> static void allocate(const unsigned int N, T*& out){
		cublasAlloc(N,sizeof(T),(void**)&out);
	}

	static void deallocate(void* ptr){
		cublasFree(ptr);
	}

	template<class T> static void extract(const unsigned int N, const T* devPtr, T* hostPtr){
		cublasGetVector(N,sizeof(T),devPtr,1,hostPtr,1);
	}
	template<class T> static void set(const unsigned int N, const T* hostPtr, T* devPtr){
		cublasSetVector(N,sizeof(T),hostPtr,1,devPtr,1);
	}

	static Real nrm2(const unsigned int N,const Real *x){
		return BLASFUN(nrm2)(N,x,1);
	}

	static bool init(unsigned int N){
		cublasStatus st = cublasInit () ;
		return true;
	}


	/*template<class Matrix> 
	static void initialize_matrix( Matrix& A)
	{
		A.attach_gpu_storage(A.load_to_gpu());
	}

	template<class Matrix> 
	static void deinitialize_matrix(Matrix& A){
		A.deallocate_gpu_storage();
	}

	template<class RealT> static void spmv(unsigned int N, SparseMatrixCRS<RealT>& A, RealT* x, RealT* b);

	template<> static void spmv<float>(unsigned int N, SparseMatrixCRS<float>& A, float* x, float* b){
		spmv_csr_float(A.get_gpu_storage(),x,b);
	}*/

	template<class RealT> static void memberwise_mul(unsigned int N, RealT* x, RealT* y,RealT* z);

	//template<> static void memberwise_mul<float>(unsigned int N, float* x, float* y,float* z)
	//{
	//	cuda_memberwise_mul_float(N,x,y,z);
	//}


};

#undef BLASFUN
#undef BYTE_SIZE 