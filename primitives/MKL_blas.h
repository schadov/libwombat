#pragma once

#include <mkl.h>

//float
#define BLASFUN(name) ::s##name

//double
//#define BLASFUN(name) s#name
#define BYTE_SIZE N*sizeof(Real)

struct MKLBlas
{
	typedef float Real;
	static const int one = 1;

	static Real dot(const unsigned int N,const Real* x,const Real* y)
	{
		return BLASFUN(dot)((const int*)&N,x,&one,y,&one);
	}

	static void scal(const unsigned int  N, const Real alpha, Real* x)
	{
		BLASFUN(scal)((const int*)&N,&alpha,x,&one);
	}

	static void axpy(	const unsigned int N,
		const Real alpha,
		const Real* x,
		Real *y
		){
			BLASFUN(axpy)((const int*)&N,&alpha,x,&one,y,&one);
	}

	static void scopy(const unsigned int N,const Real *x, Real* y){
		BLASFUN(copy)((const int*)&N,x,&one,y,&one);
	}

	static Real nrm2(const unsigned int N,const Real *x){
		return BLASFUN(nrm2)((const int*)&N,x,&one);
	}

	static void allocate(const unsigned int N, Real*& out){
		out =  (Real*)MKL_malloc(BYTE_SIZE,128);
	}

	static void deallocate(Real* p){
		MKL_free(p);
	}

	static bool init(unsigned int N){
		return true;
	}

	static void extract(const unsigned int N, const Real* devPtr, Real* hostPtr){
		memcpy(hostPtr,devPtr,BYTE_SIZE);
	}
	static void set(const unsigned int N, const Real* hostPtr, Real* devPtr){
		memcpy(devPtr,hostPtr,BYTE_SIZE);
	}
};
#undef BLASFUN
#undef BYTE_SIZE 