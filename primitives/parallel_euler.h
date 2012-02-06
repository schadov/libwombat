#pragma once
#include "StepSolverBase.h"


//Parallel euler
template <template<class R> class Blas,class RealT,class Vector,class Func, class History>
struct PEulerStep: public StepSolverBase<Blas<RealT> >{
	static void call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F, int chunk_begin,int nitems,const History* history = 0){
		//x = x + h*F(t,x);
		typename StepSolverBase<Blas<RealT> >::MyBlasVector tmp(N);
		F(t,x,tmp,chunk_begin,nitems);
		Blas<RealT>::axpy(nitems,h,&tmp[chunk_begin],&x[chunk_begin]);
	}
};

