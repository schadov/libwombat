#pragma once

#include "StepSolverBase.h"

template <template<class R> class TBlas,class RealT,class Vector,class Func,class History>
struct PFEStep : public StepSolverBase<TBlas<RealT> >{

	static void call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F,const History* history = 0){

		const int INNER_STEPS = 4;
		const RealT inner_h = h/16;
		typename StepSolverBase<TBlas<RealT> >::MyBlasVector tmp(N);
		typename StepSolverBase<TBlas<RealT> >::MyBlasVector inner_results(N*2);
		RealT inner_t = t;
		for (unsigned int i=0;i<INNER_STEPS;++i)
		{
			F(inner_t,x,tmp);
			TBlas<RealT>::axpy(N,inner_h,tmp,x);
			if(i==INNER_STEPS-1){
				TBlas<RealT>::copy(N,x,&inner_results[N]);
			}
			if(i==INNER_STEPS-2){
				TBlas<RealT>::copy(N,x,&inner_results[0]);
			}
			inner_t += inner_h;
		}
		const RealT psi = h/inner_h - INNER_STEPS;
		TBlas<RealT>::scal(N,psi+1,&inner_results[N]);
		TBlas<RealT>::axpy(N,-psi,&inner_results[0],&inner_results[N]);
		TBlas<RealT>::copy(N,&inner_results[N],x);
	
	}
};
