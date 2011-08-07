#pragma once
#include "StepSolverBase.h"

template <template<class R> class Blas,class RealT,class Vector,class Func, class History>
struct EulerStep: public StepSolverBase<Blas<RealT> >{
	static void call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F,const History* history = 0){
		//x = x + h*F(t,x);
		MyBlasVector tmp(N);
		F(t,x,tmp);
		Blas<RealT>::axpy(N,h,tmp,x);
	}
};


template <template<class R> class Blas,class RealT,class Vector,class Func,class History>
struct HeunStep : public StepSolverBase<Blas<RealT> >{
	static void call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F,const History* history = 0){
		//x = x+ h/(real_t)2.*(F(t,x)+F(t+h,x+h*F(t,x)));
		typedef Blas<RealT> Blas;
		MyBlasVector ftx(N);
		F(t,x,ftx);					//ftx = F(t,x)

		MyBlasVector y2(N);
		Blas::copy(N,x,y2);
		Blas::axpy(N,h,ftx,y2);		//x+h*F(t,x);  (y2= ftx*h + y2)

		MyBlasVector ftx2(N);
		F(t+h,y2,ftx2);				//F(t+h,x+h*F(t,x))

		Blas::axpy(N,1.0,ftx,ftx2); //(F(t,x)+F(t+h,x+h*F(t,x)))

		Blas::axpy(N,h/(RealT)2.0,ftx2,x);
	}
};
