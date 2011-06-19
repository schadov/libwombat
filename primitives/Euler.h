#pragma once
#include "StepSolverBase.h"

template <class Blas,class RealT,class Vector,class Func>
struct EulerStep: public StepSolverBase<Blas>{
	static void call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F){
		//x = x + h*F(t,x);
		MyBlasVector tmp(N);
		F(t,x,tmp);
		Blas::axpy(N,h,tmp,x);
	}
};


template <class Blas,class RealT,class Vector,class Func>
struct HeunStep : public StepSolverBase<Blas>{
	static void call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F){
		//x = x+ h/(real_t)2.*(F(t,x)+F(t+h,x+h*F(t,x)));
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


//template<class RealT>
//struct EulerImplicitClass{
//	point yprev;
//	real h;
//	pfunc f;
//	real t;
//	EulerImplicitClass(RealT t,const point_vec& y, real h, pfunc f):yprev(y.back()),h(h),f(f),t(t){}
//	point operator()(const point &y){
//		return (real_t)1./h*(y - yprev) - f(t,y);
//	}
//};
