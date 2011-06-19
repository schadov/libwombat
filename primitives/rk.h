#pragma once
#include "StepSolverBase.h"

template <class Blas,class RealT,class Vector,class Func>
struct Rk4Step : public StepSolverBase<Blas>{
	static void call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F){
		Vector k1,k2,k3,k4;

		Vector tmp;
		Blas::allocate(N,tmp);
		Blas::copy(N,x,tmp);

		Blas::allocate(N,k1);
		Blas::allocate(N,k2);
		Blas::allocate(N,k3);
		Blas::allocate(N,k4);

		//k1
		F(t,x,k1);

		//k2
		RealT t1 = t+h/2;
		Blas::axpy(N,h/2,k1,tmp);
		F(t1,tmp,k2);
		Blas::copy(N,x,tmp);

		//k3
		Blas::axpy(N,h/2,k2,tmp);
		F(t1,tmp,k3);
		Blas::copy(N,x,tmp);

		//k4
		RealT t2 = t+h;
		Blas::axpy(N,h,k3,tmp);
		F(t2,tmp,k4);

		//K
		Blas::axpy(N,2,k2,k1);
		Blas::axpy(N,2,k3,k4);
		Blas::axpy(N,1,k1,k4);

		//x
		Blas::axpy(N,h/6,k4,x);

		Blas::deallocate(tmp);
		Blas::deallocate(k1);
		Blas::deallocate(k2);
		Blas::deallocate(k3);
		Blas::deallocate(k4);
	}
};
