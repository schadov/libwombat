#pragma once
#include "StepSolverBase.h"

template <template<class R> class TBlas,class RealT,class Vector,class Func,class History>
struct Rk4Step : public StepSolverBase<TBlas<RealT> >{
	void call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F,const History* history = 0){
		Vector k1,k2,k3,k4;

		typedef TBlas<RealT> Blas;
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



template <template<class R> class TBlas,class RealT,class Vector,class Func,class History>
struct BogackiShampineStep : public StepSolverBase<TBlas<RealT> >{
	RealT call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F,const History* history = 0){
		Vector k1,k2,k3,k4,m;

		typedef TBlas<RealT> Blas;
		Vector tmp;
		Blas::allocate(N,tmp);
		Blas::copy(N,x,tmp);

		Blas::allocate(N,k1);
		Blas::allocate(N,k2);
		Blas::allocate(N,k3);
		Blas::allocate(N,k4);
		Blas::allocate(N,m);

		//k1
		F(t,x,k1);

		//k2
		RealT t1 = t+h/2;
		Blas::axpy(N,h/2,k1,tmp);
		F(t1,tmp,k2);
		Blas::copy(N,x,tmp);

		//k3
		Blas::axpy(N,RealT(3.0*h/4.0),k2,tmp);
		F(t+RealT(3.0*h/4.0),tmp,k3);
		Blas::copy(N,x,tmp);


		Blas::copy(N,x,m);
		Blas::axpy(N,RealT(2.0/9)*h,k1,m);
		Blas::axpy(N,RealT(3.0/9)*h,k2,m);
		Blas::axpy(N,RealT(4.0/9)*h,k3,m);


		//k4
		RealT t2 = t+h;
		//Blas::axpy(N,h,m,tmp);
		F(t2,m,k4);

		//K
		Blas::axpy(N,RealT(7.0/24.0)*h,k1,x);
		Blas::axpy(N,RealT(1.0/4.0)*h,k2,x);
		Blas::axpy(N,RealT(1.0/3.0)*h,k3,x);
		Blas::axpy(N,RealT(1.0/8.0)*h,k4,x);
		
		RealT d = 0;
		for (unsigned int i=0;i<N;++i){
			const RealT q = x[i] - m[i];
			d += q*q;
		}

		
		Blas::deallocate(tmp);
		Blas::deallocate(k1);
		Blas::deallocate(k2);
		Blas::deallocate(k3);
		Blas::deallocate(k4);
		Blas::deallocate(m);

		return sqrt(d);
	}
};



