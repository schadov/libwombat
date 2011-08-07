#pragma once
#include "StepSolverBase.h"
#include "implicit_step.h"



template<class RealT,template<class T> class Blas ,class Func,class History>
struct EulerImplicitFunctor{
	RealT* yn_1;
	RealT h;
	Func f;
	RealT t;
	unsigned int N_;

	EulerImplicitFunctor(
		unsigned int N,
		RealT t,
		const History *y,
		RealT h,
		Func f
		):yn_1(y->last()),
		h(h),
		f(f),
		t(t), N_(N)
	{}

	void operator()(RealT* y, RealT* out) const{
		//return (RealT)1./h*(y - yn_1) - f(t,y);
		typedef Blas<RealT> Blas; 
		BlasVector<Blas> tmp(N_);
		Blas::copy(N_,yn_1,out);
		f(t,y,tmp);								//tmp = f(t,y)
		Blas::axpy(N_,-1.0,y,out);				// out = -y + y_n1
		Blas::scal(N_,(RealT)-1.0/h,out);		// out = 1/h(y - y_n1)
		Blas::axpy(N_,-1.0,tmp,out);			// out = -tmp + out === -f(t,y) + 1/h(y - y_n1)
	}

};


//template <class RealT, class Vector, class Func, class Blas, class History,class SolverFunctor>
template <template<class RealT> class Blas,class RealT,class Vector,class Func,class History>
struct EulerImplicitStep :
	public ImplicitStepSolverBase<RealT,Vector,Func,Blas,History,EulerImplicitFunctor<RealT,Blas,Func,History> >
{
	static void call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F,const History* history = 0){
		MyBaseSolver::call(N,t,h,x,F,history);
	}

	static unsigned int history_length(){
		return 2;
	}
};

//////////////////////////////////////////////////////////////////////////

template<class RealT,template<class T> class Blas ,class Func,class History>
struct EulerTrapezoidFunctor{
	RealT* yn_1;
	RealT h;
	Func f;
	RealT t;
	unsigned int N_;

	EulerTrapezoidFunctor(
		unsigned int N,
		RealT t,
		const History *y,
		RealT h,
		Func f
		):yn_1(y->last()),
		h(h),
		f(f),
		t(t), N_(N)
	{}

	void operator()(RealT* y, RealT* out) const{
		//return (RealT)1./h*(y - yn_1) - f(t,y);
		typedef Blas<RealT> Blas; 
		BlasVector<Blas> tmp(N_);
		f(t,y,tmp);										//tmp = f(t,y)

		Blas::copy(N_,yn_1,out);
		BlasVector<Blas> tmp2(N_);
		f(t-h,out,tmp2);								//tmp2 = f(t-h,yn1)
						
		Blas::axpy(N_,1.0,tmp2,tmp);
		Blas::scal(N_,0.5,tmp);

		Blas::axpy(N_,-1.0,y,out);				// out = -y + y_n1
		Blas::scal(N_,(RealT)-1.0/h,out);		// out = 1/h(y - y_n1)
		Blas::axpy(N_,-1.0,tmp,out);			// out = -tmp + out === -f(t,y) + 1/h(y - y_n1)
	}

};

template <template<class RealT> class Blas,class RealT,class Vector,class Func,class History>
struct EulerTrapezoidStep :
	public ImplicitStepSolverBase<RealT,Vector,Func,Blas,History,EulerTrapezoidFunctor<RealT,Blas,Func,History> >
{
	static void call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F,const History* history = 0){
		MyBaseSolver::call(N,t,h,x,F,history);
	}

	static unsigned int history_length(){
		return 2;
	}
};
