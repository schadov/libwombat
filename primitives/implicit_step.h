#pragma once
#include "newton.h"
#include "lu.h"
#include "jacobian.h"

template <class RealT,
class Vector,
class Func,
template<class RealB> class Blas,
class History,
class SolverFunctor>
struct ImplicitStepSolverBase : public StepSolverBase<Blas<RealT> >{
protected:

	typedef ImplicitStepSolverBase<RealT,Vector,Func, Blas, History, SolverFunctor> MyBaseSolver;

	static void call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F,const History * history){
		SolverFunctor solver(N,t,history,h,F);
		solve_newton<RealT,NewtonSolver,Blas,LUsolver,JacobyAuto>(N,solver,x,20);
	}
};