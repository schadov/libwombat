#pragma once

template <class RealT, class Vector, class Func, class Blas, class History,class SolverFunctor>
struct ImplicitStepSolverBase : public StepSolverBase<Blas>{
protected:

	static void call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F,History * history){
		SolverFunctor solver(t,history,h,F);
	}
}