#pragma once

template <class Blas,class SolverFunctor>
struct ImplicitStepSolverBase : public StepSolverBase<Blas>{
protected:

	static void call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F){
		SolverFunctor solver()
	}
}