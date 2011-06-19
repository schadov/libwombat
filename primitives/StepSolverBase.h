#pragma once

template <class Blas>
class StepSolverBase{
protected:
	typedef BlasVector<Blas> MyBlasVector;
public:
	static unsigned int history_length(){
		return 1;
	}
};