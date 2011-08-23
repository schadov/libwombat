#pragma once

template <class Blas>
class StepSolverBase{
protected:
	typedef BlasVector<Blas> MyBlasVector;
public:
	unsigned int history_length(){
		return 1;
	}

	void init(unsigned int N,const typename Blas::FloatType*){
		return ;
	}
};