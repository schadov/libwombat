#pragma once
#include "defaults.h"

template<class Blas,
class RealT,
class Matrix,
class Vector,
class Function,
class LinearSolver,
class JacobyCalculator> 
class NewtonSolver
{

public:

	void call(unsigned int N, Function fun, Vector& x0,unsigned int max_iter = 0 ){
		typedef unsigned int uint;
		BlasVector<Blas> x(N);
		
		Blas::copy(N,x0,x);
		uint iteration_count = 0;
		RealT alpha = 0;
		
		BlasVector<Blas> b(N);
		BlasMatrix<Blas> A(N);
		BlasMatrix<Blas> dx(N);

		JacobyCalculator jacoby;
		while (max_iter==0 || iteration_count < max_iter){
			fun(x,b);
			Blas::scal(N,-1.0,b);	//TODO: get rid of mutiplication

			jacoby.call(N,fun,x,A);

			if(alpha != 0.0 && alpha!=(RealT)1.0){
				Blas::scal(N*N,alpha,A.as_vector());
				//A*=alpha;
			}

			//solve_linear<RealT,Blas,LinearSolver>(N,A,b,dx);

			LinearSolver lin_solver;
			lin_solver.call(N,A,b,dx);

			if(Blas::nrm2(N,b)< defaults::NewtonEpsilon){ //TODO: customizable accuracy
				break;
			}

			Blas::axpy(N,1.0,dx,x);
			iteration_count++;
		}
		Blas::copy(N,x,x0);

	}
};



template<class RealT,
template<class BT> class Blas,
template<class SBlas,class SRealT,class SMatrix,class SVector> class LinearSolver,
template<class Blas,class Matrix,class Vector,class Function> class JacobyCalculator,
class Vector,
class Function
>
void solve_newton(unsigned int N,
				  Function f,
				  Vector& x0,
				  unsigned int max_iter
				  )
{
	typedef Blas<RealT> MyBlas;
	typedef BlasMatrix<MyBlas> MyMatrix;
	typedef LinearSolver<MyBlas,RealT,MyMatrix,Vector> MySolver;
	typedef JacobyCalculator<MyBlas,MyMatrix,Vector,Function> MyJacoby ;

	NewtonSolver<MyBlas, RealT, MyMatrix,Vector,Function,MySolver,MyJacoby> s;
	s.call(N,f,x0,max_iter);
}