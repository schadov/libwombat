template<class Blas,class RealT,class Matrix,class Vector>
class LUsolver{
protected:
	void decompose(unsigned int N,Matrix &A)
	{
		typedef unsigned int uint;
		for (uint k=0; k<N; ++k){
			for (uint i=k+1; i<N; ++i){
				A(i,k) = A(i,k)/A(k,k);
			}
			for (uint i=k+1;i<N;++i)
				for (uint j=k+1;j<N;++j)	
					A(i,j) -= A(i,k)*A(k,j);
		}
	}

	void substitute(unsigned int N,const Matrix& A, const Vector& b, Vector& x)
	{
		typedef unsigned int uint;
		BlasVector<Blas> y(N);
		for (uint i=0;i<N;++i)
		{
			RealT sum = 0;
			for (uint j=0;j<i;++j){
				sum+=A(i,j)*y[j];
			}
			y[i] = b[i] - sum;
		}
		for (int i=N-1;i>=0;--i)
		{
			RealT sum = 0;
			for (uint j=i+1;j<N;++j){
				sum+=A(i,j)*x[j];
			}
			x[i] = (y[i] - sum)/A(i,i);
		}
	}

	void lu_solve_inplace(unsigned int N, Matrix& A, const Vector& b, Vector& x0){
		decompose(N,A);
		substitute(N,A,b,x0);
	}

	void lu_solve(unsigned int N, const Matrix& A, const Vector& b, Vector& x0){
		BlasMatrix<Blas> copy(N);
		Blas::copy(N*N,A,copy);
		decompose(N,copy);
		substitute(N,copy,b,x0);
	}


public:

	void call(unsigned int N, const Matrix& A, const Vector& b, Vector& x0){
		lu_solve(N,A,b,x0);
	}

};



template<class Blas,class RealT,class Matrix,class Vector>
class LUPsolver{

	void lup_decompose(unsigned int N, Matrix& A, Vector &pi)
	{
		typedef unsigned int uint;
		uint n = N;
		for (uint i=0;i<n;++i)
			pi[i] = i;
		int kp;	RealT p; 
		for (uint k=0;k<n;++k){
			p=0;
			for (uint i=k;i<n;++i){
				if(abs(A(i,k))>p){
					p = abs(A(i,k));
					kp = i;
				}
			}
			if(p==0)
				throw "singular matrix";
			std::swap(pi[k],pi[kp]);
			for (uint i = 0;i<n;++i)
				std::swap(A(k,i),A(kp,i));
			for (uint i=k+1;i<n;++i){
				A(i,k)=A(i,k)/A(k,k);
				for (uint j=k+1;j<n;++j)	
					A(i,j) -= A(i,k)*A(k,j);
			}
		}
	}

	void lup_substitute(unsigned int N,const Matrix& A,const Vector & pi,const Vector& b, Vector& x)
	{
		typedef unsigned int uint;
		uint n = N;
		BlasVector<Blas> y(N);
		for (uint i=0;i<n;++i)
		{
			RealT sum = 0;
			for (uint j=0;j<i;++j){
				sum+=A(i,j)*y[j];
			}
			y[i] = b[int(pi[i])] - sum;
		}
		for (int i=n-1;i>=0;--i)
		{
			RealT sum = 0;
			for (uint j=i+1;j<n;++j){
				sum+=A(i,j)*x[j];
			}
			x[i] = (y[i] - sum)/A(i,i);
		}
	}

	void lup_solve_inplace(unsigned int N, Matrix& A, const Vector& b, Vector& x0){
		BlasVector<Blas> pi;
		lup_decompose(N,A,pi);
		return lup_substitute(A,pi,b);
	}

	void lup_solve(unsigned int N, const Matrix& A, const Vector& b, Vector& x0){

		BlasMatrix<Blas> copy(N);
		Blas::copy(N*N,A,copy);

		BlasVector<Blas> pi(N);

		lup_decompose(N,copy,pi);
		return lup_substitute(N,copy,pi,b,x0);
	}

public:

	void call(unsigned int N, const Matrix& A, const Vector& b, Vector& x0){
		lup_solve(N,A,b,x0);
	}

};


/////////////////////

template<template<class Blas,class RealT,class Matrix,class Vector>class Solver, class Blas,class RealT,class Matrix,class Vector>
class LinearSolverUtils: public Solver<Blas,RealT,Matrix,Vector>
{
public:
	void decompose(unsigned int N, Matrix& A, Vector &pi){
		return Solver<Blas,RealT,Matrix,Vector>::decompose(N,A,pi);
	}

	void decompose(unsigned int N, Matrix& A){
		return Solver<Blas,RealT,Matrix,Vector>::decompose(N,A);
	}

	void substitute(unsigned int N,const Matrix& A, const Vector& b, Vector& x){
		return Solver<Blas,RealT,Matrix,Vector>::substitute(N,A,b,x);
	}	
};


template<class Solver, class Blas,class RealT,class Matrix,class Vector>
class LinearSolverUtils2: public Solver
{
public:
	void decompose(unsigned int N, Matrix& A, Vector &pi){
		return Solver::decompose(N,A,pi);
	}

	void decompose(unsigned int N, Matrix& A){
		return Solver::decompose(N,A);
	}

	void substitute(unsigned int N,const Matrix& A, const Vector& b, Vector& x){
		return Solver::substitute(N,A,b,x);
	}	
};

