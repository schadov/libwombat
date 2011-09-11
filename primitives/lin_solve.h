#pragma once

template<class RealT,
	template<class BT> class Blas,
	template<class SBlas,class SRealT,class SMatrix,class SVector> class Solver,
	class Matrix,
	class Vector
>
void solve_linear(unsigned int N,
	const Matrix& M,
	const Vector& b,
	Vector &result
){
	Solver<Blas<RealT>, RealT, Matrix, Vector> s;
	s.call(N,M,b,result);
}