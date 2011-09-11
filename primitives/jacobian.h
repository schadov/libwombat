#pragma  once
#include "defaults.h"

template<class Blas,class Matrix,class Vector,class Function>
class JacobyAuto{

	typedef typename Blas::FloatType RealT;

public:
	void call(unsigned int N,const Function& fun, const Vector &x, Matrix& J, RealT deltax=defaults::DerivDeltaDefault){
		//BlasVector<Blas> deltax(N);

		typedef BlasVector<Blas> MyBlasVector;

		MyBlasVector t(N);
		MyBlasVector t2(N);
		MyBlasVector tt(N);
		MyBlasVector ft(N);
		MyBlasVector ft2(N);
		bool has_nz = false;
		for (unsigned int j=0;j<N;++j)
		{
			Blas::copy(N,x,t);
			Blas::copy(N,x,t2);
			t[j]-=deltax;
			t2[j]+=deltax;

			fun(t,ft);
			fun(t2,ft2);
			for (unsigned int k=0;k<N;++k)
			{
				tt[k] = (ft2[k]-ft[k])/(2*deltax);
			}

			for (unsigned int i=0;i<N;++i){
				J(i,j) = tt[i];
				if(J(i,j)!=0.0)
					has_nz = true;
			}

		}

	}
};