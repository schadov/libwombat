#pragma  once

/*
template<class Fun> rmatrix Jacoby(Fun f,const point& x,real_t delta = DERIV_DELTA)
{
using namespace boost::numeric::ublas;

rmatrix J(x.size(),x.size());
point deltax(x.size());
std::fill(deltax.begin(),deltax.end(),delta);
//for (unsigned int i=0;i<x.size();++i)
//{
//	//point fx = f(x);
//	for (unsigned int j=0;j<x.size();++j)
//	{
//		point t = x,t2 = x;
//		t[j]-=delta;
//		t2[j]+=delta;
//		point tt = element_div(f(t2)-f(t),2*deltax);

//		J(i,j) = tt[i];
//	}
//	
//}

bool has_nz = false;
for (unsigned int j=0;j<x.size();++j)
{
point t = x,t2 = x;
t[j]-=delta;
t2[j]+=delta;
point tt = element_div(f(t2)-f(t),2*deltax);
JacobyFnCount+=2;
for (unsigned int i=0;i<x.size();++i){
J(i,j) = tt[i];
if(J(i,j)!=0.0)
has_nz = true;
}

}
if(!has_nz){
assert(!"aaa");
}
return J;
}
*/

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