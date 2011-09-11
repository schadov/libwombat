#pragma once
#include "StepSolverBase.h"
#include "implicit_step.h"

namespace detail{
	//returns the Chebyshev polynomial value at point x
	static real_t T(unsigned int degree, real_t x)
	{
		switch(degree)
		{
		case 0:
			return 1;
		case 1:
			return x;
		case 2:
			{return 2*x*x-1;}
		case 3:
			{return 4*x*x*x - 3*x;}
		case 4:
			{return 8*x*x*x*x - 8*x*x + 1;}
		case 5:
			{return 16*x*x*x*x*x - 20*x*x*x + 5*x;}
		case 6:
			{return 32*x*x*x*x*x*x - 48*x*x*x*x + 18*x*x - 1;}
		case 7:
			{return 64*x*x*x*x*x*x*x -112* x*x*x*x*x + 56*x*x*x - 7*x;}
		case 8:
			{return 128*x*x*x*x*x*x*x*x - 256*x*x*x*x*x*x + 160*x*x*x*x - 32*x*x +1;}
		case 9:
			//{return 256*x*x*x*x*x*x*x*x*x - 576*x*x*x*x*x*x*x+432*x*x*x*x*x - 120*x*x*x + 9*x;}
			{return cos(9*acos(x));}

		default:{throw "degreeeee tooooo biiig!!!!";}
		} 
	}

	static real_t dT(unsigned int degree, real_t x)
	{
		switch(degree)
		{
		case 0:
			return 0;
		case 1:
			return 1;
		case 2:
			{return 4*x;}
		case 3:
			{return 12*x*x - 3;}
		case 4:
			{return 32*x*x*x - 16*x;}
		case 5:
			{return 80*x*x*x*x - 60*x*x + 5;}
		case 6:
			{return 192*x*x*x*x*x - 192*x*x*x + 36*x;}
		case 7:
			{return 448*x*x*x*x*x*x -560*x*x*x*x + 168*x*x - 7;}
		case 8:
			{return 1024*x*x*x*x*x*x*x - 1536*x*x*x*x*x + 640*x*x*x - 64*x; }
		case 9:
			//{return 2304*x*x*x*x*x*x*x*x - 4032*x*x*x*x*x*x + 2160*x*x*x*x - 360*x*x + 9;}
			{return 9*sin(9*acos(x))/sqrt(1-x*x);}

		default:{throw "degreeeee tooooo biiig!!!!";}
		} 
	}

	static real_t ddT(unsigned int degree, real_t x)
	{
		switch(degree)
		{
		case 0:
			return 0;
		case 1:
			return 0;
		case 2:
			{return 4;}
		case 3:
			{return 24*x;}
		case 4:
			{return 96*x*x - 16;}
		case 5:
			{return 320*x*x*x - 120*x;}
		case 6:
			{return 960*x*x*x*x - 576*x*x + 36;}
		case 7:
			{return 2688*x*x*x*x*x -2240*x*x*x + 336*x;}
		case 8:
			{return 7168*x*x*x*x*x*x - 7680*x*x*x*x + 1920*x*x - 64; }
		case 9:
			//{return 18432*x*x*x*x*x*x*x* - 24192*x*x*x*x*x + 8640*x*x*x - 720*x ;}
			{return (-81*cos(9*acos(x))/(1-x*x)) + (9*x*sin(9*acos(x))/pow(1-x*x,3/2));}
		default:{throw "degreeeee tooooo biiig!!!!";}
		} 
	}

	template<unsigned int s> struct RKCCoeff
	{
		real_t a[s+1];
		real_t b[s+1];
		real_t c[s+1];
		real_t w0;
		real_t w1;
		real_t mu[s+1];
		real_t nu[s+1];
		real_t mu_tlde[s+1];
		real_t gamma[s+1];
		real_t eps;

		//
		static real_t calc_w0(real_t s, real_t eps){
			return 1 + eps/(s*s);
		}

		static real_t calc_w1(real_t s, real_t eps, real_t w0){
			const real_t w = w0;
			return dT(s,w)/ddT(s,w);
		}

		static real_t calc_b(unsigned int j, real_t w0)
		{
			if(j<2) j = 2;
			const real_t v = dT(j,w0);
			return ddT(j,w0)/(v*v);
		}

		static real_t calc_a(unsigned int j,real_t bj, real_t w0)
		{
			return 1 - bj*T(j,w0);
		}

		static real_t calc_c(unsigned int j, unsigned int s, real_t w0)
		{
			if(j>1) return ((real_t)j*j-1)/((real_t)s*s-1);
			else if(j == 1) return calc_c(2,s,w0)/(4*w0);
			else if(j == 0) return 0;
			return 0.0;
		}

		static real_t calc_mu(unsigned int j,real_t bj,real_t bj_1,real_t w0){
			return (2*bj*w0)/(bj_1);
		}

		static real_t calc_mu_tilde(unsigned int j,real_t bj,real_t bj_1,real_t w1){
			if(j==1)
				return bj*w1;
			else return (2*bj*w1)/(bj_1);
		}

		static real_t calc_nu(unsigned int j,real_t bj,real_t bj_2){
			return -bj/bj_2;
		}

		static real_t calc_gamma(unsigned int j,real_t aj_1,real_t mutj){
			return -aj_1*mutj;
		}

	public:
		void init(real_t eps)
		{
			this->eps = eps;
			w0 = calc_w0(s,eps);
			w1 = calc_w1(s,eps,w0);

			for (unsigned int i=0;i<=s;++i)
				b[i] = calc_b(i,w0);

			for (unsigned int i=0;i<=s;++i)
				a[i] = calc_a(i,b[i],w0);

			for (unsigned int i=0;i<=s;++i)
				c[i] = calc_c(i,s,w0);

			for (unsigned int i=1;i<=s;++i)
				mu_tlde[i] = calc_mu_tilde(i,b[i],b[i-1],w1);

			for (unsigned int i=2;i<=s;++i)
				mu[i] = calc_mu(i,b[i],b[i-1],w0);

			for (unsigned int i=2;i<=s;++i)
				nu[i] = calc_nu(i,b[i],b[i-2]);

			for (unsigned int i=2;i<=s;++i)
				gamma[i] = calc_gamma(i,a[i-1],mu_tlde[i]);

		}

	};

	template<template<class R> class Blas, class RealT,class Function> struct W{
		unsigned int s;
		unsigned int j;

		W(unsigned int s,unsigned int j):
		s(s),j(j)
		{}

		point operator()(
			unsigned int N,
			RealT tm,
			RealT h,
			const Function& F ,
			RealT* yn, 
			RealT* out, 
			const RKCCoeff& coeff,
			W_buf & buf
			)
		{
			const std::pair<bool,RealT*> bv = buf.find(j);
			if(bv.first)
				return bv.second;

			typedef BlasVector<Blas<RealT> > BlasVector;
			BlasVector W0(N);
			BlasVector Wj_1(N);
			BlasVector Wj_2(N);
			W(s,0)()(tm,h,F,yn,W0,coeff,buf);
			W(s,j-1)()(tm,h,F,yn,Wj_1,coeff,buf);
			W(s,j-2)()(tm,h,F,yn,Wj_2,coeff,buf);

			const point fv = F0calc?F0val:F(tm+coeff.c[0]*h,W0);

			const point val = (1-coeff.mu[j]-coeff.nu[j])*W0 + coeff.mu[j]*Wj_1 + coeff.nu[j]*Wj_2
				+ coeff.mu_tlde[j]*h*F(tm+coeff.c[j-1]*h,Wj_1)
				+ coeff.gamma[j]*h*fv
				;
			if(!F0calc){
				F0calc = true;
				F0val = fv;
			}

			buf.add(j,val);
			return val;
		}
	};

	template<int s> struct W<s,0>{
		__forceinline point operator()(
			real_t tm,
			real_t h,
			pfunc F ,
			point yn, 
			const RKCCoeff<s>& coeff,
			W_buf & buf
			)throw()
		{
			const std::pair<bool,point> bv = buf.find(0);
			if(bv.first)
				return bv.second;

			buf.add(0,yn);
			return yn;
		}
	};

	template<int s> struct W<s,1>{
		__forceinline point operator()(
			real_t tm,
			real_t h,
			pfunc F ,
			point yn, 
			const RKCCoeff<s>& coeff,
			W_buf & buf
			)throw()
		{
			const std::pair<bool,point> bv = buf.find(1);
			if(bv.first)
				return bv.second;
			const point W0 = W<s,0>()(tm,h,F,yn,coeff,buf);
			const point val =  W0 + h*coeff.mu_tlde[1]*F(tm+h*coeff.c[0],W0);
			buf.add(1,val);
			return val;	
		}
	};
}

template <template<class R> class Blas,class RealT,class Vector,class Func,class History>
struct RKCStep : public StepSolverBase<Blas<RealT> >{

	const unsigned int s_ = 4;

	void init(unsigned int N,RealT * init){
		//RKCCoeff<s> coeff;
		////coeff.init(2.f/13.f);
		//coeff.init(4.f);
		//W_buf buf;
		RKCCoeff coeff(s_);
		coeff.init(2.f/13.f);

	}


	static void call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F,const History* history = 0){
		x=W<s,s>()(t,h,F,x,coeff,buf);
		//x=W<s>(s,t,h,F,x,coeff,buf);
		v.push_back(x);
		F0calc = false;
		buf.clear();


	}
};