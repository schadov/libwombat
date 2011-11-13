#pragma once

namespace rkc_detail{
	//returns the Chebyshev polynomial value at point x
	template<class real_t> static real_t T(unsigned int degree, real_t x)
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

	template<class real_t> static real_t dT(unsigned int degree, real_t x)
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

	template<class real_t> static real_t ddT(unsigned int degree, real_t x)
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

	template<class RealT> struct RKCCoeff
	{
		typedef RealT real_t;
		std::vector<RealT> a;
		std::vector<RealT> b;
		std::vector<RealT> c;
		real_t w0;
		real_t w1;
		std::vector<RealT> mu;
		std::vector<RealT> nu;

		std::vector<RealT> mu_tlde;
		std::vector<RealT> gamma;

		mutable bool F0calc;
		mutable RealT* F0val;

		real_t eps;

		//
		static real_t calc_w0(unsigned int  s, real_t eps){
			const real_t fs = static_cast<real_t>(s);
			return 1 + eps/(fs*fs);
		}

		static real_t calc_w1(unsigned int  s, real_t eps, real_t w0){
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
		void init(unsigned int s,real_t eps)
		{
			a.resize(s+1);
			b.resize(s+1);
			c.resize(s+1);

			mu.resize(s+1);
			nu.resize(s+1);

			mu_tlde.resize(s+1);
			gamma.resize(s+1);

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

	template<class Blas> class W_buf{
		VectorArray<Blas> data_;

	public:

		typedef typename Blas::FloatType RealT;
		void clear(){
			data_.clear();
		}

		 std::pair<bool,RealT*> find(unsigned int j){
			if(j>=data_.occupied_items())
				return std::make_pair(false,(RealT*)0);
			return std::make_pair(true,data_.get_vector_pointer(j));
		}

		 void add(unsigned int j, const RealT* val){
			if(j==data_.occupied_items())
				data_.push(val);
			else if(j<data_.occupied_items())
				data_.set_at(j,val);
			else throw 1;
		}

		explicit W_buf(unsigned int N):data_(N,9)	//TODO: remove magic constant
		{}

		W_buf(){}

		void init(unsigned int N){
			data_.reset(N,9);
		}
	};

	template<template<class R> class Blas, class RealT,class Function> struct W{
		unsigned int s;
		unsigned int j;

		W(unsigned int s,unsigned int j):
		s(s),j(j)
		{}

		void calc_n(
			unsigned int N,
			RealT tm,
			RealT h,
			const Function& F ,
			RealT* yn, 
			RealT* out, 
			const RKCCoeff<RealT>& coeff,
			W_buf<Blas<RealT> > & buf
			)
		{
			const std::pair<bool,RealT*> bv = buf.find(j);
			if(bv.first){
				Blas<RealT>::copy(N,bv.second,out);
				return;
			}

			typedef BlasVector<Blas<RealT> > BlasVector;
			BlasVector W0(N);
			BlasVector Wj_1(N);
			BlasVector Wj_2(N);
			W(s,0)(N,tm,h,F,yn,W0,coeff,buf);
			W(s,j-1)(N,tm,h,F,yn,Wj_1,coeff,buf);
			W(s,j-2)(N,tm,h,F,yn,Wj_2,coeff,buf);

			BlasVector fv(N);
			if(coeff.F0calc){
				Blas<RealT>::copy(N,coeff.F0val,fv);
			}
			else{
				F(tm+coeff.c[0]*h,W0,fv);
			}

			/*const point val = (1-coeff.mu[j]-coeff.nu[j])*W0 + coeff.mu[j]*Wj_1 + coeff.nu[j]*Wj_2
				+ coeff.mu_tlde[j]*h*F(tm+coeff.c[j-1]*h,Wj_1)
				+ coeff.gamma[j]*h*fv
				;*/
			RealT* val = out;
			Blas<RealT>::copy(N,W0,val);
			const RealT c1 = 1-coeff.mu[j]-coeff.nu[j];

			BlasVector f1(N);
			F(tm+coeff.c[j-1]*h,Wj_1,f1);


			Blas<RealT>::scal(N,c1,val);
			Blas<RealT>::axpy(N, coeff.mu[j],Wj_1,val);
			Blas<RealT>::axpy(N, coeff.nu[j],Wj_2,val);
			Blas<RealT>::axpy(N, coeff.mu_tlde[j]*h,f1,val);
			Blas<RealT>::axpy(N, coeff.gamma[j]*h,fv,val);

			if(!coeff.F0calc){
				coeff.F0calc = true;
				//F0val = fv;
				Blas<RealT>::copy(N,fv,coeff.F0val);
			}

			buf.add(j,val);
			//return val;
		}

		void calc_0(
			unsigned int N,
			RealT tm,
			RealT h,
			const Function& F ,
			RealT* yn, 
			RealT* out, 
			const RKCCoeff<RealT>& coeff,
			W_buf<Blas<RealT> > & buf
			)
		{
			const std::pair<bool,RealT*> bv = buf.find(0);
			if(bv.first){
				Blas<RealT>::copy(N,bv.second,out);
				return;
			}

			buf.add(0,yn);
			Blas<RealT>::copy(N,yn,out);
		}

		void calc_1(
			unsigned int N,
			RealT tm,
			RealT h,
			const Function& F ,
			RealT* yn, 
			RealT* out, 
			const RKCCoeff<RealT>& coeff,
			W_buf<Blas<RealT> > & buf
			)
		{
			const std::pair<bool,RealT*> bv = buf.find(1);
			if(bv.first){
				Blas<RealT>::copy(N,bv.second,out);
				return;
			}

			typedef BlasVector<Blas<RealT> > BlasVector;

			//const point val =  W0 + h*coeff.mu_tlde[1]*F(tm+h*coeff.c[0],W0);
			BlasVector W0(N);
			W(s,0)(N,tm,h,F,yn,W0,coeff,buf);

			BlasVector f(N);
			F(tm+h*coeff.c[0],W0,f);

			Blas<RealT>::copy(N,W0,out);
			Blas<RealT>::axpy(N,h*coeff.mu_tlde[1],f,out);
			
			buf.add(1,out);
		}

		void operator()(
			unsigned int N,
			RealT tm,
			RealT h,
			const Function& F ,
			RealT* yn, 
			RealT* out, 
			const RKCCoeff<RealT>& coeff,
			W_buf<Blas<RealT> > & buf
			)
		{
			if(this->j==0){
				return calc_0(N,tm,h,F,yn,out,coeff,buf);
			}
			else if(this->j==1){
				return calc_1(N,tm,h,F,yn,out,coeff,buf);
			}
			else return calc_n(N,tm,h,F,yn,out,coeff,buf);
		}
	};

	
}
