#pragma once
#include "StepSolverBase.h"


template <template<class R> class TBlas,class RealT,class Vector,class Func,class History>
struct Skvortsov1Step : public StepSolverBase<TBlas<RealT> >{

	static void call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F,const History* history = 0){

		typedef TBlas<RealT> Blas;
		typedef BlasVector<Blas> BlasVector;

		const RealT alpha = static_cast<RealT>(0.001);

		BlasVector k0(N);
		F(t,x,k0);

		BlasVector u1(N);
		Blas::copy(N,x,u1);
		Blas::axpy(N,h,k0,u1);

		const RealT t1 = t + h;
		BlasVector k1(N);
		F(t1,u1,k1);

		BlasVector k2(N);
		BlasVector u2(N);
		Blas::copy(N,k1,u2);
		Blas::axpy(N,-1.0,k0,u2);
		Blas::scal(N,h*alpha,u2);
		F(t1,u2,k2);

		std::vector<RealT> kh0(N);
		std::vector<RealT> kh1(N);
		std::vector<RealT> kh2(N);

		std::vector<RealT> xh(N);
		std::vector<RealT> uh1(N);
		Blas::extract(N,u1,&uh1[0]);

		Blas::extract(N,k0,&kh0[0]);
		Blas::extract(N,k1,&kh1[0]);
		Blas::extract(N,k2,&kh2[0]);


		for (unsigned int i=0;i<N;++i)
		{	
			RealT a = alpha*(kh1[i]-kh0[i]);
			RealT b = kh2[i] - kh1[i];
			RealT c;
			if(abs(b)<=1.6*abs(a)){
				if(b!=0) b = b/a;
				c = static_cast<RealT>(1/2.+b/6.);
			}
			else{
				a = a/b;
				if(a<0) 
					c = -a*(1+a);
				else 
					c = static_cast<RealT>(1.23*a);
			}
			xh[i] = uh1[i]+h*c*(kh1[i]-kh0[i]);
		}

		Blas::set(N,&xh[0],x);
	}
};

template <template<class R> class TBlas,class RealT,class Vector,class Func,class History>
struct Skvortsov1StepCPU : public StepSolverBase<TBlas<RealT> >{

	void call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F,const History* history = 0){
		typedef TBlas<RealT> Blas;
		typedef BlasVector<Blas> BlasVector;

		const RealT alpha = static_cast<RealT>(0.001);

		BlasVector k0(N);
		F(t,x,k0);

		BlasVector u1(N);
		Blas::copy(N,x,u1);
		Blas::axpy(N,h,k0,u1);

		const RealT t1 = t + h;
		BlasVector k1(N);
		F(t1,u1,k1);

		BlasVector k2(N);
		BlasVector u2(N);
		Blas::copy(N,k1,u2);
		Blas::axpy(N,-1.0,k0,u2);
		Blas::scal(N,h*alpha,u2);
		F(t1,u2,k2);

		for (unsigned int i=0;i<N;++i)
		{	
			RealT a = alpha*(k1[i]-k0[i]);
			RealT b = k2[i] - k1[i];
			RealT c;
			if(abs(b)<=1.6*abs(a)){
				if(b!=0) b = b/a;
				c = static_cast<RealT>(1/2.+b/6.);
			}
			else{
				a = a/b;
				if(a<0) 
					c = -a*(1+a);
				else 
					c = static_cast<RealT>(1.23*a);
			}
			x[i] = u1[i]+h*c*(k1[i]-k0[i]);
		}
	}
};


//////////////////////////////////////////////////////////////////////////

template <template<class R> class TBlas,class RealT,class Vector,class Func,class History>
struct Skvortsov2Step : public StepSolverBase<TBlas<RealT> >{

	void call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F,const History* history = 0){

		typedef TBlas<RealT> Blas;
		typedef BlasVector<Blas> BlasVector;

		const RealT alpha = static_cast<RealT>(0.001);

		BlasVector k0(N);
		F(t,x,k0);

		//u1 = x+h*k0;
		//k1 = F(t1,u1)
		BlasVector u1(N);
		Blas::copy(N,x,u1);
		Blas::axpy(N,h,k0,u1);

		const RealT t1 = t + h;
		BlasVector k1(N);
		F(t1,u1,k1);

		//u2 = u1 + h/2*(k1-k0); 
		//k2 = F(t1,u2);
		BlasVector k2(N);
		BlasVector u2(N);
		Blas::copy(N,k1,u2);
		Blas::axpy(N,-1.0,k0,u2);
		Blas::axpy(N,h/2,u2,u1);		//u1=u2
		F(t1,u1,k2);

		//u3 = u2 + h*alpha*(k2-k1);
		//k3 = F(t1,u3);
		BlasVector k3(N);
		BlasVector u3(N);
		Blas::copy(N,k2,u3);
		Blas::axpy(N,-1.0,k1,u3);
		Blas::scal(N,h*alpha,u3);
		Blas::axpy(N,1.0,u1,u3);
		F(t1,u3,k3);

		std::vector<RealT> kh3(N);
		std::vector<RealT> kh1(N);
		std::vector<RealT> kh2(N);

		std::vector<RealT> xh(N);
		std::vector<RealT> uh2(N);
		Blas::extract(N,u1,&uh2[0]);

		Blas::extract(N,k3,&kh3[0]);
		Blas::extract(N,k1,&kh1[0]);
		Blas::extract(N,k2,&kh2[0]);

		for (unsigned int i=0;i<N;++i)
		{	
			RealT a = alpha*(kh2[i]-kh1[i]);
			RealT b = kh3[i] - kh2[i];
			RealT c;
			if(abs(b)<=2*abs(a)){
				if(b!=0) b = b/a;
				c = static_cast<RealT>(1/3.+b/24.);
			}
			else{
				a = a/b;
				if(a<0) 
					c = static_cast<RealT>(1.13*a*(1+a)/(a-1));
				else 
					c = a;
			}
			xh[i] = uh2[i]+h*c*(kh2[i]-kh1[i]);
		}

		Blas::set(N,&xh[0],x);
	}
};


template <template<class R> class TBlas,class RealT,class Vector,class Func,class History>
struct Skvortsov2StepCPU : public StepSolverBase<TBlas<RealT> >{

	void call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F,const History* history = 0){

		typedef TBlas<RealT> Blas;
		typedef BlasVector<Blas> BlasVector;

		const RealT alpha = static_cast<RealT>(0.001);

		BlasVector k0(N);
		F(t,x,k0);

		//u1 = x+h*k0;
		//k1 = F(t1,u1)
		BlasVector u1(N);
		Blas::copy(N,x,u1);
		Blas::axpy(N,h,k0,u1);

		const RealT t1 = t + h;
		BlasVector k1(N);
		F(t1,u1,k1);

		//u2 = u1 + h/2*(k1-k0); 
		//k2 = F(t1,u2);
		BlasVector k2(N);
		BlasVector u2(N);
		Blas::copy(N,k1,u2);
		Blas::axpy(N,-1.0,k0,u2);
		Blas::axpy(N,h/2,u2,u1);		//u1=u2
		F(t1,u1,k2);

		//u3 = u2 + h*alpha*(k2-k1);
		//k3 = F(t1,u3);
		BlasVector k3(N);
		BlasVector u3(N);
		Blas::copy(N,k2,u3);
		Blas::axpy(N,-1.0,k1,u3);
		Blas::scal(N,h*alpha,u3);
		Blas::axpy(N,1.0,u1,u3);
		F(t1,u3,k3);

		for (unsigned int i=0;i<N;++i)
		{	
			RealT a = alpha*(k2[i]-k1[i]);
			RealT b = k3[i] - k2[i];
			RealT c;
			if(abs(b)<=2*abs(a)){
				if(b!=0) b = b/a;
				c = static_cast<RealT>(1/3.+b/12.);
			}
			else{
				a = a/b;
				if(a<0) 
					c = static_cast<RealT>(a*(1+a)/(a-1));
				else 
					c = a;
			}
			
			x[i] = u1[i]+h*c*(k2[i]-k1[i]);
		}

	}
};


template <template<class R> class TBlas,class RealT,class Vector,class Func,class History>
struct Skvortsov2StepCPU_ : public StepSolverBase<TBlas<RealT> >{

	int call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F,const History* history = 0){

		typedef TBlas<RealT> Blas;
		typedef BlasVector<Blas> BlasVector;

		const RealT alpha = static_cast<RealT>(0.001);

		BlasVector k0(N);
		F(t,x,k0);

		//u1 = x+h*k0;
		//k1 = F(t1,u1)
		BlasVector u1(N);
		Blas::copy(N,x,u1);
		Blas::axpy(N,h,k0,u1);

		const RealT t1 = t + h;
		BlasVector k1(N);
		F(t1,u1,k1);

		//u2 = u1 + h/2*(k1-k0); 
		//k2 = F(t1,u2);
		BlasVector k2(N);
		BlasVector u2(N);
		Blas::copy(N,k1,u2);
		Blas::axpy(N,-1.0,k0,u2);
		Blas::axpy(N,h/2,u2,u1);		//u1=u2
		F(t1,u1,k2);

		//u3 = u2 + h*alpha*(k2-k1);
		//k3 = F(t1,u3);
		BlasVector k3(N);
		BlasVector u3(N);
		Blas::copy(N,k2,u3);
		Blas::axpy(N,-1.0,k1,u3);
		Blas::scal(N,h*alpha,u3);
		Blas::axpy(N,1.0,u1,u3);
		F(t1,u3,k3);

		real_t q = 0;
		int r=0;
		for (unsigned int i=0;i<N;++i)
		{	
			RealT a = alpha*(k2[i]-k1[i]);
			RealT b = k3[i] - k2[i];
			RealT c;
			if(abs(b)<=2*abs(a)){
				if(b!=0) b = b/a;
				c = static_cast<RealT>(1/3.+b/12.);
			}
			else{
				if(abs(b)>5*abs(a)){
					return 1;
				}
				
				a = a/b;
				if(a<0) 
					c = static_cast<RealT>(a*(1+a)/(a-1));
				else 
					c = a;
			}
			x[i] = u1[i]+h*c*(k2[i]-k1[i]);
		}
		return r;
	}
};



template <template<class R> class TBlas,class RealT,class Vector,class Func,class History>
struct Skvortsov3StepCPU : public StepSolverBase<TBlas<RealT> >{

	void call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F,const History* history = 0){

		typedef TBlas<RealT> Blas;
		typedef BlasVector<Blas> BlasVector;

		const RealT alpha = static_cast<RealT>(0.001);

		BlasVector k0(N);
		F(t,x,k0);

		//u1 = x + h/2*k0; 
		//k1 = F(t+h/2,u1);
		BlasVector u1(N);
		Blas::copy(N,x,u1);
		Blas::axpy(N,h/2,k0,u1);

		const RealT t1 = t + h;
		BlasVector k1(N);
		F(t1,u1,k1);

		//u2 = x + h*k0; 
		//k2 = F(t1,u2);
		BlasVector k2(N);
		BlasVector u2(N);
		Blas::copy(N,x,u2);
		Blas::axpy(N,h,k0,u2);
		F(t1,u1,k2);

		//u3 = x + h*(2*k1-(k0+k1)/2);
		//k3 = F(t1,u3);
		BlasVector k3(N);
		BlasVector u3(N);
		Blas::copy(N,k0,k3);
		Blas::copy(N,x,u3);
		Blas::axpy(N,1.0,k1,k3);
		Blas::scal(N,0.5,k3);
		Blas::axpy(N,-2,k1,k3);
		Blas::axpy(N,-h,k3,u3);
		F(t1,u3,k3);

		//u4 = x + h/6*(k0+4*k1-k2+2*k3);
		//k4 = F(t1,u4);
		BlasVector k4(N);
		BlasVector u4(N);
		Blas::copy(N,k0,k4);
		Blas::axpy(N,4,k1,k4);
		Blas::axpy(N,-1,k2,k4);
		Blas::axpy(N,2,k3,k4);
		
		Blas::copy(N,x,u4);
		Blas::axpy(N,h/6,k4,u4);

		F(t1,u4,k4);

		//u5 = u4+h*alpha*(k4-k3);
		//k5 = F(t1,u5);
		BlasVector k5(N);
		BlasVector u5(N);
		Blas::copy(N,k4,k5);
		Blas::axpy(N,-1,k3,k5);

		Blas::copy(N,u4,u5);
		Blas::axpy(N,h*alpha,k5,u5);
		F(t1,u5,k5);

		for (unsigned int i=0;i<N;++i)
		{	
			RealT a = alpha*(k4[i]-k3[i]);
			RealT b = k5[i] - k4[i];
			RealT c;
			if(abs(b)<=2.2*abs(a)){
				if(b!=0) b = b/a;
				c = static_cast<RealT>(1/4.+b/20.);
			}
			else{
				a = a/b;
				if(a<0) 
					c = -a*(a*(a*(a*6+6)+3)+1);
				else 
					c = static_cast<RealT>(0.792*a);
			}
			x[i] = u4[i]+h*c*(k4[i]-k3[i]);
		}

	}
};



template <template<class R> class TBlas,class RealT,class Vector,class Func,class History>
struct Rk4StepStabilized : public StepSolverBase<TBlas<RealT> >{
	void call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F,const History* history = 0){
		
		typedef TBlas<RealT> Blas;
		typedef BlasVector<Blas> BlasVector;
		BlasVector k1(N),k2(N),k3(N),k4(N);
		
		BlasVector tmp(N);
		Blas::copy(N,x,tmp);

		//k1
		F(t,x,k1);

		//k2
		RealT t1 = t+h/2;
		Blas::axpy(N,h/2,k1,tmp);
		F(t1,tmp,k2);
		Blas::copy(N,x,tmp);

		//k3
		Blas::axpy(N,h/2,k2,tmp);
		F(t1,tmp,k3);
		Blas::copy(N,x,tmp);

		//k4
	/*	RealT t2 = t+h;
		Blas::axpy(N,h,k3,tmp);
		F(t2,tmp,k4);*/

		for (unsigned int i=0;i<N;++i)
		{
			RealT a = k2[i]-k1[i];
			RealT b = k3[i]-k2[i];
			if(std::abs(b)<=std::abs(a) || a*b >= 0){
				tmp[i] = x[i]+h*k3[i];
			}
			else{
				b = a/b;
				b = -(1+3*b*(1+b+0.5f*b*b ));
				tmp[i] = x[i]+h*(k1[i] + a*b);
			}

		}
		F(t+h,tmp,k4);

		//K
		Blas::axpy(N,2,k2,k1);
		Blas::axpy(N,2,k3,k4);
		Blas::axpy(N,1,k1,k4);

		//x
		Blas::axpy(N,h/6,k4,x);



	}
};



template <template<class R> class TBlas,class RealT,class Vector,class Func,class History>
struct BogackiShampineStab : public StepSolverBase<TBlas<RealT> >{
	RealT call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F,const History* history = 0){
		typedef TBlas<RealT> Blas;
		typedef BlasVector<Blas> BlasVector;
		BlasVector k1(N),k2(N),k3(N),k4(N),k5(N);

		BlasVector tmp(N);
		Blas::copy(N,x,tmp);
		//k1
		F(t,x,k1);

		//k2
		RealT t1 = t+h/2;
		Blas::axpy(N,h/2,k1,tmp);
		F(t1,tmp,k2);
		Blas::copy(N,x,tmp);

		//k3
		Blas::axpy(N,RealT(3.0*h/4.0),k2,tmp);
		F(t+RealT(3.0*h/4.0),tmp,k3);
		Blas::copy(N,x,tmp);


		Blas::copy(N,x,tmp);
		Blas::axpy(N,RealT(2.0/9)*h,k1,tmp);
		Blas::axpy(N,RealT(3.0/9)*h,k2,tmp);
		Blas::axpy(N,RealT(4.0/9)*h,k3,tmp);


		//k4
		RealT t2 = t+h;
		//Blas::axpy(N,h,m,tmp);
		F(t2,tmp,k4);

		//K
		Blas::axpy(N,RealT(7.0/24.0)*h,k1,x);
		Blas::axpy(N,RealT(1.0/4.0)*h,k2,x);
		Blas::axpy(N,RealT(1.0/3.0)*h,k3,x);
		Blas::axpy(N,RealT(1.0/8.0)*h,k4,x);

		RealT d = 0;
		for (unsigned int i=0;i<N;++i){
			const RealT q = x[i] - tmp[i];
			d += q*q;
		}


		for (unsigned int i=0;i<N;++i)
		{	
			RealT a = k2[i]-k1[i];
			RealT b = k3[i] - k2[i];
			RealT c;
			if(abs(b)<=abs(a) && a*b>=0){
				if(b!=0) b = b/a;
				c = 0;
			}
			else{
				a = a/b;
				if(a<0) 
					c = -a -a*a;
				else 
					c = a;

			}
			x[i] = x[i]+c*(k3[i]-k2[i]);
		}



		return sqrt(d);
	}
};

