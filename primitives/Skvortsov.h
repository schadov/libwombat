#pragma once
#include "StepSolverBase.h"


template <template<class R> class Blas,class RealT,class Vector,class Func,class History>
struct Skvortsov1Step : public StepSolverBase<Blas<RealT> >{

	static void call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F,const History* history = 0){

		typedef Blas<RealT> Blas;
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

template <template<class R> class Blas,class RealT,class Vector,class Func,class History>
struct Skvortsov1StepCPU : public StepSolverBase<Blas<RealT> >{

	void call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F,const History* history = 0){
		typedef Blas<RealT> Blas;
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

template <template<class R> class Blas,class RealT,class Vector,class Func,class History>
struct Skvortsov2Step : public StepSolverBase<Blas<RealT> >{

	void call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F,const History* history = 0){

		typedef Blas<RealT> Blas;
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


template <template<class R> class Blas,class RealT,class Vector,class Func,class History>
struct Skvortsov2StepCPU : public StepSolverBase<Blas<RealT> >{

	void call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F,const History* history = 0){

		typedef Blas<RealT> Blas;
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
				c = static_cast<RealT>(1/3.+b/24.);
			}
			else{
				a = a/b;
				if(a<0) 
					c = static_cast<RealT>(1.13*a*(1+a)/(a-1));
				else 
					c = a;
			}
			x[i] = u1[i]+h*c*(k2[i]-k1[i]);
		}

	}
};
