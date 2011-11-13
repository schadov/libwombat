#pragma once
#include "StepSolverBase.h"
#include "rkc_impl.h"

template <template<class R> class Blas,class RealT,class Vector,class Func,class History>
struct RKCStep3 : public StepSolverBase<Blas<RealT> >{

	const static unsigned int s_ = 3;
	rkc_detail::RKCCoeff<RealT> coeff_;
	rkc_detail::W_buf<Blas<RealT> > buf;

	void init(unsigned int N,RealT * init){
		buf.init(N);
		coeff_.init(s_,2.f/13.f);
		Blas<RealT>::allocate(N,coeff_.F0val);
		coeff_.F0calc = false;
	}

    void call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F,const History* history = 0){
		BlasVector<Blas<RealT> > tmp(N);
		rkc_detail::W<Blas,RealT,Func>(s_,s_)(N,t,h,F,x,tmp,coeff_,buf);
		Blas<RealT> ::copy(N,tmp,x);
		coeff_.F0calc = false;
		buf.clear();
	}

	~RKCStep3(){
		if(coeff_.F0val!=0)
			Blas<RealT>::deallocate(coeff_.F0val);
	}

	
};

#define DECLARE_RKC_METHOD(s) template <template<class R> class Blas,class RealT,class Vector,class Func,class History>\
struct RKCStep##s :\
	public RKCStep3<Blas,RealT,Vector,Func,History >{const static unsigned int s_ = s;}

DECLARE_RKC_METHOD(4);
DECLARE_RKC_METHOD(5);
DECLARE_RKC_METHOD(6);
DECLARE_RKC_METHOD(7);
DECLARE_RKC_METHOD(8);
