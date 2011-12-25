#pragma once

#include <boost/cstdint.hpp>
#include "BlasCommon.h"
#include "rk.h"

//typedef void (double,double,double*,double*) AAA; 

template<class RealT,
	template<class Breal> class BlasT,
	template<template<class RealB> class Blas,class RealT,class Vector,class Func,class History> class TStepSolver,
	class FuncT,
	class VectorT
>
void solve_fixedstep(
					 unsigned int N,
					 RealT dbegin,
					 RealT dend,
					 RealT h,
					 FuncT F,
					 const VectorT& init,
					 VectorT result)
{
	using boost::uint64_t;

	typedef BlasT<RealT> Blas;

	const uint64_t factor	=	1000000;
	const uint64_t ih		=	(uint64_t)(h*factor+0.5);
	uint64_t ibegin	=	(uint64_t)(dbegin*factor+0.5);
	const uint64_t iend		=	(uint64_t)(dend*factor+0.5);

	VectorT &x = result;
	Blas::copy(N,init,x);
	
	typedef TStepSolver<BlasT,RealT,VectorT,FuncT,VectorDeque<BlasT<RealT> > > StepSolver;

	StepSolver solver;

	solver.init(N,init);

	//init history
	const unsigned int history_length = solver.history_length() ;
	VectorDeque<BlasT<RealT> > history(N,history_length - 1);
	if(history_length>1){	//requires at least 1 point of history, init with initial value
		history.push(init);
	}
	if(history_length>2){	//requires warmup
		const uint64_t warmup_end = ibegin+ih*(history_length-1);
		for(uint64_t ti=ibegin+ih; ti<warmup_end; ti+=ih){
			RealT t = ti/(RealT)factor;
			Rk4Step<BlasT,RealT,VectorT,FuncT,VectorDeque<BlasT<RealT> > > warmup_solver;
			warmup_solver.call(N,t,h,x,F,&history);
			history.push(x);
		}
		ibegin = warmup_end - ih;
	}

	//main solver loop
	for(uint64_t ti=ibegin+ih;ti<iend;ti+=ih){
		RealT t = ti/(RealT)factor;
		solver.call(N,t,h,x,F,&history);
		if(history_length>1){
			history.push(x);
		}
	}
}



	template<class RealT,
	template<class Breal> class BlasT,
	template<template<class RealB> class Blas,class RealT,class Vector,class Func,class History> class TStepSolver,
	class FuncT,
	class VectorT
>
int  solve_varstep_embedded(
					 unsigned int N,
					 RealT dbegin,
					 RealT dend,
					 RealT h,
					 RealT accuracy,
					 RealT min_step,
					 FuncT F,
					 const VectorT& init,
					 VectorT result)
{
	using boost::uint64_t;

	typedef BlasT<RealT> Blas;

	const uint64_t factor	=	1000000;
	const uint64_t ih		=	(uint64_t)(h*factor+0.5);
	uint64_t ibegin	=	(uint64_t)(dbegin*factor+0.5);
	const uint64_t iend		=	(uint64_t)(dend*factor+0.5);

	VectorT &x = result;
	Blas::copy(N,init,x);
	
	typedef TStepSolver<BlasT,RealT,VectorT,FuncT,VectorDeque<BlasT<RealT> > > StepSolver;

	StepSolver solver;

	solver.init(N,init);

	RealT hinit = h;
	//main solver loop
	for(uint64_t ti=ibegin+ih;ti<iend;){
		RealT t = ti/(RealT)factor;
		const RealT err = solver.call(N,t,h,x,F);
		if(err>accuracy){
			h = h/1.5f;
			if(h<min_step){
				return 1; //cant solve
			}
		}
		else{
			const uint64_t ih	=	(uint64_t)(h*factor+0.5);
			ti+=ih;
			if(h<hinit/1.5f){
				h*=1.5f;
			}
		}
	}
	return 0;
}



template<class RealT,
	template<class Breal> class BlasT,
	template<template<class RealB> class Blas,class RealT,class Vector,class Func,class History> class TStepSolver,
	class FuncT,
	class VectorT
>
int solve_fixedstep_test(
					 unsigned int N,
					 RealT dbegin,
					 RealT dend,
					 RealT h,
					 FuncT F,
					 const VectorT& init,
					 VectorT result)
{
	using boost::uint64_t;

	typedef BlasT<RealT> Blas;

	const uint64_t factor	=	1000000;
	const uint64_t ih		=	(uint64_t)(h*factor+0.5);
	uint64_t ibegin	=	(uint64_t)(dbegin*factor+0.5);
	const uint64_t iend		=	(uint64_t)(dend*factor+0.5);

	VectorT &x = result;
	Blas::copy(N,init,x);
	
	typedef TStepSolver<BlasT,RealT,VectorT,FuncT,VectorDeque<BlasT<RealT> > > StepSolver;

	StepSolver solver;

	solver.init(N,init);

	//init history
	const unsigned int history_length = solver.history_length() ;
	VectorDeque<BlasT<RealT> > history(N,history_length - 1);
	if(history_length>1){	//requires at least 1 point of history, init with initial value
		history.push(init);
	}
	if(history_length>2){	//requires warmup
		const uint64_t warmup_end = ibegin+ih*(history_length-1);
		for(uint64_t ti=ibegin+ih; ti<warmup_end; ti+=ih){
			RealT t = ti/(RealT)factor;
			Rk4Step<BlasT,RealT,VectorT,FuncT,VectorDeque<BlasT<RealT> > > warmup_solver;
			warmup_solver.call(N,t,h,x,F,&history);
			history.push(x);
		}
		ibegin = warmup_end - ih;
	}

	int r= 0 ;
	//main solver loop
	for(uint64_t ti=ibegin+ih;ti<iend;ti+=ih){
		RealT t = ti/(RealT)factor;
		r+=solver.call(N,t,h,x,F,&history);
		if(history_length>1){
			history.push(x);
		}
	}
	return r;
}