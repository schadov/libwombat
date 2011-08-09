#pragma once

#include <boost/cstdint.hpp>
#include "BlasCommon.h"

//typedef void (double,double,double*,double*) AAA; 

template<class RealT,
	template<class Breal> class BlasT,
	template<template<class RealB> class Blas,class RealT,class Vector,class Func,class History> class StepSolver,
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

	const uint64_t factor	=	1000000ui64;
	const uint64_t ih		=	(uint64_t)(h*factor+0.5);
	const uint64_t ibegin	=	(uint64_t)(dbegin*factor+0.5);
	const uint64_t iend		=	(uint64_t)(dend*factor+0.5);

	VectorT &x = result;
	Blas::copy(N,init,x);
	
	typedef StepSolver<BlasT,RealT,VectorT,FuncT,VectorDeque<RealT> > StepSolver;

	const unsigned int history_length = StepSolver::history_length();
	VectorDeque<RealT> history(N,history_length);
	if(history_length>1){
		history.push(init);
	}
	for(uint64_t ti=ibegin+ih;ti<iend;ti+=ih){
		RealT t = ti/(RealT)factor;
		StepSolver::call(N,t,h,x,F,&history);
		if(history_length>1){
			history.push(x);
		}
	}

}