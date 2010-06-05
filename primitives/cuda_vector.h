#pragma once
#include "CUDABlas.h"

template<class RealT> class CudaVector{
	
	RealT * ptr_;
	unsigned int sz_;
public:
	CudaVector():ptr_(0),sz_(0){}

	explicit CudaVector(unsigned int sz):ptr_(0),sz_(0){
		resize(sz);
	}

	void resize(unsigned int sz){
		const bool need_copy = sz_!=0;
		RealT* new_ptr = 0;
		CUDABlas::allocate(sz,new_ptr);
		if(need_copy){
			CUDABlas::copy(sz_,ptr_,new_ptr);
		}
		ptr_ = new_ptr;
		sz_ = sz;
	}

	~CudaVector(){
		if(ptr_==0) return;
		CUDABlas::deallocate(ptr_);
	}

	RealT* get(){
		return ptr_;
	}

	const RealT* get()const {
		return ptr_;
	}


};