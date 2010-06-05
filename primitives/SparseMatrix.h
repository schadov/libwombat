#pragma once
#include <vector>
#include <boost/bind.hpp>
#include <algorithm>
#include <map>
#include "tbb_helper.h"
#include "BlasCommon.h"
#include "CUDABlas.h"
#include "cuda_sparse.h"
#include <cassert>

#include "FullMatrix.h"

template<class RealT> class SparseMatrixCRS
{
	
	typedef std::vector<unsigned int> IndexArray;
	typedef std::vector<RealT> DataArray;

	IndexArray colind_;
	IndexArray rowptr_;
	DataArray data_;

	CRS_matrix_cuda<RealT> *cuda_storage_;

	DataArray diag_;

public:

	SparseMatrixCRS():cuda_storage_(0){

	}

	void from_full(const FullMatrix<RealT>& m)
	{
		diag_.clear();
		colind_.clear();
		rowptr_.clear();
		data_.clear();
		unsigned int ri = 0;
		for (unsigned int i=0;i<m.rows();++i)
		{
			const typename FullMatrix<RealT>::Row & row = m.row(i);
			if(row.empty()){
				continue;
			}
			rowptr_.push_back(ri);
			for (typename FullMatrix<RealT>::Row::const_iterator jt = row.begin();jt!=row.end();++jt)
			{
				if(jt->first==i){
					diag_.push_back(jt->second);
				}
				data_.push_back(jt->second);
				colind_.push_back(jt->first);
			}
			ri+=row.size();
		}
		rowptr_.push_back(data_.size());
	}

	const RealT* diag()const{
		return &diag_[0];
	}

	unsigned int rows()const{
		return rowptr_.size() - 1;
	}

	//only for non-critical code, quite slow, direct manipulation of members is preffered in algos
	RealT get(unsigned int i,unsigned int j){
		assert(i<=rows());
		assert(j<=rows());	//assume matrix is square
		for(unsigned int nj=rowptr[i] ; nj<rowptr[i+1]; nj++){
			if(j==nj){
				return data_[nj]
			}
		}
		return 0;
	}


/************************************************************************/
/* TBB spmv                                                             */
/************************************************************************/

private:
	friend class Tbb_spmv_worker;

	struct Tbb_spmv_worker
	{
		const SparseMatrixCRS* pa_;
		const RealT* px_;
		RealT* py_;

		Tbb_spmv_worker(const SparseMatrixCRS* A, const RealT* x,  RealT* y):pa_(A),px_(x),py_(y)
		{}
		Tbb_spmv_worker(const Tbb_spmv_worker& right,tbb::split):pa_(right.pa_),px_(right.px_),py_(right.py_)
		{}

		void operator()(tbb::blocked_range<int>& r) const
		{
			const typename SparseMatrixCRS::IndexArray& rowptr = pa_->rowptr_;
			const typename SparseMatrixCRS::IndexArray& colind = pa_->colind_;
			const typename SparseMatrixCRS::DataArray& a = pa_->data_;
			const RealT* const & x = px_;
			RealT* y = py_;

			for (int i = r.begin();i!=r.end();++i){
				RealT s = RealT(0.0);
				for(unsigned int j=rowptr[i] ; j<rowptr[i+1]; j++){
					s += a[j] * x[colind[j]];
				}
				y[i] = s;
			}
		}
	};

public:

	void spmv_tbb(RealT* x,RealT* y, unsigned int split_threshold = 87500){
		Tbb_spmv_worker worker(this,x,y);
		tbb::parallel_for(TbbRange(0,rows(),split_threshold),worker);
	}

	/************************************************************************/
	/* CUDA                                                                 */
	/************************************************************************/
	
	typedef CRS_matrix_cuda<RealT> GPU_matrix_type;

	GPU_matrix_type load_to_gpu()const{
		GPU_matrix_type gpu_storage;

		//allocate
		//TODO: check for errors
		CUDABlas::allocate(data_.size(),gpu_storage.a);
		CUDABlas::allocate(colind_.size(),gpu_storage.colind);
		CUDABlas::allocate(rowptr_.size(),gpu_storage.rowptr);

		gpu_storage.nrows = rows();
		gpu_storage.nelements = data_.size();

		CUDABlas::set(data_.size(),&data_[0],gpu_storage.a);
		CUDABlas::set(colind_.size(),&colind_[0],gpu_storage.colind);
		//CUDABlas::set(rowptr_.size(),&rowptr_[0],gpu_storage.rowptr_);

		std::vector<uint2> redundant_rp(rowptr_.size()-1);
		for (unsigned int i=0; i<rowptr_.size()-1; i++) {
			redundant_rp[i].x = rowptr_[i] ;
			redundant_rp[i].y = rowptr_[i+1] ;
		}
		CUDABlas::set(rowptr_.size()-1,&redundant_rp[0],gpu_storage.rowptr);
		
		return gpu_storage;
	}

	void load_from_gpu(GPU_matrix_type gpu_storage)
	{
		data_.resize(gpu_storage.nelements);
		colind_.resize(gpu_storage.nelements);
		CUDABlas::extract(gpu_storage.nelements,gpu_storage.a,&data_[0]);
		CUDABlas::extract(gpu_storage.nelements,gpu_storage.colind,&colind_[0]);

		std::vector<uint2> redundant_rp(gpu_storage.nrows);
		CUDABlas::extract(gpu_storage.nrows,gpu_storage.rowptr,&redundant_rp[0]);

		rowptr_.resize(gpu_storage.nrows+1);
		for (unsigned int i=0; i<rowptr_.size(); i++) {
			rowptr_[i]=redundant_rp[i].x;
		}
		rowptr_.back() = redundant_rp.back().y;
		return ;
	}

	void deallocate_gpu(GPU_matrix_type data)const{
		CUDABlas::deallocate(data.a);
		CUDABlas::deallocate(data.colind);
		CUDABlas::deallocate(data.rowptr);
	}

	static void spmv_cuda(GPU_matrix_type A, RealT *x, RealT *b){
		spmv_csr_float(A,x,b);
	}

	
	void attach_gpu_storage(GPU_matrix_type cuda_storage){
		assert(cuda_storage_==0);
		if(cuda_storage_==0){
			cuda_storage_ = new GPU_matrix_type(cuda_storage);
		}
	}

	void deallocate_gpu_storage(){
		if(cuda_storage_!=0){
			deallocate_gpu(*cuda_storage_);
			delete cuda_storage_;
		}
	}

	GPU_matrix_type get_gpu_storage()const{
		return *cuda_storage_;
	}


};