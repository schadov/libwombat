#pragma once
#include <vector>
#include <mkl.h>
#include <boost/utility.hpp>
#include <boost/type_traits.hpp>
#include "tbb_helper.h"
#include "cuda_vector.h"


template<class RealT> struct Dense_matrix_cuda{
	RealT* a;
	unsigned int nrows;
	unsigned int nelements;
};

template<class Real> class DenseMatrix
{
	std::vector<Real> data_;
	unsigned int dim_;

	friend class Row;

	unsigned int get_coord(int i,int j){
		return i+j*dim_;
	}

	Real& value(int i,int j) {
		return data_[get_coord(i,j)];
	}
	const Real& value(int i,int j) const{
		return data_[get_coord(i,j)];
	}

	typedef Dense_matrix_cuda<Real> GPU_matrix_type;
	GPU_matrix_type *cuda_storage_;
	

public:

	DenseMatrix(){
		cuda_storage_ = 0;
	}

	void allocate(unsigned int new_dim){
		data_.resize(new_dim*new_dim);
		dim_ = new_dim;
	}

	
	unsigned int dim()const{
		return dim_;
	}

	Real get_value(int i,int j)const {
		return data_[get_coord(i,j)];
	}

	class Row{
		DenseMatrix<Real> &m_;
		unsigned int nrow_;
		unsigned int beg_;
	public:
		Row(DenseMatrix& parent,unsigned int nrow):m_(parent),nrow_(nrow){
			beg_ = nrow_ * m_.dim();
		}

		Row(const DenseMatrix& parent,unsigned int nrow):m_(parent),nrow_(nrow){
			beg_ = nrow_ * m_.dim();
		}
		
		Real& operator [](unsigned int n){
			return m_.data_[beg_ + n];
		} 

		const Real& operator [](unsigned int n)const{
			return m_.data_[beg_ + n];
		} 
	};

	Row operator [](unsigned int n){
		return Row(*this,n);
	}

	Row operator [](unsigned int n)const{
		return Row(*this,n);
	}


	///********** SPMV***********************

	//----------------sequential naive version----------------
	void spmv(const Real* x,Real* y){
		
		for (unsigned int j=0;j<dim();++j)
		{
			const Row r = operator[](j);
			Real sum = Real(0);
			for (unsigned int i=0;i<dim();++i){
				sum += r[i] * x[i];
			}
			y[j] = sum;
		}

	}


	void spmv_asm(const float* x,float* y){

		float* data = &data_[0];
		int sz =dim();
		const float *endx = x + dim(); 
		const float *end_data = &data_[0] + data_.size(); 
		
		__asm{

			mov ebx, data
			mov esi,x
			mov edi, y
			mov eax,0
			xorps xmm7,xmm7
nex:
			movups xmm0, [ebx]
			movups xmm1, [esi]
			mulps xmm0,xmm1
			addps xmm7,xmm0
			add ebx,16
			add esi,16
			inc eax
			cmp esi,endx
			jnz nex
			mov esi,x
			xor eax,eax
			haddps xmm7,xmm7
			haddps xmm7,xmm7
			movss [edi],xmm7
			add edi,4
			xorps xmm7,xmm7
			cmp ebx,end_data
			jnz nex  

		}
		return;
	}


	//------------------ TBB------------------------------------

	friend class Tbb_spmv_worker;

	struct Tbb_spmv_worker
	{
		DenseMatrix* pa_;
		const Real* px_;
		Real* py_;

		Tbb_spmv_worker(DenseMatrix* A, const Real* x,  Real* y):pa_(A),px_(x),py_(y)
		{}
		Tbb_spmv_worker(const Tbb_spmv_worker& right,tbb::split):pa_(right.pa_),px_(right.px_),py_(right.py_)
		{}

		void operator()(tbb::blocked_range<int>& r) const
		{		
			for (int j = r.begin();j!=r.end();++j)
			{
				const DenseMatrix::Row r = pa_->operator[](j);
				Real sum = Real(0);
				for (unsigned int i=0;i<pa_->dim();++i){
					sum += r[i] * px_[i];
				}
				py_[j] = sum;
			}
		}
	};

public:

	void spmv_tbb(Real* x,Real* y, unsigned int split_threshold = 128){
		Tbb_spmv_worker worker(this,x,y);
		tbb::parallel_for(TbbRange(0,dim(),split_threshold),worker);
	}

	//------------------MKL TBB-----------------------------------

	struct Tbb_spmv_worker_mkl
	{
		DenseMatrix* pa_;
		const Real* px_;
		Real* py_;

		Tbb_spmv_worker_mkl(DenseMatrix* A, const Real* x,  Real* y):pa_(A),px_(x),py_(y)
		{}
		Tbb_spmv_worker_mkl(const Tbb_spmv_worker& right,tbb::split):pa_(right.pa_),px_(right.px_),py_(right.py_)
		{}

		template<class RealT> void _gemv(tbb::blocked_range<int>& r) const;

		template<> void _gemv<float>(tbb::blocked_range<int>& r)const{
			const int N = r.size();
			const int Ncols = pa_->dim();
			const int one = 1;
			const float done = 1.0;
			const float beta = 0;
			sgemv("t",&Ncols,&N,&done,&pa_->data_[r.begin()*pa_->dim()],&Ncols,&px_[0],&one,&beta,&py_[r.begin()],&one);
		}

		template<>  void _gemv<double>(tbb::blocked_range<int>& r)const{
			const int N = r.size();
			const int Ncols = pa_->dim();
			const int one = 1;
			const double done = 1.0;
			const double beta = 0;
			dgemv("t",&Ncols,&N,&done,&pa_->data_[r.begin()*pa_->dim()],&Ncols,&px_[0],&one,&beta,&py_[r.begin()],&one);
		}

		void operator()(tbb::blocked_range<int>& r) const
		{		
			_gemv<Real>(r);
		}
	};

	void spmv_tbb_mkl(Real* x,Real* y, unsigned int split_threshold = 256){
		Tbb_spmv_worker_mkl worker(this,x,y);
		tbb::parallel_for(TbbRange(0,dim(),split_threshold),worker);
	}

	//-------------- MKL generic------------------------------------

	void spmv_mkl(const double* x,double* y){
		const int N = dim();
		const int one = 1;
		const double done = 1.0;
		const double beta = 0;
		dgemv("n",&N,&N,&done,&data_[0],&N,x,&one,&beta,y,&one);	
		//dgemv("n",&N,&N,&done,A,&N,X,&one,&beta,Y,&one);	
	}

	void spmv_mkl(const float* x,float* y){
		const int N = dim();
		const int one = 1;
		const float done = 1.0;
		const float beta = 0;
		sgemv("t",&N,&N,&done,&data_[0],&N,x,&one,&beta,y,&one);	
		//dgemv("n",&N,&N,&done,A,&N,X,&one,&beta,Y,&one);	
	}

	//----CUDA-----

	void spmv_gpu(const CudaVector<float>& x,CudaVector<float>& y){
		
		if(cuda_storage_ == 0){
			throw std::exception("No GPU storage for matrix");
		}
		GPU_matrix_type gpu_matrix = get_gpu_storage();
		spmv_dense_float(gpu_matrix.a,const_cast<float*>(x.get()),y.get(),dim(),128);
	}

	void transponse(DenseMatrix &out){
		out.allocate(dim());
		for (unsigned int i=0;i< dim() ;++i)
		{
			for(unsigned int j=0;j<dim();++j){
				out[j][i] = (*this)[i][j];
			}
		}
	}

	GPU_matrix_type load_to_gpu()const{
		GPU_matrix_type gpu_storage;

		//allocate
		//TODO: check for errors
		CUDABlas::allocate(data_.size(),gpu_storage.a);
	
		gpu_storage.nrows = dim();
		gpu_storage.nelements = data_.size();

		CUDABlas::set(data_.size(),&data_[0],gpu_storage.a);
		
		return gpu_storage;
	}

	void load_from_gpu(GPU_matrix_type gpu_storage)
	{
		data_.resize(gpu_storage.nelements);
		CUDABlas::extract(gpu_storage.nelements,gpu_storage.a,&data_[0]);
		return ;
	}

	void deallocate_gpu(GPU_matrix_type data)const{
		CUDABlas::deallocate(data.a);
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