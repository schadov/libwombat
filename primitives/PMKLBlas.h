#pragma once

#pragma once
#include "tbb_helper.h"
#include <mkl.h>
#include "MKL_blas.h"

static const float one = 1.0f;
static const float minusOne = -1.0f;
static const int intOne = 1;

#define NTHREADITEMS N/4

#define BLASFUN(name) ::s##name

#define BYTE_SIZE N*sizeof(Real)


/************************************************************************/
/*                    vvblas_custom_cpu main class                      */
/************************************************************************/

struct PMKLBlas
{
	static const int one = 1;
	typedef float Real;

/************************************************************************/
/* local classes: TBB workers                                           */
/************************************************************************/

	struct Tbb_dot:public Binary_tbb_worker<Real>
{
	mutable Float s;

	Tbb_dot(const Float* a,const Float* b):Binary_tbb_worker<Real>((float*)a,(float*)b,0),s(0)
	{}

	Tbb_dot(Tbb_dot& a, tbb::split):Binary_tbb_worker(a.m_a,a.m_b,0),s(0){}

	void operator()(TbbRange& r)const
	{
		const int N = r.end() - r.begin();
		s = BLASFUN(dot)((const int*)&N,m_a,&one,m_b,&one);
	}
	
	void join(Tbb_dot& right){
		s+=right.s;
	}

};


struct Tbb_scal:public Unary_tbb_worker<Real>
{
	Float m_c;
	Tbb_scal(const Float* a,Float c,Float* res):Unary_tbb_worker<Real>((float*)a,res),m_c(c)
	{}

	Tbb_scal(Tbb_scal& a, tbb::split):Unary_tbb_worker((float*)a.m_a,a.m_result),m_c(a.m_c)
	{}

	void operator()(TbbRange& r)const{
		const int nitems = r.end()-r.begin();
		BLASFUN(scal)(&nitems,&m_c,&m_a[r.begin()],&intOne);
	}
};

struct Tbb_axpy:public Binary_tbb_worker<Real>
{
	Float m_alpha ;
	Tbb_axpy(const Float a,const Float* x,Float* y):Binary_tbb_worker<Real>((float*)x,(float*)y,y),m_alpha(a)
	{}

	Tbb_axpy(Tbb_axpy& right, tbb::split):Binary_tbb_worker(right.m_a,right.m_b,right.m_result),m_alpha(right.m_alpha)
	{}

	void operator()(TbbRange& r)const
	{
		const int nitems = r.end()-r.begin();
		::saxpy(&nitems,&m_alpha,&m_a[r.begin()],&intOne,&m_b[r.begin()],&intOne);
	}

};


struct Tbb_nrm2:public Unary_tbb_worker<Real>
{
	mutable Float s;

	Tbb_nrm2(const Float* a):Unary_tbb_worker<Real>((float*)a,0),s(0)
	{}

	Tbb_nrm2(Tbb_nrm2& a, tbb::split):Unary_tbb_worker(a.m_a,0),s(0){}

	void operator()(TbbRange& r)const{
		for (int i = r.begin();i!=r.end();++i){
			s+=m_a[i]*m_a[i];
		}
		/*const int N = r.end() - r.begin()
		s = BLASFUN(nrm2)((const int*)&N,m_a,&one,m_b,&one)*/
		
	}

	void join(Tbb_nrm2& right){
		s+=right.s;
	}

};

struct Tbb_vecvecmult:public Binary_tbb_worker<Real>
{
	Tbb_vecvecmult(const Float* a,const Float* b,Float *y):Binary_tbb_worker<Real>((float*)a,(float*)b,y)
	{}

	Tbb_vecvecmult(Tbb_vecvecmult& right, tbb::split):Binary_tbb_worker(right.m_a,right.m_b,right.m_result)
	{}

	void operator()(TbbRange& r)const{
		for (int i = r.begin();i!=r.end();++i){
			m_result[i]=m_a[i]*m_b[i];
		}
	}

};

/************************************************************************/
/*main blas functions                                                   */
/************************************************************************/
	
public:

	static Real dot(const unsigned int N,const Real* x,const Real* y)
	{
		/*float sd = ::sdot((const int*)&N,x,&one,y,&one);
		return sd;*/

		Tbb_dot dot_worker(x,y);
		tbb::parallel_reduce(TbbRange(0,N,NTHREADITEMS),dot_worker);
		return dot_worker.s;
	}

	static void scal(const unsigned int  N, const Real alpha, Real* x)
	{
		/*::sscal((const int*)&N,&alpha,x,&one);
		return ;*/
		Tbb_scal scal_worker(x,alpha,x);
		tbb::parallel_for(TbbRange(0,N,NTHREADITEMS),scal_worker);
	}

	static void axpy(	const unsigned int N,
		const Real alpha,
		const Real* x,
		Real *y
		)
	{
		//::saxpy((const int*)&N,&alpha,x,&one,y,&one);
		//return; 
		Tbb_axpy axpy_worker(alpha,x,y);
		tbb::parallel_for(TbbRange(0,N,NTHREADITEMS),axpy_worker);
	}

	static void copy(const unsigned int N,const Real *x, Real* y){
		BLASFUN(copy)((const int*)&N,x,&one,y,&one);
	}


	template<class T> static void allocate(const unsigned int N, T*& out){
		MKLBlas::allocate(N,out);
	}

	static void deallocate(void* ptr){
		MKLBlas::deallocate((MKLBlas::Real*)ptr);
	}


	static Real nrm2(const unsigned int N,const Real *x)
	{
		Tbb_nrm2 nrm2_worker(x);
		tbb::parallel_reduce(TbbRange(0,N,NTHREADITEMS),nrm2_worker);

		return sqrt(nrm2_worker.s);		
	}


	static bool init(unsigned int N){
		static tbb::task_scheduler_init init;
		return true;
	}


	template<class T> static void extract(const unsigned int N, const T* devPtr, T* hostPtr){
		memcpy(hostPtr,devPtr,BYTE_SIZE);
	}
	template<class T> static void set(const unsigned int N, const T* hostPtr, T* devPtr){
		memcpy(devPtr,hostPtr,BYTE_SIZE);
	}



	template<class Matrix> 
	static void initialize_matrix( Matrix& A)
	{
	}

	template<class Matrix> 
	static void deinitialize_matrix(Matrix& A){
	}

	template<class RealT> static void spmv(unsigned int N, SparseMatrixCRS<RealT>& A, RealT* x, RealT* b);

	template<> static void spmv<float>(unsigned int N, SparseMatrixCRS<float>& A, float* x, float* b){
		A.spmv_tbb(x,b);
	};

	template<class RealT> static void memberwise_mul(unsigned int N, RealT* x, RealT* y,RealT* z);

	template<> static void memberwise_mul<float>(unsigned int N, float* a, float*b,float* y)
	{
		Tbb_vecvecmult vecvec_worker(a,b,y);
		tbb::parallel_for(TbbRange(0,N,NTHREADITEMS),vecvec_worker);
	}

};

#undef NTHREADITEMS
#undef BLASFUN
#undef  BYTE_SIZE