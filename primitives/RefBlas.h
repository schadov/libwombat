#pragma once




template<class Real> struct RefBlas
{
	typedef Real FloatType;
#define BYTE_SIZE N*sizeof(Real)
	static const int one = 1;

	/************************************************************************/
	/* level 1                                                              */
	/************************************************************************/

	static Real dot(const unsigned int N,const Real* x,const Real* y)
	{
		Real s = 0;
		for (unsigned int i=0;i<N;++i)
		{
			s+=x[i]*y[i];
		}
		return s;

	}

	static void scal(const unsigned int  N, const Real alpha, Real* x)
	{
		for (unsigned int i=0;i<N;++i){
			x[i]*=alpha;
		}		
	}

	static void axpy(	const unsigned int N,
		const Real alpha,
		const Real* x,
		Real *y
		)
	{
		for (unsigned int i=0;i<N;++i){
			y[i] += alpha*x[i];
		}	
	}

	static void copy(const unsigned int N,const Real *x, Real* y)
	{
		for (unsigned int i=0;i<N;++i){
			y[i] = x[i];
		}
	}

	static Real nrm2(const unsigned int N,const Real *x){
		Real s = 0;
		for (unsigned int i=0;i<N;++i)
		{
			s+=x[i]*x[i];
		}
		return sqrt(s);
	}

	/************************************************************************/
	/* utility                                                              */
	/************************************************************************/

	static void allocate(const unsigned int N, Real*& out){
		out =  (Real*)malloc(BYTE_SIZE);
	}

	static void deallocate(Real* p){
		free(p);
	}

	static bool init(unsigned int N){
		return true;
	}

	static void extract(const unsigned int N, const Real* devPtr, Real* hostPtr){
		memcpy(hostPtr,devPtr,BYTE_SIZE);
	}
	static void set(const unsigned int N, const Real* hostPtr, Real* devPtr){
		memcpy(devPtr,hostPtr,BYTE_SIZE);
	}

	/************************************************************************/
	/* lev2                                                                 */
	/************************************************************************/

	/*static void spmv_crs(BlasCommon::CRS_matrix<Real> A,Real* x,Real* y)
	{
		for (unsigned int i=0;i<A.nrows;++i)
		{
			Real s = 0;
			for (unsigned int j = A.rowptr[i], j<A.rowptr[i+1];++j)
			{
				s += A.data[j]*x[A.colind[j]];
			}
			y[i] = s;
		}
	}*/

	/*static void spmv_crs(const SparseMatrixCRS<Real>& A,Real* x,Real* y)
	{
		BlasCommon::CRS_matrix a={&A.data_[0],&A.colind_[0],&A.rowptr_[0],A.rows()};
		return spmv_crs(a,x,y)
	}*/

};

#undef BYTE_SIZE 