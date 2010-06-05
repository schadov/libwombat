// primitives.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>

#include <boost/bind.hpp>

#include "MKL_blas.h"
#include "CUDABlas.h"
#include "PMKLBlas.h"
#include "SparseMatrix.h"
#include "cuda_vector.h"

#include "bicg_stab.h"
#include "chebyshev.h"

void blas_test() 
{
	float arr []= {1,2,3,4,5};
	float arr2 []= {1,2,3,4,5};
	float* bla = 0;
	const int N = 5;
#define Blas PMKLBlas
	Blas::init(N);
	Blas::allocate(N,bla);
	Blas::set(N,arr,bla);
	Blas::scal(N,2.5,bla);
	Blas::extract(N,bla,arr2);

	for (unsigned int i=0;i<N;++i)
	{
		std::cout << arr2[i] << std::endl;
	}
}

void empty_fun(int,float){

}

void sparse_test(){
	FullMatrix<float> matr;
	CNCLoader<float>::load(L"N:\\libs\\CNC\\examples\\out.dat",
		boost::bind(&FullMatrix<float>::set_value,boost::ref(matr),_1,_2,_3),
		empty_fun,
		boost::bind(&FullMatrix<float>::reserve,boost::ref(matr),_1)
	);

	//check if load is ok
	/*if(fabs(matr.get_value(12730, 9919) -  -.002050355294728273)>0.01){
		std::cout << "Err " << "12730, 9919 != -.002050355294728273 == " <<matr.get_value(12730, 9919) << std::endl; 
	}

	if(fabs(matr.get_value(5275, 15621 ) - 1.481165344704301)>0.01){
		std::cout << "Err " << "5275, 15621 != 1.481165344704301 == " <<matr.get_value(12730, 9919) ;
	}*/

	//convert to sparse and check spmv
	SparseMatrixCRS<float> sp1,sp2;
	sp1.from_full(matr);
	sp2.from_full(matr);

	std::vector<float> x(matr.rows());
	for(unsigned int i=0;i<x.size();++i){
		if(i%2==0)
			x[i] = 0;
		else x[i] = 2.0f;
	}
	const int sz = matr.rows();
	std::vector<float> y1(sz);
	std::vector<float> y2(sz);

	sp1.spmv_tbb(&x[0],&y1[0]);

	//float* gpu_x,*gpu_y;
	
	CudaVector<float> gpu_x(sz),gpu_y(sz);
	CUDABlas::set(sz,&x[0],gpu_x.get());

	CRS_matrix_cuda<float> cum =  sp2.load_to_gpu();
	SparseMatrixCRS<float>::spmv_cuda(cum,gpu_x.get(),gpu_y.get());

	
	CUDABlas::extract(sz,gpu_y.get(),&y2[0]);
	
	sp2.deallocate_gpu(cum);

	for (unsigned int i=0;i<y1.size();++i)
	{
		if(fabs(y1[i]-y2[i])>0.00000000001){
			std::cout << "SPMV mismatch " << y1[i] <<" "<<y2[i] << std::endl;
		}
	}

	int d = 0;
}

void allocate_matrix_and_vector(FullMatrix<float>& matr,std::vector<float>&vec, unsigned int N ){
	matr.reserve(N);
	vec.resize(N);
}

const float coeff = 1.f;
template<class T,class V> void set_value(T& c, unsigned int p,const V&v){
	c[p] = v*coeff;
}

void set_matrix_value(FullMatrix<float>& matr,int a,int b,float v){
	matr.set_value(a,b,v*coeff);
}

void solve_test(){
	std::vector<float> b;
	FullMatrix<float> matr;
	CNCLoader<float>::load(L"N:\\libs\\CNC\\examples\\example_1.dat",
		boost::bind(set_matrix_value,boost::ref(matr),_1,_2,_3),
		boost::bind(set_value<std::vector<float>,float>,boost::ref(b),_1,_2),
		boost::bind(allocate_matrix_and_vector,boost::ref(matr),boost::ref(b),_1)
	);

	SparseMatrixCRS<float> A;
	A.from_full(matr);

	std::vector<float> x0(b.size());
	unsigned int N = b.size();

	float eps = 0.004f;
	//solve_bicgstab(A,&b[0],&x0[0],N,8000,eps,16,CUDABlas());
	solve_chebyshev(A,&b[0],&x0[0],N,2,eps,16,CUDABlas());
	//solve_chebyshev2(A,&b[0],&x0[0],N,8000,eps,16,CUDABlas());
	//solve_chebyshev3(A,&b[0],&x0[0],N,8000,eps,16,CUDABlas());
	//solve_chebyshev(A,&b[0],&x0[0],N,8000,eps,16,PMKLBlas());


}

int _tmain(int argc, _TCHAR* argv[])
{

	//blas_test();
	tbb::task_scheduler_init init;
	solve_test();

	return 0;
}

