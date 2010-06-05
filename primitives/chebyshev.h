#pragma once
#include <time.h>

template<class Matrix,class Real, class Blas> 
bool solve_chebyshev
(
 Matrix &A,
 const Real* b0,
 Real* x0,
 const unsigned int N,
 const unsigned int max_iter,
 const Real epsilon,
 const unsigned int block_size,
 Blas blas = Blas()
 )
{

	const Real * diag_matrix = A.diag();

	Blas::init(N);

#define ARRAY_ALLOCATE(x) Real *x=0;Blas::allocate<Real> ( N+4, x )
	ARRAY_ALLOCATE(r);
	ARRAY_ALLOCATE(p);
	ARRAY_ALLOCATE(z);
	ARRAY_ALLOCATE(x);
	ARRAY_ALLOCATE(b);
	ARRAY_ALLOCATE(diag_inv);
#undef ARRAY_ALLOCATE


	Blas::initialize_matrix(A);			//!!!

	// building the Jacobi preconditionner
	std::vector<Real> cpu_diag_inv ( N+4) ;
	for(unsigned int i=0; i<N; i++) {
		cpu_diag_inv[i] = (Real)((i >= N || diag_matrix[i] == 0.0) ? 1.0 : 1.0 / diag_matrix[i]) ;
	}


	Blas::set(N,&cpu_diag_inv[0],diag_inv);

	//cublasSetVector ( N, 4, x0.data()       , 1, x,        1 ) ;
	Blas::set(N,x0,x);

	//cublasSetVector ( N, 4, b0.data()       , 1, gpu_b,        1 ) ;
	Blas::set(N,b0,b);

	const int one = 1;

	unsigned int its=0;

	// r = A*x
	Blas::spmv(N,A,x,r);

	// r = b - A*x
	Blas::axpy(N,-1.0,b,r);
	Blas::scal(N,-1.0,r);

	printf("Chebyshev iteration started. initial residue:%f\n ",Blas::nrm2(N,(Real*)r));

	clock_t time_start = clock();

	const Real lmax = 1.1f,
		lmin=1.f;

	const Real c = (lmax-lmin)/2;
	const Real d = (lmax+lmin)/2;
	Real alpha = 0,beta = 0;
	while ( /*cur_err > err &&*/ (int)its < max_iter)
	{
		//solve M*phat = p
		//z = linsolve(preCond,r);
		Blas::memberwise_mul( N, diag_inv, r, z );

		if(its==0){
			Blas::copy(N,z,p);
			alpha = 2/d;
		}
		else{
			beta = (c*alpha/2)*(c*alpha/2);
			alpha = 1/(d-beta);
			Blas::axpy(N,beta,p,z);	//z = z + beta*p;
			std::swap(p,z);	//z invalid
		}
		//x=x+alpha*p;
		Blas::axpy(N,alpha,p,x);

		Blas::spmv(N,A,x,r);

		// r = b - A*x
		Blas::axpy(N,-1.0,b,r);
		Blas::scal(N,-1.0,r);

		its++;
		if(Blas::nrm2(N,r)<epsilon){
			break;
		}	
	}

	Blas::extract(N,x,x0);

	clock_t time_finished = clock();

	/// r = A*x
	Blas::spmv(N,A,x,r);
	// r = b - A*x
	Blas::axpy ( N,-1.0,b,r) ;
	Blas::scal ( N, -1.0,r) ;

	printf("=====Calculation residue:\n");
	printf("%f\n",Blas::nrm2(N,(Real*)r));
	printf("niterations=%d\n",its);
	printf("time=%d\n",time_finished-time_start);
	printf("---------------------------Chebyshev iteration end-----------------------------\n\n");

#define ARRAY_DEALLOCATE(x) Blas::deallocate ( (void*)x )
	ARRAY_DEALLOCATE(r);
	ARRAY_DEALLOCATE(p);
	ARRAY_DEALLOCATE(x);
	ARRAY_DEALLOCATE(z);
	ARRAY_DEALLOCATE(b);
	ARRAY_DEALLOCATE(diag_inv);
#undef ARRAY_DEALLOCATE

	Blas::deinitialize_matrix(A);

	return (its<max_iter) ;
}




template<class Matrix,class Real, class Blas> 
bool solve_chebyshev2
(
 Matrix &A,
 const Real* b0,
 Real* x0,
 const unsigned int N,
 const unsigned int max_iter,
 const Real epsilon,
 const unsigned int block_size,
 Blas blas = Blas()
 )
{

	const Real * diag_matrix = A.diag();

	Blas::init(N);

#define ARRAY_ALLOCATE(x) Real *x=0;Blas::allocate<Real> ( N+4, x )
	ARRAY_ALLOCATE(r);
	ARRAY_ALLOCATE(p);
	ARRAY_ALLOCATE(z);
	ARRAY_ALLOCATE(x);
	ARRAY_ALLOCATE(b);
	ARRAY_ALLOCATE(diag_inv);
#undef ARRAY_ALLOCATE


	Blas::initialize_matrix(A);			//!!!

	// building the Jacobi preconditionner
	std::vector<Real> cpu_diag_inv ( N+4) ;
	for(unsigned int i=0; i<N; i++) {
		cpu_diag_inv[i] = (Real)((i >= N || diag_matrix[i] == 0.0) ? 1.0 : 1.0 / diag_matrix[i]) ;
	}


	Blas::set(N,&cpu_diag_inv[0],diag_inv);

	//cublasSetVector ( N, 4, x0.data()       , 1, x,        1 ) ;
	Blas::set(N,x0,x);

	//cublasSetVector ( N, 4, b0.data()       , 1, gpu_b,        1 ) ;
	Blas::set(N,b0,b);

	const int one = 1;

	unsigned int its=0;

	// r = A*x
	Blas::spmv(N,A,x,r);

	// r = b - A*x
	Blas::axpy(N,-1.0,b,r);
	Blas::scal(N,-1.0,r);

	printf("Chebyshev iteration started. initial residue:%f\n ",Blas::nrm2(N,(Real*)r));

	clock_t time_start = clock();

	while ((int)its < max_iter)
	{
		chebyshev_iteration(A.get_gpu_storage(),x,b,r,z,p,diag_inv,its,block_size);
		its ++;

		axmb_csr_float(A.get_gpu_storage(),x,b,r);

		if(its%32==0 || its >= max_iter){
			if(Blas::nrm2(N,r)<epsilon){
				break;
			}	
		}
		
	}
	//Blas::extract(N,z,x0);
	Blas::extract(N,x,x0);

	clock_t time_finished = clock();

	/// r = A*x
	Blas::spmv(N,A,x,r);
	// r = b - A*x
	Blas::axpy ( N,-1.0,b,r) ;
	Blas::scal ( N, -1.0,r) ;

	printf("=====Calculation residue:\n");
	printf("%f\n",Blas::nrm2(N,(Real*)r));
	printf("niterations=%d\n",its);
	printf("time=%d\n",time_finished-time_start);
	printf("---------------------------Chebyshev iteration end-----------------------------\n\n");

#define ARRAY_DEALLOCATE(x) Blas::deallocate ( (void*)x )
	ARRAY_DEALLOCATE(r);
	ARRAY_DEALLOCATE(p);
	ARRAY_DEALLOCATE(x);
	ARRAY_DEALLOCATE(z);
	ARRAY_DEALLOCATE(b);
	ARRAY_DEALLOCATE(diag_inv);
#undef ARRAY_DEALLOCATE

	Blas::deinitialize_matrix(A);

	return (its<max_iter) ;
}


 template<class Matrix,class Real, class Blas> 
 bool solve_chebyshev3
	 (
	 Matrix &A,
	 const Real* b0,
	 Real* x0,
	 const unsigned int N,
	 const unsigned int max_iter,
	 const Real epsilon,
	 const unsigned int block_size,
	 Blas blas = Blas()
	 )
 {

	 const Real * diag_matrix = A.diag();

	 Blas::init(N);

#define ARRAY_ALLOCATE(x) Real *x=0;Blas::allocate<Real> ( N+4, x )
	 ARRAY_ALLOCATE(r);
	 ARRAY_ALLOCATE(p);
	 ARRAY_ALLOCATE(z);
	 ARRAY_ALLOCATE(x);
	 ARRAY_ALLOCATE(b);
	 ARRAY_ALLOCATE(diag_inv);
#undef ARRAY_ALLOCATE


	 Blas::initialize_matrix(A);			//!!!

	 // building the Jacobi preconditionner
	 std::vector<Real> cpu_diag_inv ( N+4) ;
	 for(unsigned int i=0; i<N; i++) {
		 cpu_diag_inv[i] = (Real)((i >= N || diag_matrix[i] == 0.0) ? 1.0 : 1.0 / diag_matrix[i]) ;
	 }


	 Blas::set(N,&cpu_diag_inv[0],diag_inv);

	 //cublasSetVector ( N, 4, x0.data()       , 1, x,        1 ) ;
	 Blas::set(N,x0,x);

	 //cublasSetVector ( N, 4, b0.data()       , 1, gpu_b,        1 ) ;
	 Blas::set(N,b0,b);

	 const int one = 1;

	 unsigned int its=0;

	 // r = A*x
	 Blas::spmv(N,A,x,r);

	 // r = b - A*x
	 Blas::axpy(N,-1.0,b,r);
	 Blas::scal(N,-1.0,r);

	 printf("Chebyshev iteration started. initial residue:%f\n ",Blas::nrm2(N,(Real*)r));

	 clock_t time_start = clock();

	 const int it_block = 8;
	 while ((int)its < max_iter)
	 {
		 chebyshev_iteration_s(A.get_gpu_storage(),x,b,r,z,p,diag_inv,its,it_block,block_size);
		 its +=it_block;

		 if(its%32==0 || its >= max_iter){
			 if(Blas::nrm2(N,r)<epsilon){
				 break;
			 }	
		 }

	 }
	 //Blas::extract(N,z,x0);
	 Blas::extract(N,x,x0);

	 clock_t time_finished = clock();

	 /// r = A*x
	 Blas::spmv(N,A,x,r);
	 // r = b - A*x
	 Blas::axpy ( N,-1.0,b,r) ;
	 Blas::scal ( N, -1.0,r) ;

	 printf("=====Calculation residue:\n");
	 printf("%f\n",Blas::nrm2(N,(Real*)r));
	 printf("niterations=%d\n",its);
	 printf("time=%d\n",time_finished-time_start);
	 printf("---------------------------Chebyshev iteration end-----------------------------\n\n");

#define ARRAY_DEALLOCATE(x) Blas::deallocate ( (void*)x )
	 ARRAY_DEALLOCATE(r);
	 ARRAY_DEALLOCATE(p);
	 ARRAY_DEALLOCATE(x);
	 ARRAY_DEALLOCATE(z);
	 ARRAY_DEALLOCATE(b);
	 ARRAY_DEALLOCATE(diag_inv);
#undef ARRAY_DEALLOCATE

	 Blas::deinitialize_matrix(A);

	 return (its<max_iter) ;
 }