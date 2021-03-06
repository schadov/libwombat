#pragma once
#include <time.h>

template<class Matrix,class Real, class Blas> 
bool solve_cg
(
 Matrix &A,
 const Real* b0,
 Real* x0,
 const unsigned int N,
 const unsigned int max_iter,
 const Real epsilon,
 const unsigned int block_size,
 Blas 
 )
{

	const Real * diag_matrix = A.diag();

	Blas::init(N);

#define ARRAY_ALLOCATE(x) Real *x=0;Blas::allocate<Real> ( N+4, x )
	ARRAY_ALLOCATE(x_prev);
	ARRAY_ALLOCATE(r_prev);
	ARRAY_ALLOCATE(z_prev);
	ARRAY_ALLOCATE(q);
	ARRAY_ALLOCATE(p);
	ARRAY_ALLOCATE(p_prev);
	ARRAY_ALLOCATE(x);
	ARRAY_ALLOCATE(phat);
	ARRAY_ALLOCATE(shat);
	ARRAY_ALLOCATE(s);
	ARRAY_ALLOCATE(t);
	ARRAY_ALLOCATE(r);
	ARRAY_ALLOCATE(rtmp);

	ARRAY_ALLOCATE(gpu_Ad);
	ARRAY_ALLOCATE(diag_inv);
	ARRAY_ALLOCATE(tmp);
	ARRAY_ALLOCATE(b);
	ARRAY_ALLOCATE(d);
#undef ARRAY_ALLOCATE


	Blas::initialize_matrix(A);			

	// building the Jacobi preconditionner
	std::vector<Real> cpu_diag_inv ( N+4) ;
	for(unsigned int i=0; i<N; i++) {
		cpu_diag_inv[i] = (Real)((i >= N || diag_matrix[i] == 0.0) ? 1.0 : 1.0 / diag_matrix[i]) ;
	}

	Blas::set(N,&cpu_diag_inv[0],diag_inv);

	Blas::set(N,x0,x);

	Blas::set(N,b0,b);

	const int one = 1;

	Blas::copy(N,x,x_prev);

	unsigned int its=0;

	// r = A*x
	Blas::spmv(N,A,x,r);

	// r = b - A*x
	Blas::axpy(N,-1.0,b,r);
	Blas::scal(N,-1.0,r);


	//r_prev = r
	Blas::copy(N,r,r_prev);

	// d = M-1 * r
	Blas::memberwise_mul ( N, diag_inv, r, d ) ;

	// cur_err = rT*d
	Real cur_err = Blas::dot(N,r,d);

	// err = cur_err
	Real err = (Real)(cur_err * epsilon * epsilon) ;

	Real  rho_prev,rho_prev2;
	Real beta_prev=1e6;
	Real alpha;

	Real en=0;
	const Real rho_coeff = 1;

	printf("CG started. initial residue:%f\n ",Blas::nrm2(N,(Real*)r));

	clock_t time_start = clock();

	while ( /*cur_err > err &&*/ (int)its < max_iter)
	{

		Blas::memberwise_mul( N, diag_inv, r_prev, z_prev );

		rho_prev = Blas::dot(N,r_prev,z_prev);

		if(its==0){
			Blas::copy(N,z_prev,p);
		}
		else{
			beta_prev = (rho_prev / rho_prev2);

			//p(i)= z(i−1)+ βi−1p(i−1)
			Blas::axpy(N,beta_prev,p_prev,z_prev);
			std::swap(p,z_prev);
		}

		//q(i)= Ap(i)
		Blas::spmv(N,A,p,q);

		alpha = rho_prev/(Blas::dot(N,p,q));

		Blas::axpy(N,alpha,p,x_prev);

		Blas::axpy(N,-alpha,q,r_prev);

		Real r_norm = Blas::nrm2(N,r_prev);
		if(r_norm<epsilon){
			std::swap(x,x_prev);
			break;
		}

		rho_prev2 = rho_prev;
		//std::swap(p,p_prev);

		its++;

	}

	if ( its==max_iter ) {
		std::swap(x,x_prev);
	}

	//cublasGetVector ( N, 4, (Real*)x, 1, x0.data(), 1 ) ;
	//memcpy(x0.data(),x,N*sizeof(Real));
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
	printf("---------------------------CG end-----------------------------\n\n");

#define CUDA_ARRAY_DEALLOCATE(x) Blas::deallocate ( (void*)x )
	CUDA_ARRAY_DEALLOCATE(x_prev);
	CUDA_ARRAY_DEALLOCATE(r_prev);
	CUDA_ARRAY_DEALLOCATE(z_prev);
	CUDA_ARRAY_DEALLOCATE(q);
	CUDA_ARRAY_DEALLOCATE(p);
	CUDA_ARRAY_DEALLOCATE(p_prev);
	CUDA_ARRAY_DEALLOCATE(x);
	CUDA_ARRAY_DEALLOCATE(phat);
	CUDA_ARRAY_DEALLOCATE(shat);
	CUDA_ARRAY_DEALLOCATE(s);
	CUDA_ARRAY_DEALLOCATE(t);
	CUDA_ARRAY_DEALLOCATE(r);

	CUDA_ARRAY_DEALLOCATE(gpu_Ad);
	CUDA_ARRAY_DEALLOCATE(diag_inv);
	CUDA_ARRAY_DEALLOCATE(tmp);
	CUDA_ARRAY_DEALLOCATE(b);
	CUDA_ARRAY_DEALLOCATE(d);
#undef CUDA_ARRAY_DEALLOCATE

	Blas::deinitialize_matrix(A);

	return (its<max_iter) ;
}