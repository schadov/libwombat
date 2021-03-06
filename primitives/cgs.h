#pragma once
#include <time.h>

template<class Matrix,class Real, class Blas> 
bool solve_cgs
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
	ARRAY_ALLOCATE(rhat);
	ARRAY_ALLOCATE(x_prev);
	ARRAY_ALLOCATE(r_prev);
	ARRAY_ALLOCATE(nu);
	ARRAY_ALLOCATE(p);
	ARRAY_ALLOCATE(p_prev);
	ARRAY_ALLOCATE(x);
	ARRAY_ALLOCATE(phat);
	ARRAY_ALLOCATE(qhat);
	ARRAY_ALLOCATE(q);
	ARRAY_ALLOCATE(u);
	ARRAY_ALLOCATE(r);
	ARRAY_ALLOCATE(q_prev);

	ARRAY_ALLOCATE(gpu_Ad);
	ARRAY_ALLOCATE(diag_inv);
	ARRAY_ALLOCATE(b);
	ARRAY_ALLOCATE(d);
	ARRAY_ALLOCATE(uhat);
	ARRAY_ALLOCATE(tmp);
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

	//rhat = r
	Blas::copy(N,r,rhat);

	//r_prev = r
	Blas::copy(N,r,r_prev);

	// d = M-1 * r
	Blas::memberwise_mul ( N, diag_inv, r, d ) ;

	// cur_err = rT*d
	Real cur_err = Blas::dot(N,r,d);

	// err = cur_err
	Real err = (Real)(cur_err * epsilon * epsilon) ;

	Real  rho_prev,rho_prev2;
	Real beta_prev;
	Real alpha;

	Real en=0;
	const Real rho_coeff = 1;

	printf("CGS started. initial residue:%f\n ",Blas::nrm2(N,(Real*)r));

	clock_t time_start = clock();

	while ( /*cur_err > err &&*/ (int)its < max_iter)
	{
		rho_prev = Blas::dot(N,rhat,r_prev);
		if(rho_prev==0){
			printf("Cannot solve\n");
			break;
		}
		if(its==0){
			Blas::copy(N,r_prev,u);	//u = r
			Blas::copy(N,u,p);	//u = r
		}
		else{
			beta_prev = (rho_prev / rho_prev2);

			//u_i = r_i-1 + \beta_(i-1) * q_(i-1)
			Blas::copy(N,r_prev,u);
			Blas::axpy(N,beta_prev,q_prev,u);
			//std::swap(u,r_prev);

			//pi = ui + b_(i-1)*( q_(i-1) + b_(i-1)*p_(i-1))
			Blas::copy(N,q_prev,tmp);
			Blas::axpy(N,beta_prev,p_prev,tmp);
			Blas::copy(N,u,p);
			Blas::axpy(N,beta_prev,tmp,p);
			//std::swap(p,u);	

		}
		Blas::memberwise_mul( N, diag_inv, p, phat );

		Blas::spmv(N,A,phat,nu);

		alpha = rho_prev/(Blas::dot(N,rhat,nu));

		Blas::copy(N,u,q);
		Blas::axpy(N,-alpha,nu,q);

		Blas::axpy(N,1.0,q,u);

		Blas::memberwise_mul( N, diag_inv, u, uhat );

		// 				x(i) = x(i−1) + αi*u^	
		Blas::axpy(N,alpha,uhat,x_prev);	//x_prev = uhat+x_prev

		Blas::spmv(N,A,uhat,qhat);

		//ri = ri-1 - ai*q^
		Blas::axpy(N,-alpha,qhat,r_prev);

		Real r_norm = Blas::nrm2(N,r_prev);
		//printf("r_norm = %10.10f\n",r_norm);
		if(r_norm<epsilon){
			std::swap(x,x_prev);
			break;
		}

		rho_prev2 = rho_prev;

		//std::swap(r_prev,r);
		std::swap(p_prev,p);
		std::swap(q_prev,q);

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
	printf("---------------------------CGS end-----------------------------\n\n");

#define CUDA_ARRAY_DEALLOCATE(x) Blas::deallocate ( (void*)x )
	CUDA_ARRAY_DEALLOCATE(rhat);
	CUDA_ARRAY_DEALLOCATE(x_prev);
	CUDA_ARRAY_DEALLOCATE(r_prev);
	CUDA_ARRAY_DEALLOCATE(nu);
	CUDA_ARRAY_DEALLOCATE(p);
	CUDA_ARRAY_DEALLOCATE(p_prev);
	CUDA_ARRAY_DEALLOCATE(x);
	CUDA_ARRAY_DEALLOCATE(phat);
	CUDA_ARRAY_DEALLOCATE(qhat);
	CUDA_ARRAY_DEALLOCATE(q);
	CUDA_ARRAY_DEALLOCATE(u);
	CUDA_ARRAY_DEALLOCATE(r);

	CUDA_ARRAY_DEALLOCATE(gpu_Ad);
	CUDA_ARRAY_DEALLOCATE(diag_inv);
	CUDA_ARRAY_DEALLOCATE(q_prev);
	CUDA_ARRAY_DEALLOCATE(b);
	CUDA_ARRAY_DEALLOCATE(d);
	CUDA_ARRAY_DEALLOCATE(uhat);
	CUDA_ARRAY_DEALLOCATE(tmp);
#undef CUDA_ARRAY_DEALLOCATE

	Blas::deinitialize_matrix(A);

	return (its<max_iter) ;
}