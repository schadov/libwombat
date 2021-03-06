#pragma once
#include <time.h>

template<class Matrix,class Real, class Blas> 
bool solve_bicgstab
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
	 ARRAY_ALLOCATE(nu_prev);
	 ARRAY_ALLOCATE(nu);
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
	 Real alpha_prev,alpha;
	 Real omega_prev,omega;

	 Real en=0;
	 const Real rho_coeff = 1;

	 printf("BICGSTAB started. initial residue:%f\n ",Blas::nrm2(N,(Real*)r));
	 
	 clock_t time_start = clock();

	 while ( /*cur_err > err &&*/ (int)its < max_iter)
	 {
		 //rho_prev = r~ <dot> r_prev
		 rho_prev =  Blas::dot(N,rhat,r_prev)*rho_coeff;

		 if(rho_prev==0){
			 printf("Cannot solve\n");
			 break;
		 }

		 const Real one_f = 1.0f;
		 if(its==0){
			 // p = rprev
			 Blas::copy(N,r_prev,p);
		 }
		 else{
			 //βi−1 = (ρi−1/ρi−2)(αi−1/ωi−1)
			 //p(i) = r(i−1) + βi−1(p(i−1) − ωi−1v(i−1))
			 beta_prev = (rho_prev / rho_prev2)*(alpha_prev/omega_prev);
			 //p = r_prev + beta_prev*(p_prev - omega_prev*nu_prev)
			 //==  r_prev+ beta_prev*p_prev - beta_prev*omega_prev*nu_prev

			 const Real momegaXbeta = -omega*beta_prev;
			 Blas::scal(N,momegaXbeta,nu_prev);
			 Blas::axpy(N,beta_prev,p_prev,nu_prev);
			 Blas::axpy(N,1.0f,r_prev,nu_prev);
			 std::swap(p,nu_prev);

			}

		 //solve M*phat = p
		 Blas::memberwise_mul( N, diag_inv, p, phat );

		 //nu = A*phat
		 Blas::spmv(N,A,phat,nu);

		 //α = rho_prev/(r <dot> nu)
		 alpha = rho_prev/(Blas::dot(N,rhat,nu)*rho_coeff);

		 //s = r(i−1) − αiv(i)	s = r_prev - alpha*nu
		 Blas::axpy(N,-alpha,nu,r_prev);
		 std::swap(s,r_prev);

		 //check the norm of s
		 if((en=Blas::nrm2(N,s))<epsilon){//0.003
			 Blas::axpy(N,alpha,phat,x_prev);
			 std::swap(x,x_prev);
			 break;
		 }

		 // 				solve Mˆs = s
		 Blas::memberwise_mul( N, diag_inv, s, shat );

		 // 				t = A*shat
		 Blas::spmv(N,A,shat,t);

		 // 				ωi = tT s/tT t
		 omega = Blas::dot(N,t,s)/Blas::dot(N,t,t);

		 // 				x(i) = x(i−1) + αi ˆp + ωiˆs		x = x_prev + alpha*phat+omega*shat;
		 Blas::scal(N,omega,shat);		//shat = omega*shat
		 Blas::axpy(N,alpha,phat,shat);		//shat = alpha*phat+shat
		 Blas::axpy(N,1.0f,shat,x_prev);	//x_prev = shat+x_prev

		 // 				r(i) = s − ωt
		 Blas::axpy(N,-omega,t,s);		//s = -omega*t + s
		 std::swap(r,s);

		 if(omega==0)
			 break;

		 Real r_norm = Blas::nrm2(N,r);
		 //printf("r_norm = %10.10f\n",r_norm);
		 if(r_norm<epsilon){
			 std::swap(x,x_prev);
			 break;
		 }

		 rho_prev2 = rho_prev;
		 alpha_prev = alpha;
		 omega_prev = omega;
		 std::swap(r_prev,r);
		 std::swap(p_prev,p);
		 std::swap(nu_prev,nu);

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
	 printf("---------------------------BICGSTAB end-----------------------------\n\n");

#define CUDA_ARRAY_DEALLOCATE(x) Blas::deallocate ( (void*)x )
	 CUDA_ARRAY_DEALLOCATE(rhat);
	 CUDA_ARRAY_DEALLOCATE(x_prev);
	 CUDA_ARRAY_DEALLOCATE(r_prev);
	 CUDA_ARRAY_DEALLOCATE(nu_prev);
	 CUDA_ARRAY_DEALLOCATE(nu);
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