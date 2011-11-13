#pragma once
#include "StepSolverBase.h"
#include "implicit_step.h"

template<class RealT,template<class T> class Blas ,class Func,class Vector>
struct SDIrkFunctor{

	unsigned int nstages_;
	unsigned int nequations_;

	//typedef BlasMatrix<Blas<RealT> > BlasMatrix;
	//typedef BlasVector<Blas<RealT> > BlasVector;

	const RealT* a_;
	const Vector &x_;
	const Vector &k_;
	const Func& F_;
	const RealT h_;
	const RealT t_;
	const unsigned int i_;

	mutable BlasVector<Blas<RealT> > s_;

	SDIrkFunctor(
		unsigned int N,
		const Vector& a,
		const Vector& k,
		const Vector& x,
		unsigned int i,
		const Func& F,
		RealT h,
		RealT t
		):
	a_(a),x_(x),k_(k),h_(h),F_(F),t_(t),i_(i)
	{
		nequations_ = N;
		s_.reset(N);
		/*std::vector<RealT> slocal(N);
		Blas::extract(N,x,slocal);

		for(unsigned int i=0;i<i_;++i){
			for (unsigned int j=0;j<N;++j){
				slocal[j]+=a_[i]*k_[i][j];
			}
		}
		Blas::set(N,&slocal[0],s);*/
		Blas<RealT>::copy(N,x,s_);
		for(unsigned int i=0;i<i_;++i){
			Blas<RealT>::axpy(N,a_[i],k_,s_);
		}

	}
	
	void operator()(RealT* in, RealT* out) const
	{
		BlasVector<Blas<RealT> > s(nequations_);
		Blas<RealT>::copy(nequations_,s_,s);
		Blas<RealT>::axpy(nequations_,a_[i_],in,s);
		BlasVector<Blas<RealT> > fts(nequations_);
		F_(t_,s,fts);

		Blas<RealT>::copy(nequations_,in,out);
		Blas<RealT>::scal(nequations_,-1.0,out);
		Blas<RealT>::axpy(nequations_,h_,fts,out);
	}
};



template <template<class RealT> class Blas,class RealT,class Vector,class Func,class History>
struct SDirkGeneric : StepSolverBase<Blas<RealT> >
{
protected:
	typedef BlasVector<Blas<RealT> > BlasVector;

	std::vector<RealT> b_;
	std::vector<RealT> c_;

	std::vector<std::vector<RealT> > A_;

	unsigned int nstages_;

	void call_impl(unsigned int N,RealT t,RealT h, Vector &x, const Func &F,const History* history = 0){
		const unsigned int nequations = N;
		const unsigned int nstages = nstages_;

		typedef Blas<RealT> MyBlas;

		VectorDeque<MyBlas> k(N,nstages_);

		int first_idx = 0;	//first nonzero Butcher tableu row index
		BlasMatrix<MyBlas> J(N);

		//k[0] = h*F(dbegin,x);
		RealT * p = k.get_vector_pointer(0);
		F(t,x,p);
		MyBlas::scal(N,h,p);

		for (unsigned int m = 0; m < nstages_; ++m)	//for each r-k stage...
		{
			//if the first Butcher's tableu row is zero, we can calculate k0 explicitly
			if(m==0 && A_[0][0]==0){
				RealT *f = k.get_vector_pointer(0);
				F(t,x,f);
				MyBlas::scal(N,h,f);
				first_idx = 1;
			}
			else{	//for nonzero rows of Butcher tableu
				RealT ti = t + h*c_[m];		//time for the current stage

				const int q = m!=0?m-1:m;
				RealT *pkm = k.get_vector_pointer(m);
				if(q!=m){
					RealT *pkm1 = k.get_vector_pointer(m-1);
					MyBlas::copy(N,pkm1,pkm);
				}

				//SDirk_eq eq(a[m],k,x,m,F,h,ti);	//m-th equation object
				typedef SDIrkFunctor<RealT,Blas, Func,Vector> SDirkFunctorType;
				SDirkFunctorType eq(N,&A_[m][0],k.get_vector_pointer(m),x,m,F,h,ti);

				if(m==first_idx){	//if we haven't done it before, calc the Jacoby Matrix
					JacobyAuto<MyBlas,BlasMatrix<MyBlas>,Vector,SDirkFunctorType> jacoby_calculator;
					jacoby_calculator.call(N,eq,x,J);

					LinearSolverUtils<LUsolver,MyBlas,RealT,BlasMatrix<MyBlas>,Vector> lu;
					lu.decompose(N,J);
					//rmatrix A = Jacoby(eq,x);
					//lu_decompose(A,L,U);
				}

				//solve the m-th equation
				//k[m] = newton_simplified_ex(eq,default_lin_solver,k[m!=0?m-1:m],L,U);
				NewtonSimplifiedExSolver<
					MyBlas,
					RealT,
					BlasMatrix<MyBlas>,
					Vector,
					SDirkFunctorType,
					LUsolver,
					JacobyAuto<MyBlas, BlasMatrix<MyBlas>, Vector,SDirkFunctorType >
				> nsolver;
				
				
				nsolver.call(N,eq,J,pkm,defaults::ImplicitMethodMaxNewtonSteps);
			}
		}
		BlasVector tmp(N);
		for (unsigned int m = 0;m<nstages_;++m){
			RealT* km = k.get_vector_pointer(m);
			MyBlas::copy(N,km,tmp);
			MyBlas::axpy(N,b_[m],tmp,x);
		}

		RealT* k0 = k.get_vector_pointer(0);
		RealT* klast = k.get_vector_pointer(nstages_-1);
		MyBlas::copy(N,k0,klast);

	}

public:

	void call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F,const History* history = 0){
		return call_impl(N,t,h,x,F,history);
	}

	static unsigned int history_length(){
		return 1;
	}
};


/************************************************************************/
/* SDIRK-3 method  (Hairer et al, vol 1, table 7.2)                                                                    */
/************************************************************************/
template <template<class RealT> class Blas,class RealT,class Vector,class Func,class History>
struct SDirk3Step :
	public SDirkGeneric<Blas,RealT,Vector,Func,History>
{

	SDirk3Step()
	{
		//A
		std::vector<std::vector<RealT> > as;
		std::vector<RealT> tmp(2);
		tmp[0]=static_cast<RealT>(((3+sqrt(3.))/6.));
		tmp[1]=(0.0);
		as.push_back(tmp);

		tmp[0]=static_cast<RealT>((-sqrt(3.)/3));
		tmp[1]=static_cast<RealT>(((3+sqrt(3.))/6));
		as.push_back(tmp);

		//b
		std::vector<RealT> b(2);
		b[0] = 0.5;
		b[1] = 0.5;
		//c
		std::vector<RealT> c(2);
		c[0]= static_cast<RealT>((3+sqrt(3.))/6.);
		c[1]= static_cast<RealT>((3-sqrt(3.))/6.);

		this->c_ = c;
		this->b_= b;
		this->A_ = as;
		this->nstages_ = 2;
	}

};


/************************************************************************/
/* NT1 method  of some obscure origin                                                                   */
/************************************************************************/
template <template<class RealT> class Blas,class RealT,class Vector,class Func,class History>
struct SDirkNT1Step :
	public SDirkGeneric<Blas,RealT,Vector,Func,History>
{

	SDirkNT1Step()
	{
		this->nstages_ = 4;
		std::vector<std::vector<RealT> > as;
		std::vector<RealT> tmp(4,0.0);
		as.push_back(tmp);

		tmp[0]=static_cast<RealT>(5./12);
		tmp[1]=static_cast<RealT>(5./12);
		as.push_back(tmp);


		tmp[0]=static_cast<RealT>(95./588);
		tmp[1]=static_cast<RealT>(-5./49);
		tmp[2]=static_cast<RealT>(5./12);
		as.push_back(tmp);

		tmp[0]=static_cast<RealT>(59./600);
		tmp[1]=static_cast<RealT>(-31./75);
		tmp[2]=static_cast<RealT>(539./600);
		tmp[3]=static_cast<RealT>(5./12);

		as.push_back(tmp);		

		const int s = this->nstages_;
		//b
		std::vector<RealT> b(s);
		b[0] = static_cast<RealT>(59./600);
		b[1] = static_cast<RealT>(-31./75);
		b[2] = static_cast<RealT>(539./600);
		b[3] = static_cast<RealT>(5./12);

		/*std::vector<RealT> b(s);
		b[0] = static_cast<RealT>(-37./600);
		b[1] = static_cast<RealT>(-31./75);
		b[2] = static_cast<RealT>(1813./6600);
		b[3] = static_cast<RealT>(37./132);*/

		//c
		std::vector<RealT> c(s);
		c[0] = 0;
		c[1] = static_cast<RealT>(5./6);
		c[2] = static_cast<RealT>(10./21);
		c[3] = static_cast<RealT>(1.);


		this->c_ = c;
		this->b_= b;
		this->A_ = as;
		
	}

};



template <template<class RealT> class Blas,class RealT,class Vector,class Func,class History>
struct SDirkLStableStep :
	public SDirkGeneric<Blas,RealT,Vector,Func,History>
{

	SDirkLStableStep()
	{
		this->nstages_ = 5;
		std::vector<std::vector<RealT> > as;
		std::vector<RealT> tmp(this->nstages_,0.0);
		tmp[0] = static_cast<RealT>(1./4);
		as.push_back(tmp);

		tmp[0]=static_cast<RealT>(1./2);
		tmp[1]=static_cast<RealT>(1./4);
		as.push_back(tmp);


		tmp[0]=static_cast<RealT>(17./50);
		tmp[1]=static_cast<RealT>(-1./25);
		tmp[2]=static_cast<RealT>(1./4);
		as.push_back(tmp);

		tmp[0]=static_cast<RealT>(371./1360);
		tmp[1]=static_cast<RealT>(-137./2720);
		tmp[2]=static_cast<RealT>(15./544);
		tmp[3]=static_cast<RealT>(1./4);
		as.push_back(tmp);		

		tmp[0]=static_cast<RealT>(25./24);
		tmp[1]=static_cast<RealT>(-49./48);
		tmp[2]=static_cast<RealT>(125./16);
		tmp[3]=static_cast<RealT>(-85./12);
		tmp[4]=static_cast<RealT>(1./4);
		as.push_back(tmp);	

		const int s = this->nstages_;
		//b
		std::vector<RealT> b(s);
		b[0] = static_cast<RealT>(25./24);
		b[1] = static_cast<RealT>(-49./48);
		b[2] = static_cast<RealT>(125./16);
		b[3] = static_cast<RealT>(-85./12);
		b[4] = static_cast<RealT>(1./4);

		/*std::vector<RealT> b(s);
		b[0] = static_cast<RealT>(-37./600);
		b[1] = static_cast<RealT>(-31./75);
		b[2] = static_cast<RealT>(1813./6600);
		b[3] = static_cast<RealT>(37./132);*/

		//c
		std::vector<RealT> c(s);
		c[0] = static_cast<RealT>(1./4);;
		c[1] = static_cast<RealT>(3./4);
		c[2] = static_cast<RealT>(11./20);
		c[3] = static_cast<RealT>(1./2);
		c[4] = static_cast<RealT>(1.);


		this->c_ = c;
		this->b_= b;
		this->A_ = as;
		
	}

};