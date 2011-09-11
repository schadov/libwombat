#pragma once
#include "StepSolverBase.h"
#include "implicit_step.h"

template<class RealT,template<class T> class Blas ,class Func,class Vector>
struct IrkFunctor{

	unsigned int nstages_;
	unsigned int nequations_;

	typedef BlasMatrix<Blas<RealT> > BlasMatrix;
	//typedef BlasVector<Blas<RealT> > BlasVector;

	const BlasMatrix& A_;
	const Vector &x_;
	const Func& F_;
	const RealT h_;
	const BlasVector<Blas<RealT> > & times_;

	mutable std::vector<std::vector<RealT> > s;

	IrkFunctor(
		unsigned int N,
		unsigned int nstages,
		const BlasMatrix& A,
		const Vector &x,
		const Func& F,
		RealT h,
		const BlasVector<Blas<RealT> >& times
		):
	A_(A),x_(x),h_(h),F_(F),times_(times)
	{
		nstages_ = nstages;
		nequations_ = N;

		s.resize(nstages);
		for (unsigned int m=0;m<nstages_;++m){
			s[m].resize(nequations_);
			s[m].assign(&x_[0],&x_[nequations_]);
		}
	}

	void operator()(RealT* in, RealT* out) const
	{
	
		for (unsigned int m=0;m<nstages_;++m)
		{
			for (unsigned int i=0;i<A_.get_dim();++i){
				for (unsigned int j=0;j<nequations_;++j){
					s[m][j] = A_(m,i)*in[j+i*nequations_];
				}
			}
	
			BlasVector<Blas<RealT> > t(nequations_);
			F_(times_[m],&s[m][0],t);
			Blas<RealT>::scal(nequations_,h_,t);
			//BlasVector t = h_ * F_(times_[m],s[m]);
			for (unsigned int i=0;i<nequations_;++i){
				t[i] -= in[i+m*nequations_];
				out[i+m*nequations_] = t[i];
			}

		}	

	}
};

template <template<class RealT> class Blas,class RealT,class Vector,class Func,class History>
struct IrkGeneric : StepSolverBase<Blas<RealT> >
	//public ImplicitStepSolverBase<RealT,Vector,Func,Blas,History,ThreeEightsFunctor<RealT,Blas,Func,History> >
{
protected:
	typedef BlasVector<Blas<RealT> > BlasVector;
	BlasVector ks_;

	BlasVector b_;
	BlasVector c_;

	BlasMatrix<Blas<RealT> > A_;

	unsigned int nstages_;

	void call_impl(unsigned int N,RealT t,RealT h, Vector &x, const Func &F,const History* history = 0){
		const unsigned int nequations = N;
		const unsigned int nstages = nstages_;

		BlasVector times(nstages);
		for (unsigned int i=0;i<nstages;++i){
			times[i] = t + h*c_[i];
		}

		IrkFunctor<RealT,Blas, Func,Vector> solver(nequations,nstages,A_,x,F,h,times);
		solve_newton<RealT,NewtonSolver,Blas,LUsolver,JacobyAuto>(nequations*nstages,solver,ks_,defaults::ImplicitMethodMaxNewtonSteps);

		for (unsigned int i=0;i<nstages;++i)
		{
			BlasVector t(nequations);
			for (unsigned int j = 0;j<nequations;++j){
				t[j] = ks_[j+i*nequations];
			}

			Blas<RealT>::scal(N,b_[i],t);
			Blas<RealT>::axpy(N,1.0,t,x);
		}

	}

public:
	void init(unsigned int N,RealT * init){
		unsigned int neqs = N;
		unsigned int nstages = nstages_;
		ks_.reset(neqs*nstages_);

		for (unsigned int m =0; m< nstages;++m){
			for (unsigned int j=0;j<neqs;++j){
				ks_[j+m*neqs] = init[j];
			}
		}
	}
	
	void call(unsigned int N,RealT t,RealT h, Vector &x, const Func &F,const History* history = 0){
		return call_impl(N,t,h,x,F,history);
	}

	static unsigned int history_length(){
		return 1;
	}
};


/************************************************************************/
/* Radau IA method of order 3, see Hairer et al. vol 2                  */
/************************************************************************/
template <template<class RealT> class Blas,class RealT,class Vector,class Func,class History>
struct IrkRadauIA3Step :
	public IrkGeneric<Blas,RealT,Vector,Func,History>
{

	IrkRadauIA3Step()
	{
		A_.reset(2);
		A_(0,0) = (RealT)(1./4);
		A_(0,1) = (RealT)(-1./4);

		A_(1,0) = (RealT)(1./4);
		A_(1,1) = (RealT)(5./12);

		b_.reset(2);
		b_[0] = (RealT)(1./4);
		b_[1] = (RealT)(3./4);

		c_.reset(2);
		c_[0] = 0;
		c_[1] = (RealT)(2./3);

		this->nstages_ = 2;
	}

};



/************************************************************************/
/* Gauss method of 4-th order, see hairer et al. vol 2                  */
/************************************************************************/
template <template<class RealT> class Blas,class RealT,class Vector,class Func,class History>
struct IrkGauss4Step :
	public IrkGeneric<Blas,RealT,Vector,Func,History>
{

	IrkGauss4Step()
	{
		A_.reset(2);
		A_(0,0) = (RealT)(1./4);
		A_(0,1) = (RealT)(-0.0386751345948129); // 1/4 - sqrt(3)/6

		A_(1,0) = (RealT)(0.538675134594813);  //  1/4 + sqrt(3)/6
		A_(1,1) = (RealT)(1./4);

		b_.reset(2);
		b_[0] = (RealT)(1./2);
		b_[1] = (RealT)(1./2);

		c_.reset(2);
		c_[0] = (RealT)0.211324865405187;				// 1./2 - sqrt(3.)/6;
		c_[1] = (RealT)0.538675134594813;				// 1./4 + sqrt(3.)/6;

		this->nstages_ = 2;
	}

};



/************************************************************************/
/* Lobatto IIIC method of order 6, see Hairer et al. vol 2              */
/************************************************************************/
template <template<class RealT> class Blas,class RealT,class Vector,class Func,class History>
struct IrkLobattoIIIC6Step :
	public IrkGeneric<Blas,RealT,Vector,Func,History>
{

	IrkLobattoIIIC6Step()
	{
		A_.reset(4);
		A_(0,0) = (RealT)(1./12);
		A_(0,1) = (RealT)(-0.186338998124982);	// sqrt(5)/12
		A_(0,2) = (RealT)0.186338998124982;			//sqrt(5)/12
		A_(0,3) = (RealT)(-1./12);

		A_(1,0) = (RealT)(1./12);
		A_(1,1) = (RealT)(1./4);	
		A_(1,2) = (RealT)-0.0942079307083088;			//((10 - 7*sqrt(5.))/60);
		A_(1,3) = (RealT)(-1./12);

		A_(2,0) = (RealT)(1./12);
		A_(2,1) = (RealT)(0.427541264041642);		// ((10 + 7*sqrt(5.))/60);
		A_(2,2) = (RealT)0.186338998124982;			//sqrt(5)/12
		A_(2,3) = (RealT)(-0.0372677996249965);		//-sqrt(5)/60

		A_(3,0) = (RealT)(1./12);
		A_(3,1) = (RealT)(5./12);	
		A_(3,2) = (RealT)(5./12);			
		A_(3,3) = (RealT)(1./12);


		b_.reset(4);
		b_[0]=(RealT)(1./12);
		b_[1]=(RealT)(5./12);
		b_[2]=(RealT)(5./12);
		b_[3]=(RealT)(1./12);

		c_.reset(4);
		c_[0] = 0;
		c_[1] = (RealT)0.276393202250021;				//(5 - sqrt(5.))/10.;
		c_[2] = (RealT)0.723606797749979;				//(5 + sqrt(5.))/10.;
		c_[3] = 1;

		this->nstages_ = 4;
	}

};