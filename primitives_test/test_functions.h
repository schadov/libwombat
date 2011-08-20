#pragma once

//
inline void equation(float t, const float* x, float* F){
	F[0] = t*2;
}

inline void sin_eq(double t, const double* x, double* F)
{
	F[0] = 3*sin(4*t);
}

inline double sin_eq_analytic(double tm)
{
	return -3./4*cos(4*tm);
}


template<
template <template<class RealT> class Blas,class RealT,class Vector,class Func,class History> class Solver
> void test_equation1(){
	const unsigned int ndim = 1;
	Vector<float> init(ndim);
	init[0] = 0.0;
	Vector<float> result(ndim);
	const float step = 0.01f;
	const float dend = 4.0f;

	solve_fixedstep<float,RefBlas,Solver>(ndim,
		0.0f,dend,
		step,
		equation,
		init.data(),
		result.data()
		);

	const float epsilon = result[0] * step;
	EXPECT_EQ(epsilon > 0,true);
	EXPECT_EQ(std::abs(dend*dend - result[0])<epsilon, true);
}

template<
template <template<class RealT> class Blas,class RealT,class Vector,class Func,class History> class Solver
> void test_equation2(){
	const unsigned int ndim = 1;
	Vector<double> init(ndim);
	init[0] = sin_eq_analytic(0.0);
	Vector<double> result(ndim);
	const double step = 0.001;
	const double dend = 3.0;

	solve_fixedstep<double,RefBlas,EulerImplicitStep>(ndim,
		0.0f,dend,
		step,
		sin_eq,
		init.data(),
		result.data()
		);

	const double epsilon = 0.01;
	EXPECT_EQ(epsilon > 0,true);
	EXPECT_EQ(std::abs(sin_eq_analytic(dend) - result[0])<epsilon, true);
}
