#include <gtest/gtest.h>
#include "../primitives/types.h"
#include "../primitives/ode_solver.h"
#include "../primitives/euler_implicit.h"
#include "../primitives/RefBlas.h"

#include "test_functions.h"


TEST(EulerImplicitTest,SolvesTestEq1){
	const unsigned int ndim = 1;
	Vector<float> init(ndim);
	init[0] = 0.0;
	Vector<float> result(ndim);
	const float step = 0.01f;
	const float dend = 4.0f;

	solve_fixedstep<float,RefBlas,EulerImplicitStep>(ndim,
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

TEST(EulerImplicitTest,SolvesTestEq2){
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

//////////////////////////////////////////////////////////////////////////

TEST(EulerTrapezoidTest,SolvesTestEq1){
	const unsigned int ndim = 1;
	Vector<float> init(ndim);
	init[0] = 0.0;
	Vector<float> result(ndim);
	const float step = 0.01f;
	const float dend = 4.0f;

	solve_fixedstep<float,RefBlas,EulerTrapezoidStep>(ndim,
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

TEST(EulerTrapezoidTest,SolvesTestEq2){
	const unsigned int ndim = 1;
	Vector<double> init(ndim);
	init[0] = sin_eq_analytic(0.0);
	Vector<double> result(ndim);
	const double step = 0.001;
	const double dend = 3.0;

	solve_fixedstep<double,RefBlas,EulerTrapezoidStep>(ndim,
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

//////////////////////////////////////////////////////////////////////////


TEST(SimpsonImplicitTest,SolvesTestEq1){
	const unsigned int ndim = 1;
	Vector<float> init(ndim);
	init[0] = 0.0;
	Vector<float> result(ndim);
	const float step = 0.01f;
	const float dend = 4.0f;

	solve_fixedstep<float,RefBlas,SimpsonImplicitStep>(ndim,
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

TEST(SimpsonImplicitTest,SolvesTestEq2){
	const unsigned int ndim = 1;
	Vector<double> init(ndim);
	init[0] = sin_eq_analytic(0.0);
	Vector<double> result(ndim);
	const double step = 0.001;
	const double dend = 3.0;

	solve_fixedstep<double,RefBlas,SimpsonImplicitStep>(ndim,
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