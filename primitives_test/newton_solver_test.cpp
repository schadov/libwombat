#include <gtest/gtest.h>
#include "../primitives/types.h"
#include "../primitives/lu.h"
#include "../primitives/lin_solve.h"
#include "../primitives/RefBlas.h"
#include "../primitives/newton.h"
#include "../primitives/jacobian.h"


void simple_nonlinear_function(double* in, double* out){
	out[0] = in[0]*in[0] - 256;
}




TEST(NewtonSolverTest,SolvesTestEq1){


	BlasVector<RefBlas<double> > x(1);
	x[0] = 2;
	solve_newton<double,NewtonSolver,RefBlas,LUsolver,JacobyAuto>(1,simple_nonlinear_function,x,10);

	const double epsilon = 0.00001;
	for (unsigned int i=0;i<1;++i)
	{
		EXPECT_EQ(std::abs(16 - x[0])<epsilon, true);
	}


}


/*
TEST(NewtonSimplifiedSolverTest,SolvesTestEq1){


	BlasVector<RefBlas<double> > x(1);
	x[0] = 8;
	solve_newton<double,NewtonSimplifiedSolver,RefBlas,LUsolver,JacobyAuto>(1,simple_nonlinear_function,x,20);

	const double epsilon = 0.00001;
	for (unsigned int i=0;i<1;++i)
	{
		EXPECT_EQ(std::abs(16 - x[0])<epsilon, true);
	}


}
*/
