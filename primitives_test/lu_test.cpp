#include <gtest/gtest.h>
#include "../primitives/types.h"
#include "../primitives/lu.h"
#include "../primitives/lin_solve.h"
#include "../primitives/RefBlas.h"

#include "test_linear_systems.h"


TEST(LUTest,SolvesTestEq1){


	BlasMatrix<RefBlas<double> > m1(3);
	m1(0,0) = 3.0;
	m1(1,0) = 2.0;
	m1(2,0) = -1.0;

	m1(0,1) = 2.0;
	m1(1,1) = -2.0;
	m1(2,1) = 0.5;

	m1(0,2) = -1.0;
	m1(1,2) = 4.0;
	m1(2,2) = -1.0;

	BlasVector<RefBlas<double> > b(3);
	b[0] = 1;
	b[1] = -2;
	b[2] = 0;

	BlasVector<RefBlas<double> > test(3);
	test[0] = 1;
	test[1] = -2;
	test[2] = -2;

	BlasVector<RefBlas<double> > res(3);

	solve_linear<double,RefBlas,LUsolver>(3,m1,b,res);


	const double epsilon = 0.00001;
	for (unsigned int i=0;i<3;++i)
	{
		EXPECT_EQ(std::abs(test[0]- res[0])<epsilon, true);
	}

	
}


TEST(LUTest,SolvesTest10Equations){

	const unsigned int N = 10;
	BlasMatrix<RefBlas<double> > A(N);
	BlasVector<RefBlas<double> > b(N);
	BlasVector<RefBlas<double> > roots(N);
	generate_linear_system<double>(N,A,b,roots,31337);
	
	BlasVector<RefBlas<double> > res(N);

	solve_linear<double,RefBlas,LUsolver>(N,A,b,res);


	const double epsilon = 0.00001;
	for (unsigned int i=0;i<N;++i){
		EXPECT_EQ(std::abs(roots[0]- res[0])<epsilon, true);
	}
}


TEST(LUTest,SolvesTest100Equations){

	const unsigned int N = 100;
	BlasMatrix<RefBlas<double> > A(N);
	BlasVector<RefBlas<double> > b(N);
	BlasVector<RefBlas<double> > roots(N);
	generate_linear_system<double>(N,A,b,roots,511);

	BlasVector<RefBlas<double> > res(N);

	solve_linear<double,RefBlas,LUsolver>(N,A,b,res);


	const double epsilon = 0.00001;
	for (unsigned int i=0;i<N;++i){
		EXPECT_EQ(std::abs(roots[0]- res[0])<epsilon, true);
	}
}

#ifndef _DEBUG

TEST(LUTest,SolvesTest1000Equations){

	const unsigned int N = 1000;
	BlasMatrix<RefBlas<double> > A(N);
	BlasVector<RefBlas<double> > b(N);
	BlasVector<RefBlas<double> > roots(N);
	generate_linear_system<double>(N,A,b,roots,511);

	BlasVector<RefBlas<double> > res(N);

	solve_linear<double,RefBlas,LUsolver>(N,A,b,res);


	const double epsilon = 0.00001;
	for (unsigned int i=0;i<N;++i){
		EXPECT_EQ(std::abs(roots[0]- res[0])<epsilon, true);
	}
}
#endif


//////////////////////////////////////////////////////////////////////////

TEST(LUPTest,SolvesTestEq1){


	BlasMatrix<RefBlas<double> > m1(3);
	m1(0,0) = 3.0;
	m1(1,0) = 2.0;
	m1(2,0) = -1.0;

	m1(0,1) = 2.0;
	m1(1,1) = -2.0;
	m1(2,1) = 0.5;

	m1(0,2) = -1.0;
	m1(1,2) = 4.0;
	m1(2,2) = -1.0;

	BlasVector<RefBlas<double> > b(3);
	b[0] = 1;
	b[1] = -2;
	b[2] = 0;

	BlasVector<RefBlas<double> > test(3);
	test[0] = 1;
	test[1] = -2;
	test[2] = -2;

	BlasVector<RefBlas<double> > res(3);

	solve_linear<double,RefBlas,LUPsolver>(3,m1,b,res);


	const double epsilon = 0.00001;
	for (unsigned int i=0;i<3;++i)
	{
		EXPECT_EQ(std::abs(test[0]- res[0])<epsilon, true);
	}


}


TEST(LUPTest,SolvesTest10Equations){

	const unsigned int N = 10;
	BlasMatrix<RefBlas<double> > A(N);
	BlasVector<RefBlas<double> > b(N);
	BlasVector<RefBlas<double> > roots(N);
	generate_linear_system<double>(N,A,b,roots,31337);

	BlasVector<RefBlas<double> > res(N);

	solve_linear<double,RefBlas,LUPsolver>(N,A,b,res);


	const double epsilon = 0.00001;
	for (unsigned int i=0;i<N;++i){
		EXPECT_EQ(std::abs(roots[0]- res[0])<epsilon, true);
	}
}


TEST(LUPTest,SolvesTest100Equations){

	const unsigned int N = 100;
	BlasMatrix<RefBlas<double> > A(N);
	BlasVector<RefBlas<double> > b(N);
	BlasVector<RefBlas<double> > roots(N);
	generate_linear_system<double>(N,A,b,roots,511);

	BlasVector<RefBlas<double> > res(N);

	solve_linear<double,RefBlas,LUPsolver>(N,A,b,res);


	const double epsilon = 0.00001;
	for (unsigned int i=0;i<N;++i){
		EXPECT_EQ(std::abs(roots[0]- res[0])<epsilon, true);
	}
}

#ifndef _DEBUG

TEST(LUPTest,SolvesTest1000Equations){

	const unsigned int N = 1000;
	BlasMatrix<RefBlas<double> > A(N);
	BlasVector<RefBlas<double> > b(N);
	BlasVector<RefBlas<double> > roots(N);
	generate_linear_system<double>(N,A,b,roots,511);

	BlasVector<RefBlas<double> > res(N);

	solve_linear<double,RefBlas,LUPsolver>(N,A,b,res);


	const double epsilon = 0.00001;
	for (unsigned int i=0;i<N;++i){
		EXPECT_EQ(std::abs(roots[0]- res[0])<epsilon, true);
	}
}
#endif