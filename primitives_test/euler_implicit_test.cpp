#include <gtest/gtest.h>
#include "../primitives/types.h"
#include "../primitives/ode_solver.h"
#include "../primitives/euler_implicit.h"
#include "../primitives/RefBlas.h"

#include "test_functions.h"



TEST(EulerImplicitTest,SolvesTestEq1){
	test_equation1<EulerImplicitStep>();
}

TEST(EulerImplicitTest,SolvesTestEq2){
	test_equation2<EulerImplicitStep>();
}

//////////////////////////////////////////////////////////////////////////

TEST(EulerTrapezoidTest,SolvesTestEq1){
	test_equation1<EulerTrapezoidStep>();
}

TEST(EulerTrapezoidTest,SolvesTestEq2){
	test_equation2<EulerTrapezoidStep>();
}

//////////////////////////////////////////////////////////////////////////


TEST(SimpsonImplicitTest,SolvesTestEq1){
	test_equation1<SimpsonImplicitStep>();
}

TEST(SimpsonImplicitTest,SolvesTestEq2){
	test_equation2<SimpsonImplicitStep>();
}

//////////////////////////////////////////////////////////////////////////

TEST(TickImplicitTest,SolvesTestEq1){
	test_equation1<TickImplicitStep>();
}

TEST(TickImplicitTest,SolvesTestEq2){
	test_equation2<TickImplicitStep>();
}

//////////////////////////////////////////////////////////////////////////

TEST(ThreeEightsImplicitTest,SolvesTestEq1){
	test_equation1<ThreeEightsImplicitStep>();
}

TEST(ThreeEightsImplicitTest,SolvesTestEq2){
	test_equation2<ThreeEightsImplicitStep>();
}