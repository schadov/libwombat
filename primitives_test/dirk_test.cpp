#include <gtest/gtest.h>
#include "../primitives/types.h"
#include "../primitives/ode_solver.h"
#include "../primitives/RefBlas.h"
#include "../primitives/dirk.h"

#include "test_functions.h"

TEST(SDirk3Test,SolvesTestEq1){
	test_equation1<SDirk3Step>();
}

TEST(SDirk3Test,SolvesTestEq2){
	test_equation2<SDirk3Step>();
}

TEST(SDirkNT1Test,SolvesTestEq1){
	test_equation1<SDirkNT1Step>();
}

TEST(SDirkNT1Test,SolvesTestEq2){
	test_equation2<SDirkNT1Step>();
}


TEST(SDirkLStableTest,SolvesTestEq1){
	test_equation1<SDirkLStableStep>();
}

TEST(SDirkLStableTest,SolvesTestEq2){
	test_equation2<SDirkLStableStep>();
}
