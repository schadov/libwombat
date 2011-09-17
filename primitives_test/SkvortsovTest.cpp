#include <gtest/gtest.h>
#include "../primitives/types.h"
#include "../primitives/ode_solver.h"
#include "../primitives/RefBlas.h"
#include "../primitives/skvortsov.h"

#include "test_functions.h"

TEST(Skvortsov1Test,SolvesTestEq1){
	test_equation1<Skvortsov1Step>();
}

TEST(Skvortsov1Test,SolvesTestEq2){
	test_equation2<Skvortsov1Step>();
}

TEST(Skvortsov1Test,SolvesTestEq1CPU){
	test_equation1<Skvortsov1StepCPU>();
}

TEST(Skvortsov1Test,SolvesTestEq2CPU){
	test_equation2<Skvortsov1StepCPU>();
}

//////////////////////////////////////////////////////////////////////////

TEST(Skvortsov2Test,SolvesTestEq1){
	test_equation1<Skvortsov2Step>();
}

TEST(Skvortsov2Test,SolvesTestEq2){
	test_equation2<Skvortsov2Step>();
}

TEST(Skvortsov2Test,SolvesTestEq1CPU){
	test_equation1<Skvortsov2StepCPU>();
}

TEST(Skvortsov2Test,SolvesTestEq2CPU){
	test_equation2<Skvortsov2StepCPU>();
}

//////////////////////////////////////////////////////////////////////////

TEST(Skvortsov3Test,SolvesTestEq1CPU){
	test_equation1<Skvortsov3StepCPU>();
}

TEST(Skvortsov3Test,SolvesTestEq2CPU){
	test_equation2<Skvortsov3StepCPU>();
}

